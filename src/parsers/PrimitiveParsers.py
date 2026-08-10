'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import copy
import warnings
import yaml
try:
    from ruamel.yaml.scalarfloat import ScalarFloat
except Exception:
    ScalarFloat = None
import re
from datetime import date

from utilities.protocol_shapes import materialise_shapes, validate_trace_references
from param_id.modifier_funcs import (BUILTIN_MODIFIER_FUNCS, get_modifier_funcs,
                                     probe_affine)

try:
    from mpi4py import MPI
    mpi_available = True
    rank = MPI.COMM_WORLD.Get_rank()
except:
    mpi_available = False
    rank=0

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))


# ---------------------------------------------------------------------------
# Backend solver schema
# ---------------------------------------------------------------------------
# CasADi integrator methods that use SUNDIALS adjoint sensitivity for AD, which fails on a long
# warmup with CV_TOO_MUCH_WORK. The symbolic methods (bdf, semi_implicit_euler, collocation, rk)
# build a plain reverse-mode graph instead and handle nonzero pre_time fine, so they must NOT warn
# and ARE suitable for CasADi AD. Single source of truth for both warn_if_casadi_nonzero_pre_time
# and SOLVER_SCHEMA['ad_suitable_methods'] below, so the two cannot drift.
_CASADI_ADJOINT_METHODS = ('cvodes', 'idas')

# AADC solver methods whose forward integration the tape can record step-for-step. An adaptive
# integrator picks its step sizes from the state, so the sequence of operations changes with the
# parameters and cannot be replayed from a tape. Lives here (not in param_id/aadc_backend.py,
# which imports it) so the schema and the check that enforces it cannot drift, and so the
# dependency points one way: the backend depends on the schema, not the reverse.
AADC_TAPE_CONSISTENT_METHODS = ('rk4', 'implicit_euler_ift', 'semi_implicit', 'implicit_newton')

# The stiff BDF methods do not go through the standard replay tape at all: aadc_backend.
# cost_and_grad dispatches each to its own gradient implementation *before* the
# AADC_TAPE_CONSISTENT_METHODS check, so they are AD-capable without being members of it. Kept
# as a separate tuple rather than folded into the one above, because that one means "the standard
# tape can replay this step sequence" and these are not that -- but both are AD-capable, which is
# what ad_suitable_methods advertises.
AADC_BDF_AD_METHODS = ('bdf_newton', 'semi_implicit_signed')

# 'bdf_tape' and 'bdf_kernel' were never two integrators: both step the same signed
# semi-implicit scheme, differing only in where the loop runs (one AADC tape, or a C++ kernel
# replay that falls back to the tape when the extension is absent). They are now one method,
# 'semi_implicit_signed', with the execution choice in solver_info['gradient_strategy'].
# Accepted for configs written before the split (issue #346).
AADC_LEGACY_METHOD_ALIASES = {
    'bdf_tape': ('semi_implicit_signed', 'tape'),
    'bdf_kernel': ('semi_implicit_signed', 'kernel'),
}

# The methods aadc_python_solver_helper.run() can actually integrate. A tool building a
# "run a simulation" menu must use this rather than methods_by_solver, which is a superset
# (issue #346).
AADC_FORWARD_METHODS = ('adaptive_rk45', 'semi_implicit', 'semi_implicit_signed',
                        'implicit_euler_ift', 'implicit_newton', 'bdf_newton', 'rk4')

# Every AADC method that can produce an analytic gradient, by either route.
AADC_AD_METHODS = AADC_TAPE_CONSISTENT_METHODS + AADC_BDF_AD_METHODS

# Single source of truth for which generated model_types exist, which solvers are
# valid for each, and which methods/plugins are valid for each solver. Used for
# input validation here AND surfaced to downstream tools (e.g. the CUFLynx
# settings UI) so they don't hardcode these lists. Keep this in sync with
# solver_wrappers.get_simulation_helper.
SOLVER_SCHEMA = {
    'model_types': ['cellml_only', 'python', 'cpp', 'casadi_python', 'aadc_python', 'python_user_defined'],
    'solvers_by_model_type': {
        'cellml_only': ['CVODE_opencor', 'CVODE_myokit'],
        'python': ['solve_ivp'],
        'cpp': ['CVODE', 'RK4', 'PETSC'],
        'casadi_python': ['casadi_integrator'],
        'aadc_python': ['aadc_semi_implicit'],
        # The user supplies their own ODE wrapper in funcs_user/; it is integrated
        # by the shared SciPy PythonSimulationHelper (see solver_wrappers).
        'python_user_defined': ['user_defined'],
    },
    # Methods/plugins valid for each solver.
    'methods_by_solver': {
        'CVODE_opencor': ['CVODE'],
        'CVODE_myokit': ['CVODE'],
        'solve_ivp': ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'forward_euler'],
        'CVODE': ['CVODE'],
        'RK4': ['RK4'],
        'PETSC': ['PETSC'],
        # 'semi_implicit_euler' is a fixed-step damped scheme implemented in the
        # CasADi helper (not a SUNDIALS plugin); used for stiff models whose cvodes
        # adjoint-sensitivity gradient fails (e.g. 3compartment).
        # 'bdf' is a fixed-step implicit BDF (order 2, BDF1 startup) built as a symbolic
        # CasADi graph with a rootfinder per step; stable for stiff models and, unlike
        # cvodes adjoint sensitivity, fully supports CasADi AD (rootfinder is
        # differentiable via the implicit-function theorem).
        'casadi_integrator': ['cvodes', 'idas', 'collocation', 'rk', 'semi_implicit_euler', 'bdf'],
        # 'rk4' is fixed-step and is what the AADC tape records, so it is the method for which
        # the forward cost and the tape gradient are of the same function (see do_ad).
        # There is deliberately no 'bdf' here: that method handed the solve to scipy's
        # solve_ivp with AADC supplying only the RHS and Jacobian, so the trajectory never
        # reached the tape. The AD tape has no bdf branch either, so do_ad silently recorded
        # rk4 instead -- cost and gradient were different functions. Use 'semi_implicit' for
        # stiff models, or model_type 'casadi_python' for a differentiable symbolic BDF.
        'aadc_semi_implicit': ['adaptive_rk45', 'semi_implicit', 'semi_implicit_signed',
                               'implicit_euler_ift', 'implicit_newton', 'bdf_newton', 'rk4'],
        # The user wrapper supplies the rhs; the framework integrates it with the
        # same scipy solve_ivp methods as model_type 'python'.
        'user_defined': ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'forward_euler'],
    },
    # Default solver for each model_type (used when none is specified).
    'default_solver_by_model_type': {
        'cellml_only': 'CVODE_opencor',
        'python': 'solve_ivp',
        'cpp': 'CVODE',
        'casadi_python': 'casadi_integrator',
        'aadc_python': 'aadc_semi_implicit',
        'python_user_defined': 'user_defined',
    },
}

# Per-integrator suitability of the analytic-gradient backends, so a front-end (e.g. CUFLynx) can
# gate its "Gradient" menu on the *selected integrator*, not just model_type/solver -- issue #298.
#
# AD-suitable casadi_integrator methods are exactly those NOT using SUNDIALS adjoint sensitivity
# (cvodes/idas), whose adjoint integration fails on a long warmup (CV_TOO_MUCH_WORK); the symbolic
# methods (collocation, rk, semi_implicit_euler, bdf) are differentiated by reverse mode and fully
# support CasADi AD. Derived from _CASADI_ADJOINT_METHODS so the flag and the warning cannot drift.
# AD-suitable aadc_semi_implicit methods are exactly the fixed-step ones the tape can replay;
# 'adaptive_rk45' chooses its steps from the state, so the forward solve and the gradient would
# integrate different systems. Derived from AADC_TAPE_CONSISTENT_METHODS, which aadc_backend
# enforces, so the schema and the runtime check cannot drift (issue #336).
SOLVER_SCHEMA['ad_suitable_methods'] = {
    'casadi_integrator': [m for m in SOLVER_SCHEMA['methods_by_solver']['casadi_integrator']
                          if m not in _CASADI_ADJOINT_METHODS],
    'aadc_semi_implicit': [m for m in SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']
                           if m in AADC_AD_METHODS],
}
# Which methods each solver can run as a plain forward solve. Only aadc_semi_implicit currently
# differs from methods_by_solver -- see AADC_FORWARD_METHODS.
SOLVER_SCHEMA['forward_methods_by_solver'] = {
    solver: (list(AADC_FORWARD_METHODS) if solver == 'aadc_semi_implicit' else list(methods))
    for solver, methods in SOLVER_SCHEMA['methods_by_solver'].items()
}
# Myokit CVODES forward-sensitivity (FSA) is the analytic gradient for stiff cellml_only models;
# its method is 'CVODE' on the CVODE solvers. (CA's get_gradient currently produces FSA only for
# CVODE_myokit; the CVODE_opencor entry records that its FSA-capable method would likewise be
# CVODE, so a tool gating a method menu stays correct if/when it is wired up.)
SOLVER_SCHEMA['fsa_suitable_methods'] = {
    'CVODE_myokit': ['CVODE'],
    'CVODE_opencor': ['CVODE'],
}
# Which integrators can be trusted on a STIFF model. A tool offering a method menu for a stiff
# model (the cardiovascular ones are stiff) should restrict to these: the others either fail
# outright or, worse, return a plausible-looking trace that is badly wrong.
#
# The aadc_semi_implicit entry is measured, not assumed -- 3compartment, sim_time=0.2,
# pre_time=0, output heart/u_lv, against CVODE_myokit (7294 ... 7.171e4), from issue #346:
#
#   rk4                  OverflowError at dt 1e-3, 1e-4 and 1e-5
#   adaptive_rk45        no return; killed at 180 s for a 0.01 s horizon
#   semi_implicit        OK, +6.7%
#   implicit_newton      OK, -1.9%
#   implicit_euler_ift   OK but -84%  <- the dangerous one: it completes and looks plausible
#   bdf_newton           fails on floor() over an active idouble
#   semi_implicit_signed OK, +2%  (at its defaults: max_step 0.001, jac_lag 10)
#
# semi_implicit_signed was previously withheld from this list after measuring ~4.4x high while
# its forward integrator was being written. That measurement was taken from the model's own cold
# initial conditions with pre_time=0, where the whole window is a startup transient: every
# variant of the scheme -- and CVODE itself -- peaks near 3.1e5 there, so it showed a difference
# between two setups rather than between two integrators. Re-measured after a 5 s spin-up, the
# scheme's four distinguishing features (signed diagonal, lagged Jacobian, sub-stepping, no valve
# clamp) land it within 2% of CVODE_myokit, which is better than semi_implicit's +16% at dt=0.01.
# Its stability depends on max_step and jac_lag together, so both are documented above.
#
# implicit_euler_ift is deliberately excluded despite completing. It is still in
# ad_suitable_methods, so a gradient-based calibration will use it without complaint -- being
# wrong by a factor of six while returning a smooth trace is worse than raising. Why it is wrong
# is not yet established (issue #346).
#
# The others follow from the integrators themselves: CVODE is BDF-based; solve_ivp's stiff
# solvers are Radau/BDF/LSODA; CasADi's implicit methods (cvodes/idas/bdf/semi_implicit_euler)
# are stable where its explicit rk is not.
SOLVER_SCHEMA['stiff_suitable_methods'] = {
    'CVODE_myokit': ['CVODE'],
    'CVODE_opencor': ['CVODE'],
    'CVODE': ['CVODE'],
    'solve_ivp': ['Radau', 'BDF', 'LSODA'],
    'user_defined': ['Radau', 'BDF', 'LSODA'],
    'casadi_integrator': ['cvodes', 'idas', 'bdf', 'semi_implicit_euler'],
    'aadc_semi_implicit': ['semi_implicit', 'semi_implicit_signed', 'implicit_newton'],
    # Explicitly empty rather than absent: the cpp RK4 solver offers only a fixed-step explicit
    # method, and PETSC's plugin choice is not yet assessed against a stiff model. A consumer
    # must be able to tell "assessed, nothing qualifies" from "not in the table at all", so
    # every solver in methods_by_solver has an entry here (enforced by the schema tests).
    'RK4': [],
    'PETSC': [],
}

# Recommended default integrator per solver, for a front-end to pre-select an AD-friendly method:
# casadi_python -> 'bdf' (stable and AD-suitable) rather than the adjoint 'cvodes'. This is the
# value a tool (CUFLynx) should default its menu to; it is advisory and does not change CA's own
# internal fallback (a plain run without a method still uses the helper's default).
SOLVER_SCHEMA['default_method_by_solver'] = {
    'casadi_integrator': 'bdf',
    # 'implicit_newton', not 'rk4'. rk4 was chosen in #336 for being tape-consistent, without
    # checking it could integrate anything: on 3compartment it raises OverflowError at dt 1e-3,
    # 1e-4 and 1e-5, while implicit_newton lands within 2% of CVODE_myokit. A default has to be
    # able to produce a number first and be AD-friendly second (issue #346).
    'aadc_semi_implicit': 'implicit_newton',
}


# The integrator-specific `solver_info` settings each solver accepts, in schema form, so a tool
# (e.g. CUFLynx) can auto-populate the solver settings form when a solver is picked -- the
# companion to SOLVER_SCHEMA's solver/method menus. Each descriptor: `name` (the solver_info key),
# `type` ('int' | 'float' | 'bool' | 'str' | 'dict' | 'enum'), `default` (None => no built-in
# default; falls back to the integrator's own), `required`, `description`; enums add `choices`.
# The framework keys ('solver', 'method', 'dt_solver') are handled separately (method comes from
# SOLVER_SCHEMA['methods_by_solver']) and are not listed here. Defaults mirror
# get_solver_info_default(); this is the single source of truth for _SOLVER_INTEGRATOR_KEYS below.
_SI_RTOL = {'name': 'rtol', 'type': 'float', 'default': 1e-8, 'required': False,
            'description': 'Relative integration tolerance.'}
_SI_ATOL = {'name': 'atol', 'type': 'float', 'default': 1e-8, 'required': False,
            'description': 'Absolute integration tolerance.'}
# CVODE-family backends (opencor, and the cpp CVODE/RK4/PETSC) share the same fields.
_CVODE_FAMILY_SOLVER_INFO = [
    {'name': 'MaximumStep', 'type': 'float', 'default': 0.001, 'required': False,
     'description': 'Maximum integrator step size.'},
    {'name': 'MaximumNumberOfSteps', 'type': 'int', 'default': 5000, 'required': False,
     'description': 'Maximum number of internal integrator steps per output step.'},
    _SI_RTOL, _SI_ATOL,
]

# CVODE_myokit is deliberately NOT in that family. myokit.Simulation exposes only
# set_max_step_size / set_min_step_size / set_tolerance -- there is no max-step-count
# knob, so myokit_helper never reads MaximumNumberOfSteps. Advertising a setting the
# code never reads makes a downstream tool (e.g. CUFLynx) render a control that
# silently does nothing -- the same reasoning as the note on 'aadc_semi_implicit'
# below. Configs that already set it are migrated with a warning, not rejected; see
# migrate_legacy_solver_info_keys.
_MYOKIT_SOLVER_INFO = [
    {'name': 'MaximumStep', 'type': 'float', 'default': 0.001, 'required': False,
     'description': 'Maximum integrator step size (myokit Simulation.set_max_step_size).'},
    # rel 1e-6 / abs 1e-8, not the 1e-8/1e-8 the rest of the CVODE family declares. abs stays
    # at the 1e-8 floor previous users ran at, so existing models do not start failing; rel is
    # relaxed to 1e-6 because the relative knob is where most of the 1e-8/1e-8 interactive
    # solve cost was (1e-8/1e-8 measured ~2.3x slower than Myokit's own defaults on
    # 3compartment). Front-ends seed interactive solves from these declared defaults, and the
    # helper applies the same values when the user sets neither, so declared and effective
    # cannot drift (mirrored by myokit_helper.CA_DEFAULT_*; a test pins them equal). FSA still
    # forces 1e-8/1e-8 when the user set none -- a sloppy forward solve makes a poor gradient
    # (see myokit_helper.apply_cvodes_tolerances).
    {**_SI_RTOL, 'default': 1e-6},
    {**_SI_ATOL, 'default': 1e-8},
]

# Shared by 'solve_ivp' and 'user_defined': the user wrapper supplies only the RHS and is
# integrated by the same scipy solve_ivp helper, so the two accept an identical settings set.
# Shared rather than duplicated so they cannot drift apart.
_SOLVE_IVP_SOLVER_INFO = [
    _SI_RTOL, _SI_ATOL,
    {'name': 'max_step', 'type': 'float', 'default': 0.001, 'required': False,
     'description': 'Maximum step size passed to scipy.integrate.solve_ivp.'},
    {'name': 'vectorized', 'type': 'bool', 'default': False, 'required': False,
     'description': 'Whether the RHS accepts vectorised input (scipy solve_ivp option).'},
    {'name': 'dense_output', 'type': 'bool', 'default': False, 'required': False,
     'description': 'Whether to compute a continuous (dense) solution.'},
    {'name': 'jac', 'type': 'str', 'default': None, 'required': False,
     'description': 'Optional Jacobian specification for implicit methods.'},
]

SOLVER_INFO_FIELDS = {
    'CVODE_opencor': _CVODE_FAMILY_SOLVER_INFO,
    'CVODE_myokit': _MYOKIT_SOLVER_INFO,
    'CVODE': _CVODE_FAMILY_SOLVER_INFO,
    'RK4': _CVODE_FAMILY_SOLVER_INFO,
    'PETSC': _CVODE_FAMILY_SOLVER_INFO,
    'solve_ivp': _SOLVE_IVP_SOLVER_INFO,
    'user_defined': _SOLVE_IVP_SOLVER_INFO,
    'casadi_integrator': [
        {'name': 'max_step_size', 'type': 'float', 'default': 0.001, 'required': False,
         'description': 'Maximum step size for the adaptive CasADi integrators (cvodes/idas/etc).'},
        {'name': 'max_step', 'type': 'float', 'default': 0.001, 'required': False,
         'description': ('Internal sub-step cap for the symbolic bdf method (self.dt is split '
                         'into ceil(dt/max_step) implicit sub-steps); distinct from '
                         'max_step_size, which sizes the adaptive integrators.')},
        {'name': 'max_num_steps', 'type': 'int', 'default': 5000, 'required': False,
         'description': 'Maximum number of internal integrator steps.'},
        {'name': 'reltol', 'type': 'float', 'default': 1e-8, 'required': False,
         'description': 'Relative integration tolerance (SUNDIALS naming).'},
        {'name': 'abstol', 'type': 'float', 'default': 1e-10, 'required': False,
         'description': 'Absolute integration tolerance (SUNDIALS naming).'},
        {**_SI_RTOL, 'default': None,
         'description': 'Relative-tolerance alias (reltol is preferred for CasADi).'},
        {**_SI_ATOL, 'default': None,
         'description': 'Absolute-tolerance alias (abstol is preferred for CasADi).'},
        {'name': 'options', 'type': 'dict', 'default': None, 'required': False,
         'description': 'Extra options passed straight through to the CasADi integrator.'},
    ],
    'aadc_semi_implicit': [
        {'name': 'tol', 'type': 'float', 'default': 1e-8, 'required': False,
         'description': 'Integration tolerance for the adaptive AADC integrator.'},
        {'name': 'threads', 'type': 'int', 'default': 4, 'required': False,
         'description': 'Number of threads for AADC evaluation.'},
        {'name': 'max_step', 'type': 'float', 'default': 0.001, 'required': False,
         'description': "Internal sub-step cap for bdf_newton and semi_implicit_signed (default "
                        "0.001, matching CasADi BDF). The number of sub-steps per output step is "
                        "ceil(dt/max_step); raising it towards dt removes the sub-stepping that "
                        "makes a lagged Jacobian safe -- see jac_lag."},
        {'name': 'jac_lag', 'type': 'int', 'default': 10, 'required': False,
         'description': "How many sub-steps the signed scheme reuses one diagonal Jacobian for "
                        "before recomputing it (semi_implicit_signed, and the tape/kernel "
                        "gradients of the same scheme). Higher is faster and less stable, and "
                        "it is only safe in combination with sub-stepping: measured on "
                        "3compartment at dt=0.01, jac_lag=10 with no sub-stepping diverges "
                        "within 15 steps, while the same lag at max_step=0.001 (10 sub-steps) "
                        "stays within 2% of CVODE_myokit. Set to 1 to recompute every sub-step."},
        {'name': 'gradient_strategy', 'type': 'enum', 'default': 'tape', 'required': False,
         'choices': ['tape', 'kernel'],
         'description': "How method 'semi_implicit_signed' evaluates its gradient: 'tape' "
                        "records the whole integration on one AADC tape and replays it; "
                        "'kernel' replays a recorded kernel from C++ (faster, and falls back "
                        "to 'tape' when the C++ extension is not built). Same integration "
                        "either way -- this chooses where the loop runs, not what it solves. "
                        "Ignored by every other method."},
        # No 'gradient_method' here: nothing reads it. AD vs FD is chosen by the `do_ad` flag
        # (see SciPyMinimizeOptimiser.run, which falls back to approx_fprime when it is off),
        # and which AD backend runs follows from model_type/solver in
        # OpencorParamID.get_gradient. Advertising a setting the code never reads makes CUFLynx
        # render a control that silently does nothing.
    ],
}
# Expose the solver_info field schema alongside the solver/method menus for one-stop discovery.
SOLVER_SCHEMA['solver_info_fields_by_solver'] = SOLVER_INFO_FIELDS


# Option (setting) descriptors for the per-method `optimiser_options` blocks, so downstream tools
# (e.g. the CUFLynx settings UI) can auto-populate the correct settings fields when a calibration
# method is selected instead of hardcoding them. Each descriptor: `name` (the optimiser_options
# key), `type` ('int' | 'float' | 'bool' | 'enum'), `default` (None => must be supplied by the
# user; no built-in default), `required`, and a `description`; enum settings add `choices`. Keep
# in sync with the optimiser classes in param_id/optimisers.py.
_OPT_NUM_CALLS = {
    'name': 'num_calls_to_function', 'type': 'int', 'default': None, 'required': True,
    'description': 'Evaluation budget: maximum number of cost-function calls.',
}
_OPT_COST_CONVERGENCE = {
    'name': 'cost_convergence', 'type': 'float', 'default': 1e-4, 'required': False,
    'description': 'Stop once the cost drops below this value.',
}
_OPT_MAX_PATIENCE = {
    'name': 'max_patience', 'type': 'int', 'default': 10, 'required': False,
    'description': 'Stop after this many generations without an improvement in cost.',
}
# Genetic-algorithm population sizing. These are the single source of truth for the defaults --
# GeneticAlgorithmOptimiser._population_sizes() reads them from here rather than duplicating the
# numbers, so the schema and the code cannot drift and a front-end can pre-fill the real values
# (a None default would render as a blank field, see #277). DEBUG substitutes a documented
# quick-run scale-down, and that too is advertised here via `debug_default` -- the single source
# of truth GeneticAlgorithmOptimiser._debug_population() derives from, so a front-end can show/pass
# the exact values CA runs under DEBUG without hardcoding them (#313). An option with no DEBUG
# variant simply omits `debug_default`. The population per generation is
# num_survivors + num_survivors*num_mutations_per_survivor + num_cross_breed.
_OPT_GA_NUM_ELITE = {
    'name': 'num_elite', 'type': 'int', 'default': 12, 'debug_default': 4, 'required': False,
    'description': 'Genetic algorithm: top individuals carried over unchanged each generation '
                   '(elitism). Reduced to 4 when DEBUG is on.',
}
_OPT_GA_NUM_SURVIVORS = {
    'name': 'num_survivors', 'type': 'int', 'default': 48, 'debug_default': 6, 'required': False,
    'description': 'Genetic algorithm: individuals that survive to reproduce each generation. '
                   'Reduced to 6 when DEBUG is on.',
}
_OPT_GA_NUM_MUTATIONS_PER_SURVIVOR = {
    'name': 'num_mutations_per_survivor', 'type': 'int', 'default': 12, 'debug_default': 2,
    'required': False,
    'description': 'Genetic algorithm: mutated offspring generated per survivor each generation. '
                   'Reduced to 2 when DEBUG is on.',
}
_OPT_GA_NUM_CROSS_BREED = {
    'name': 'num_cross_breed', 'type': 'int', 'default': 120, 'debug_default': 10,
    'required': False,
    'description': 'Genetic algorithm: cross-bred (recombined) offspring per generation. '
                   'Reduced to 10 when DEBUG is on.',
}


# Single source of truth for the prior distributions a params_for_id `prior` column may name,
# i.e. the valid values of that column. Surfaced to downstream tools (e.g. the CUFLynx
# params_for_id editor) the same way PARAM_ID_METHODS is, so they can populate a prior picker
# without hardcoding the list. Keep in sync with OpencorParamID.get_lnprior_from_params().
#
# Declared rather than left implicit because an unrecognised value used to be *accepted*: the
# column was read straight to a numpy array, and get_lnprior_from_params matched it against
# 'uniform'/'exponential'/'normal' and fell through every branch when it matched none. Falling
# through skips that parameter's own range check, so a mis-spelled prior -- 'Normal', say --
# silently stopped bounding the parameter at all, and an MCMC walker could leave [min, max]
# with a finite lnprior instead of -inf. A typo must not quietly unbound a parameter.
# Each prior also declares the values it takes, in `params`. Those were previously hardcoded
# in get_lnprior_from_params -- the exponential's rate behind a "TODO make this user
# modifiable", the normal's mean and std behind a "temporarily" -- so a user who wanted a
# prior centred anywhere other than the middle of the range had no way to say so, and no way
# to discover that the number was fixed. Each entry names a params_for_id column, with the
# previous hardcoded value as its default so existing files are unaffected.
PARAM_PRIOR_TYPES = {
    'uniform': {
        'label': 'Uniform',
        'description': 'Flat across [min, max]. The default when no prior is given.',
        'params': [],
    },
    'exponential': {
        'label': 'Exponential',
        'description': ('Decays from prior_origin with scale prior_scale, favouring smaller '
                        'values. Truncated to [min, max] unless the parameter is unbounded.'),
        # One-sided: it decays away from its origin in one direction, so an unbounded
        # exponential's derived range runs from the origin rather than straddling it.
        'support': 'one_sided',
        'params': [
            {'name': 'prior_lambda', 'type': 'float', 'default': 1.0, 'positive': True,
             'role': 'rate',
             'description': ('Decay rate relative to max, the original parameterisation. '
                             'Only used when prior_scale is not given.')},
            {'name': 'prior_origin', 'type': 'float', 'default': 0.0, 'positive': False,
             'role': 'location', 'default_expr': '0',
             'description': 'Where the decay starts. Defaults to zero, as it always was.'},
            {'name': 'prior_scale', 'type': 'float', 'default': None, 'positive': True,
             'role': 'scale', 'default_expr': 'max / prior_lambda',
             'description': ('Decay scale, in the parameter\'s own units. Defaults to '
                             'max / prior_lambda, which reproduces the original rate. '
                             'Required when unbounded, since there is then no max.')},
        ],
    },
    'normal': {
        'label': 'Normal',
        'description': 'Gaussian, truncated to [min, max] unless the parameter is unbounded.',
        'support': 'symmetric',
        'params': [
            {'name': 'prior_mean', 'type': 'float', 'default': None, 'positive': False,
             'within_bounds': True, 'role': 'location',
             'default_expr': '(min + max) / 2',
             'description': 'Centre of the Gaussian. Defaults to the centre of [min, max].'},
            {'name': 'prior_std', 'type': 'float', 'default': None, 'positive': True,
             'role': 'scale', 'default_expr': '(max - min) / 6',
             'description': ('Standard deviation. Defaults to one sixth of the range, which '
                             'puts [min, max] at +/- 3 sigma.')},
        ],
    },
}

# The params_for_id column that marks a parameter as unbounded, i.e. having no min/max of its
# own: the prior defines where it lives, and the range CA needs for everything else is derived
# from that prior instead of typed.
PARAM_UNBOUNDED_COLUMN = 'unbounded'

# ---------------------------------------------------------------------------------------------
# params_for_id as JSON.
#
# The CSV is still readable and always will be, but it is converted to this structure on read, so
# there is exactly one code path behind the front door. The JSON shape exists because the CSV
# cannot express two things the user asked for: a group of parameters with *different* names
# (a CSV row has one param_name for all its vessels), and a parameter that modifies other
# parameters. `targets` -- a list of full component/param qnames -- removes the first restriction;
# the second arrives with modifier entries in a follow-up.
PARAMS_FOR_ID_JSON_VERSION = 1

# What a modifier parameter can do to the parameters it names. Exported as data, not hardcoded
# downstream: a front-end builds its menu by reading this, the same way it reads PARAM_PRIOR_TYPES
# and the cost-func registry, so adding an operation here is the only edit needed.
#
# `default_min`/`default_max` are advisory bounds for the UI. A scale multiplier is
# dimensionless, so unlike every other parameter its bounds are not physical values -- which is
# the most likely user error, and the reason the UI should be able to offer sane ones.
PARAM_MODIFIER_OPERATIONS = {
    'scale': {
        'description': 'one calibrated multiplier applied to every target\'s default value',
        'applies_to': 'value',
        'dimensionless': True,
        'default_min': 0.5,
        'default_max': 2.0,
        # The theta at which the operation leaves every target at its baseline. Local
        # sensitivity analysis evaluates there: theta's model-default is not a model value
        # (reading a target's default as theta would scale every target by it).
        'identity': 1.0,
    },
}
DEFAULT_PARAM_MODIFIER_OPERATION = 'scale'


def resolve_modifier_baselines(param_id_info, sim_helper):
    """Fill in each modifier's `baselines`, `resolved_inputs` and `affine` from the model, once.

    Resolved once at setup and never re-derived, because on some backends
    ``get_init_param_vals`` reads the live parameter array: after ``set_param_vals`` has written,
    it returns the value just written, not the model default. Re-deriving a scale baseline from
    it mid-calibration would apply theta to an already-scaled value and compound the factor every
    iteration -- theta=1.2 twice giving 1.44x. ``get_default_param_vals`` reads the frozen
    snapshot instead, and this runs before any parameter has been set.

    Idempotent: a modifier whose baselines are already resolved is left alone.
    """
    # param_id_info is not always set by the time a simulation helper exists -- several entry
    # points build the helper first and call set_param_id_info afterwards -- so this has to be a
    # no-op rather than an error when there is nothing to resolve yet. The later call from
    # set_param_id_info does the work in that case.
    if not param_id_info:
        return param_id_info
    modifiers = param_id_info.get("modifiers") or []
    if not modifiers:
        return param_id_info

    if not hasattr(sim_helper, 'get_default_param_vals'):
        raise NotImplementedError(
            f"{type(sim_helper).__name__} does not implement get_default_param_vals, which a "
            f"modifier parameter needs to resolve its baselines. A modifier applies "
            f"theta * baseline_i, and reading the baseline from the live parameter array would "
            f"compound the factor across calibration iterations.")

    funcs = get_modifier_funcs(param_id_info.get("modifier_funcs_external_path"))

    def _default(qname):
        value = sim_helper.get_default_param_vals([[qname]])[0]
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)

    for mod in modifiers:
        if mod.get("baselines") is None:
            raw = sim_helper.get_default_param_vals([[q] for q in mod["targets"]])
            baselines = []
            for value in raw:
                if isinstance(value, (list, tuple)):
                    value = value[0]
                baselines.append(float(value))
            mod["baselines"] = baselines

        fn = funcs.get(mod["operation"])
        if fn is None:
            raise ValueError(
                f"modifier '{mod['name']}' uses operation {mod['operation']!r}, which is not "
                f"in the modifier-function registry. Registered: {sorted(funcs)}.")

        # Declared inputs resolve to model defaults exactly once, like the baselines -- a
        # value read mid-calibration would already have been written by the optimiser.
        if mod.get("resolved_inputs") is None:
            resolved = {}
            for name, qnames in (mod.get("inputs") or {}).items():
                if isinstance(qnames, list):
                    resolved[name] = [_default(q) for q in qnames]
                else:
                    resolved[name] = _default(qnames)
            mod["resolved_inputs"] = resolved

        # p_i = a_i*theta + b_i, probed numerically per target. a_i is the constant
        # chain-rule weight the analytic gradients apply (dp_i/dtheta); a non-affine
        # function is refused here, before it can make a gradient silently wrong.
        if mod.get("affine") is None:
            a_list, b_list = [], []
            for baseline in mod["baselines"]:
                a, b = probe_affine(fn, baseline, mod["resolved_inputs"], mod["operation"])
                a_list.append(a)
                b_list.append(b)
            mod["affine"] = {"a": a_list, "b": b_list}
    return param_id_info


def expand_modifier_param_vals(param_id_info, param_vals):
    """Turn the optimiser's parameter vector into the values set_param_vals receives.

    A modifier occupies one slot in the vector (its theta) but names N model parameters, so its
    slot expands to N values, one per target. Everything else passes through untouched.

    The expansion is the whole of the modifier's arithmetic: the entry's modifier function is
    called per target, ``p_i = fn(theta, baseline_i, **resolved_inputs)`` -- ``scale`` is just
    the built-in ``theta * baseline_i``.

    N names against N values is paired positionally by pair_names_with_values (#376), which is
    why no backend needs to know modifiers exist.
    """
    modifiers = param_id_info.get("modifiers") or []
    if not modifiers:
        return list(param_vals)

    funcs = get_modifier_funcs(param_id_info.get("modifier_funcs_external_path"))
    by_index = {mod["index"]: mod for mod in modifiers}
    out = []
    for idx, value in enumerate(param_vals):
        mod = by_index.get(idx)
        if mod is None:
            out.append(value)
            continue
        baselines = mod.get("baselines")
        if baselines is None:
            raise ValueError(
                f"modifier '{mod['name']}' has no resolved baselines. Call "
                f"resolve_modifier_baselines(param_id_info, sim_helper) once at setup, before "
                f"any parameter has been written.")
        fn = funcs.get(mod["operation"])
        if fn is None:
            raise NotImplementedError(
                f"modifier operation {mod['operation']!r} is declared but not in the "
                f"modifier-function registry. Registered: {sorted(funcs)}.")
        declared = getattr(fn, 'modifier_inputs', {}) or {}
        resolved = mod.get("resolved_inputs")
        if declared and resolved is None:
            raise ValueError(
                f"modifier '{mod['name']}' has unresolved inputs {sorted(declared)}. Call "
                f"resolve_modifier_baselines(param_id_info, sim_helper) once at setup.")
        out.append([float(fn(float(value), b, **(resolved or {}))) for b in baselines])
    return out


def param_entry_labels(param_id_info):
    """One reporting label per calibrated variable (theta), for anything keyed by parameter.

    Reads ``param_labels`` (a grouped row joins its qnames, a modifier uses its own name), with
    the same fallback as ``sobolSA._param_labels`` for a param_id_info built by hand or loaded
    from before the key existed (#355). Sensitivities and their consumers key on this rather
    than on a row's first member: a grouped derivative is d/dtheta over all members, and
    labelling it with one member's qname would report it as a different quantity.
    """
    labels = param_id_info.get("param_labels")
    if labels is not None:
        return list(labels)
    return ['+'.join(n) if isinstance(n, (list, tuple)) else str(n)
            for n in param_id_info["param_names"]]


def apply_modifier_identity_nominals(param_id_info, nominal_vals):
    """Overwrite each modifier's slot in ``nominal_vals`` with the theta that leaves its
    targets at their baselines.

    A nominal parameter vector read from the model (``get_init_param_vals`` over first members)
    is right for ordinary and grouped rows, but a modifier's slot is theta, not a model value --
    the first target's default there would be fed to the modifier function as theta. The right
    nominal is the theta at which the targets sit at their model defaults: invert the affine
    mapping at the first target, ``theta0 = (baseline_0 - b_0) / a_0`` (for scale that is
    exactly the identity, 1.0). Multi-target entries whose targets would invert to different
    thetas keep the first target exact -- the others are as close as one theta allows, which
    is the entry's own approximation, not this function's. Falls back to the operation's
    static ``identity`` metadata for records without probed coefficients (built by hand or
    saved before the probe existed). Returns ``nominal_vals``, modified in place.
    """
    for mod in param_id_info.get("modifiers") or []:
        operation = mod.get("operation", DEFAULT_PARAM_MODIFIER_OPERATION)
        affine = mod.get("affine")
        baselines = mod.get("baselines")
        if affine is not None and baselines:
            a0, b0 = float(affine["a"][0]), float(affine["b"][0])
            if a0 == 0.0:
                raise ValueError(
                    f"modifier '{mod['name']}' ({operation!r}) has dp/dtheta = 0 at its first "
                    f"target -- theta does not move it, so no nominal theta exists and "
                    f"calibrating it is meaningless.")
            nominal_vals[mod["index"]] = (float(baselines[0]) - b0) / a0
            continue
        meta = PARAM_MODIFIER_OPERATIONS.get(operation)
        if meta is None or 'identity' not in meta:
            raise NotImplementedError(
                f"modifier operation {operation!r} has no affine coefficients resolved and no "
                f"identity value; cannot choose a nominal theta for '{mod['name']}'. Call "
                f"resolve_modifier_baselines(param_id_info, sim_helper) first.")
        nominal_vals[mod["index"]] = meta['identity']
    return nominal_vals


def modifier_weights_by_index(param_id_info):
    """Per-entry chain-rule weights: ``{entry_index: [w_i per member]}`` for modifier entries.

    A calibrated theta that governs several model parameters has
    ``d(anything)/d(theta) = sum_i w_i * d(anything)/d(p_i)`` with ``w_i = dp_i/dtheta``. Every
    modifier function is affine in theta (``p_i = a_i*theta + b_i``, enforced by the probe at
    resolve time), so ``w_i = a_i`` -- for the built-in scale that is the baseline itself.
    Shared-value groups (``w_i = 1``) are not in the map -- absence means unit weights. Raises
    if a modifier is unresolved, because a weight guessed at is a gradient silently wrong.
    """
    out = {}
    for mod in param_id_info.get("modifiers") or []:
        baselines = mod.get("baselines")
        if baselines is None:
            raise ValueError(
                f"modifier '{mod['name']}' has no resolved baselines. Call "
                f"resolve_modifier_baselines(param_id_info, sim_helper) once at setup, before "
                f"any parameter has been written.")
        affine = mod.get("affine")
        if affine is not None:
            out[mod["index"]] = [float(a) for a in affine["a"]]
        elif mod.get("operation", DEFAULT_PARAM_MODIFIER_OPERATION) == 'scale':
            # A record from before the affine probe existed (or built by hand): for scale,
            # dp_i/dtheta is the baseline itself, so the old behaviour is still exact.
            out[mod["index"]] = [float(b) for b in baselines]
        else:
            raise ValueError(
                f"modifier '{mod['name']}' ({mod['operation']!r}) has no affine coefficients. "
                f"Call resolve_modifier_baselines(param_id_info, sim_helper) once at setup.")
    return out


def save_param_modifiers(param_id_info, output_dir):
    """Write ``param_modifiers.json`` to ``output_dir`` (rank 0 only; no-op without modifiers).

    A scale result is uninterpretable without this record -- best_param_vals holds theta, and
    theta alone does not say what any model parameter ended up at. Recording the baselines
    means reproducing a result does not depend on the model file being unchanged. Callable at
    any point (idempotent overwrite): the parse-time save happens before baselines can be
    resolved, so the calibration run saves again once they are.
    """
    modifiers = param_id_info.get("modifiers") or []
    if modifiers and rank == 0:
        modifiers_path = os.path.join(output_dir, 'param_modifiers.json')
        with open(modifiers_path, 'w') as f:
            json.dump(modifiers, f, indent=2)


def param_modifier_operations(external_path=None):
    """The modifier operations available to params_for_id entries, as introspectable data.

    One record per registered modifier function -- built-ins, ``funcs_user`` and (when
    ``external_path`` is given) an external file -- each carrying ``description``,
    ``applies_to``, ``inputs`` (``{name: 'float'|'list'}``: what the entry must supply
    qnames for) and ``user_defined``. Built-ins with static UI metadata (scale's
    ``default_min``/``default_max``/``dimensionless``/``identity``) keep those keys. A
    front-end builds its modifier form from this rather than hardcoding CA's vocabulary,
    the same way it reads the cost-func registry.
    """
    out = {}
    for name, fn in get_modifier_funcs(external_path).items():
        meta = {
            'description': getattr(fn, 'modifier_description', name),
            'applies_to': 'value',
            'inputs': dict(getattr(fn, 'modifier_inputs', {}) or {}),
            'user_defined': BUILTIN_MODIFIER_FUNCS.get(name) is not fn,
        }
        static = PARAM_MODIFIER_OPERATIONS.get(name)
        if static is not None and BUILTIN_MODIFIER_FUNCS.get(name) is fn:
            meta = {**static, **meta, 'description': static.get('description',
                                                               meta['description'])}
        out[name] = meta
    return out


# Keys an entry may carry. Anything else is a typo and is refused, on the same reasoning as
# operation_kwargs/cost_kwargs: a key nothing reads changes nothing and gives no sign it was
# ignored.
PARAMS_FOR_ID_ENTRY_KEYS = frozenset({
    'name', 'targets', 'param_type', 'min', 'max', 'name_for_plotting',
    'prior', 'prior_params', PARAM_UNBOUNDED_COLUMN, 'comment',
    # A modifier entry names the parameters it acts on with `modifies` instead of `targets`, and
    # says what it does to them with `operation` -- the name of a registered modifier function
    # (built-in, funcs_user, or modifier_funcs_external_path). `inputs` supplies the model
    # qname(s) for each input the function declares; their *default* values are what the
    # function receives. min/max/prior belong to the modifier's own calibrated value, not to
    # the parameters it modifies.
    'modifies', 'operation', 'inputs',
})


# How wide the derived range is, in standard deviations either side of the prior's centre.
# Five puts ~1 sample in 3.5 million outside it, so the box is not meaningfully constraining
# while staying finite -- which it must be. min/max are not only the prior's truncation: they
# are the optimiser's search box, the Sobol sampling range, the denominator of the parameter
# normalisation, and the fallback finite-difference step. An actually infinite range makes the
# normalisation NaN and every calibration with it, so "unbounded" has to mean "wide enough not
# to bind", not "absent".
UNBOUNDED_SIGMA_SPAN = 5.0


def eval_prior_default(expr, bounds=None, params=None):
    """Evaluate a ``default_expr`` against a row's bounds and its other prior params.

    The expressions live in PARAM_PRIOR_TYPES so there is one statement of what a
    blank field means -- CA computes the default from it, and a downstream editor
    shows the same number in the field's placeholder. Previously the formulas were
    written twice: once in get_lnprior_from_params and once in whatever prose a UI
    chose, which is how "the centre of the range" and the value actually used drift
    apart.

    Deliberately tiny: names (min/max/other params), numbers, and arithmetic. No
    calls, no attributes, no comprehensions -- these come from CA's own schema, but
    an evaluator that can only do arithmetic cannot become something else later.
    Returns None when a name it needs is unavailable (an unbounded row has no max).
    """
    import ast as _ast

    names = dict(bounds or {})
    names.update({k: v for k, v in (params or {}).items() if v is not None})

    def _ev(node):
        if isinstance(node, _ast.Expression):
            return _ev(node.body)
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError('non-numeric constant')
        if isinstance(node, _ast.Name):
            if node.id not in names or names[node.id] is None:
                raise KeyError(node.id)
            return float(names[node.id])
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
            v = _ev(node.operand)
            return v if isinstance(node.op, _ast.UAdd) else -v
        if isinstance(node, _ast.BinOp) and isinstance(
                node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
            a, b = _ev(node.left), _ev(node.right)
            if isinstance(node.op, _ast.Add):
                return a + b
            if isinstance(node.op, _ast.Sub):
                return a - b
            if isinstance(node.op, _ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError
            return a / b
        raise ValueError(f'unsupported expression node {type(node).__name__}')

    try:
        value = _ev(_ast.parse(str(expr), mode='eval'))
    except (KeyError, ZeroDivisionError, ValueError, SyntaxError, TypeError):
        return None
    return value if np.isfinite(value) else None


def prior_param_default(prior_type, name, bounds=None, params=None):
    """The value a blank ``name`` takes for ``prior_type``: its literal default, or
    the one derived from the row's bounds. None when it cannot be derived."""
    for spec in PARAM_PRIOR_TYPES.get(prior_type, {}).get('params', []):
        if spec['name'] != name:
            continue
        if spec.get('default_expr'):
            derived = eval_prior_default(spec['default_expr'], bounds, params)
            if derived is not None:
                return derived
        return spec.get('default')
    return None


def _truthy_flag(value):
    """Whether a params_for_id boolean cell is set.

    Spelled several ways in the wild -- 1, true, TRUE, yes, y -- and blank/NaN is
    not set. Anything unrecognised is an error rather than a quiet False: a cell
    the user filled in must not be ignored.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ('', 'nan'):
        return False
    if text in ('1', '1.0', 'true', 'yes', 'y'):
        return True
    if text in ('0', '0.0', 'false', 'no', 'n'):
        return False
    raise ValueError(
        f"'{PARAM_UNBOUNDED_COLUMN}' must be true/false (or 1/0, yes/no), got {value!r}.")


def _prior_param_by_role(prior_type, role):
    for spec in PARAM_PRIOR_TYPES.get(prior_type, {}).get('params', []):
        if spec.get('role') == role:
            return spec['name']
    return None


def prior_supports_unbounded(prior_type):
    """Whether a prior can stand in for a parameter's range.

    It can when it declares both where it sits and how wide it is -- a location
    and a scale -- because those are what the derived range is built from. A
    uniform has neither and is *defined* by the range, so it cannot; an
    exponential has a rate but no location, so there is no centre to build
    around.
    """
    return bool(_prior_param_by_role(prior_type, 'location')
                and _prior_param_by_role(prior_type, 'scale'))


def derive_bounds_from_prior(prior_type, params, row_idx=None):
    """``(min, max)`` for an unbounded parameter, from its prior's own centre and width.

    ``params`` is the validated output of :func:`normalise_prior_params`. Both the
    location and the scale must be stated: their usual defaults are derived *from*
    the range, so leaving them out here would be circular.
    """
    where = '' if row_idx is None else f' (params_for_id row {row_idx})'
    if not prior_supports_unbounded(prior_type):
        supported = ', '.join(sorted(p for p in PARAM_PRIOR_TYPES if prior_supports_unbounded(p)))
        raise ValueError(
            f"'{PARAM_UNBOUNDED_COLUMN}' is set{where}, but the prior is '{prior_type}', which "
            f"has no centre and width to derive a range from. Priors that do: {supported}.")

    loc_name = _prior_param_by_role(prior_type, 'location')
    scale_name = _prior_param_by_role(prior_type, 'scale')
    loc, scale = params.get(loc_name), params.get(scale_name)
    missing = [n for n, v in ((loc_name, loc), (scale_name, scale)) if v is None]
    if missing:
        raise ValueError(
            f"'{PARAM_UNBOUNDED_COLUMN}' is set{where}, so {' and '.join(missing)} must be "
            f"given: without a range there is nothing left to derive them from.")

    span = UNBOUNDED_SIGMA_SPAN * float(scale)
    # A symmetric prior straddles its centre; a one-sided one decays away from its
    # origin in a single direction, so a range centred on that origin would put half
    # the box where the prior has no mass at all.
    if PARAM_PRIOR_TYPES[prior_type].get('support') == 'one_sided':
        return (float(loc), float(loc) + span)
    return (float(loc) - span, float(loc) + span)

# Every `params` entry above names a params_for_id column. Collected once so the parser can
# tell a prior hyper-parameter column from an unrelated one, and so a downstream tool can
# render exactly the fields the chosen prior uses.
PARAM_PRIOR_PARAM_NAMES = tuple(
    spec['name'] for meta in PARAM_PRIOR_TYPES.values() for spec in meta['params']
)

# The CSV columns that become prior_params entries in the JSON structure rather than top-level
# keys. Derived from the prior declarations so the converter cannot miss a newly added
# hyper-parameter (params_for_id JSON, issue #355 follow-up).
PARAMS_FOR_ID_CSV_PRIOR_COLUMNS = tuple(PARAM_PRIOR_PARAM_NAMES)

# What an absent, blank or NaN `prior` entry means -- and what the whole column defaults to
# when params_for_id has no `prior` at all.
DEFAULT_PARAM_PRIOR_TYPE = 'uniform'


def normalise_prior_type(value, row_idx=None):
    """The canonical ``PARAM_PRIOR_TYPES`` key for one ``prior`` cell.

    Blank/NaN means the default. Surrounding whitespace and letter case are
    normalised, so a hand-written 'Normal' resolves to 'normal' instead of
    unbounding the parameter. Anything genuinely unrecognised raises, because
    the alternative -- the historical behaviour -- was to accept it and silently
    drop that parameter's range check.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return DEFAULT_PARAM_PRIOR_TYPE
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return DEFAULT_PARAM_PRIOR_TYPE
    key = text.lower()
    if key not in PARAM_PRIOR_TYPES:
        where = '' if row_idx is None else f' (params_for_id row {row_idx})'
        raise ValueError(
            f"unknown prior '{text}'{where}. Valid priors are: "
            f"{', '.join(sorted(PARAM_PRIOR_TYPES))}."
        )
    return key


def normalise_prior_params(prior_type, row, row_idx=None):
    """The hyper-parameters for one row's prior, validated against its declaration.

    ``row`` is anything supporting ``.get(name)`` -- a DataFrame row or a plain
    dict. Returns ``{name: float or None}`` for exactly the values this prior
    declares; None means "not stated", which the likelihood turns into the
    documented default (the centre of the range, a sixth of the range) since
    those depend on bounds this function does not own.

    A value supplied for a prior that does not take it is an error, not something
    to ignore: `prior_std` on a uniform row means the user believes they set a
    width, and silently dropping it gives them a different posterior than the one
    they asked for.

    A value declared ``within_bounds`` is checked against the row's own min/max
    when the row carries them. Every prior is truncated to [min, max], so a centre
    outside it describes a peak the sampler can never reach: every draw sits on a
    tail and is pulled to the nearer bound. That is legal arithmetic and almost
    never what was meant, so it is refused here rather than run.
    """
    where = '' if row_idx is None else f' (params_for_id row {row_idx})'
    declared = {spec['name']: spec for spec in PARAM_PRIOR_TYPES[prior_type]['params']}

    for name in PARAM_PRIOR_PARAM_NAMES:
        if name in declared:
            continue
        raw = row.get(name) if hasattr(row, 'get') else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
            continue
        raise ValueError(
            f"'{name}' was given{where}, but the prior is '{prior_type}', which does not use "
            f"it. {prior_type} takes: {', '.join(declared) if declared else 'no parameters'}."
        )

    out = {}
    for name, spec in declared.items():
        raw = row.get(name) if hasattr(row, 'get') else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
            out[name] = spec['default']
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{name}'{where} must be a number, got {raw!r}.") from exc
        if not np.isfinite(value):
            raise ValueError(f"'{name}'{where} must be finite, got {raw!r}.")
        if spec['positive'] and value <= 0:
            raise ValueError(
                f"'{name}'{where} must be greater than zero, got {value}.")
        if spec.get('within_bounds'):
            bounds = _row_bounds(row)
            if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                raise ValueError(
                    f"'{name}'{where} must lie within the parameter's range "
                    f"[{bounds[0]}, {bounds[1]}], got {value}. Every prior is truncated to "
                    f"that range, so a centre outside it is a peak the sampler can never "
                    f"reach.")
        out[name] = value
    return out


def _row_bounds(row):
    """``(min, max)`` for a params_for_id row, or None when it does not carry them.

    None rather than an error: this function is also called with a bare dict of
    hyper-parameters (a downstream editor validating one row's fields), and a
    caller that cannot supply bounds should still get every other check.
    """
    try:
        lo = row.get('min') if hasattr(row, 'get') else None
        hi = row.get('max') if hasattr(row, 'get') else None
        if lo is None or hi is None:
            return None
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (lo, hi)


# Single source of truth for the parameter-identification (calibration) methods, i.e. the valid
# values of `param_id_method`. Surfaced to downstream tools (e.g. the CUFLynx settings UI) the
# same way SOLVER_SCHEMA is, so they can populate a calibration-method menu AND the per-method
# settings form without hardcoding either. Each method's `options` lists the `optimiser_options`
# keys it reads (see _OPT_* above). Keep in sync with OpencorParamID.run()'s param_id_method
# dispatch (paramID.py) and the optimiser classes (optimisers.py).
PARAM_ID_METHODS = {
    'genetic_algorithm': {
        'label': 'Genetic algorithm',
        'gradient_based': False,
        'description': 'Gradient-free population-based global search.',
        'options': [_OPT_NUM_CALLS, _OPT_COST_CONVERGENCE, _OPT_MAX_PATIENCE,
                    _OPT_GA_NUM_ELITE, _OPT_GA_NUM_SURVIVORS,
                    _OPT_GA_NUM_MUTATIONS_PER_SURVIVOR, _OPT_GA_NUM_CROSS_BREED],
    },
    'CMA-ES': {
        'label': 'CMA-ES',
        'aliases': ['CMAES', 'cmaes'],
        'gradient_based': False,
        'description': 'Covariance-matrix-adaptation evolution strategy (gradient-free).',
        'options': [
            # CMA-ES falls back to a 10000-call budget when none is given (GA requires one).
            {**_OPT_NUM_CALLS, 'default': 10000, 'required': False},
            {'name': 'sigma0', 'type': 'float', 'default': None, 'required': False,
             'description': ('Initial CMA-ES step size (standard deviation) in normalised '
                             'parameter space; omitted lets CMA choose.')},
            _OPT_COST_CONVERGENCE, _OPT_MAX_PATIENCE,
        ],
    },
    'bayesian': {
        'label': 'Bayesian optimisation',
        'gradient_based': False,
        'description': 'Surrogate-model Bayesian optimisation (experimental / untested).',
        # The acquisition-function constants (acq_func, n_initial_points, random_state) are
        # currently hardcoded in paramID.py, not user-configurable, so they are not listed here.
        # Bayesian falls back to a 10000-call budget when none is given (param_id_run_script.py);
        # only the genetic algorithm genuinely requires it, so advertise the default here (#277).
        'options': [{**_OPT_NUM_CALLS, 'default': 10000, 'required': False}],
    },
    'sp_minimize': {
        'label': 'Gradient descent (L-BFGS-B)',
        'gradient_based': True,
        'description': ('Local bounded L-BFGS-B. Uses an automatic-differentiation gradient for '
                        'casadi_python, aadc_python, or cellml_only + CVODE_myokit + do_ad; '
                        'finite differences otherwise.'),
        # The gradient source is the top-level `do_ad` user input, not an optimiser_option; the
        # sources available for a given model_type/solver are exposed by gradient_sources().
        'options': [_OPT_COST_CONVERGENCE],
    },
    'multi_start_sp_minimize': {
        'label': 'Multi-start gradient descent',
        'gradient_based': True,
        'description': ('L-BFGS-B from many scattered starts, so it exploits the gradient while '
                        'still escaping local minima. Same AD gradient sources as sp_minimize.'),
        'options': [
            {'name': 'num_starts', 'type': 'int', 'default': 10, 'required': False,
             'description': 'Number of L-BFGS-B starts scattered across the parameter box '
                            '(defaults to 4 when DEBUG is on).'},
            {'name': 'start_sampling', 'type': 'enum', 'default': 'sobol', 'required': False,
             'choices': ['sobol', 'latin_hypercube', 'random'],
             'description': 'How the start points are sampled across the parameter box.'},
            {'name': 'include_init_point', 'type': 'bool', 'default': True, 'required': False,
             'description': 'Include the parameters-CSV x0 as one of the starts.'},
            {'name': 'seed', 'type': 'int', 'default': 0, 'required': False,
             'description': 'Seed for the deterministic start sampling.'},
            {'name': 'fd_step', 'type': 'float', 'default': 1e-4, 'required': False,
             'description': 'Finite-difference step used when no AD gradient is available.'},
            {'name': 'no_new_starts_on_convergence', 'type': 'bool', 'default': True,
             'required': False,
             'description': 'Stop launching new starts once one has converged.'},
            {'name': 'convergence_cluster_tol_frac', 'type': 'float', 'default': 0.02,
             'required': False,
             'description': ('Fraction of each parameter range within which two minima are '
                             'treated as the same cluster.')},
            _OPT_COST_CONVERGENCE,
        ],
    },
}


def valid_param_id_methods():
    """All accepted `param_id_method` strings: canonical names plus their aliases."""
    names = []
    for canonical, meta in PARAM_ID_METHODS.items():
        names.append(canonical)
        names.extend(meta.get('aliases', []))
    return names


def param_id_method_options(param_id_method):
    """The `optimiser_options` settings a given `param_id_method` accepts (aliases resolved), for
    tools that auto-populate a per-method settings form. Returns the list of option descriptor
    dicts (see PARAM_ID_METHODS); an empty list for an unknown method."""
    for canonical, meta in PARAM_ID_METHODS.items():
        if param_id_method == canonical or param_id_method in meta.get('aliases', []):
            return meta.get('options', [])
    return []


def solver_info_fields(solver):
    """The `solver_info` settings a given solver accepts (see SOLVER_INFO_FIELDS), for tools that
    auto-populate the solver settings form. Empty list for an unknown solver."""
    return SOLVER_INFO_FIELDS.get(solver, [])


def gradient_sources(model_type, solver=None, method=None):
    """The gradient sources available for the gradient-based param-id methods (`sp_minimize`,
    `multi_start_sp_minimize`) with this `model_type` + `solver` (+ optional integrator `method`),
    for a front-end that offers a "Gradient" menu without hardcoding CA's rules.

    Each descriptor:
      * ``value``  -- 'FD' | 'AD' | 'FSA' (the UI selector value)
      * ``label``  -- human-readable name
      * ``do_ad``  -- the top-level ``do_ad`` user-input flag CA needs for this source. There is
                      no per-method "gradient" option in CA: AD vs finite differences is chosen by
                      ``do_ad``, and which analytic backend runs follows from model_type/solver.
                      Finite differences is ``do_ad`` False; every analytic source is ``do_ad`` True.
      * ``requires_all_differentiable`` -- True only for CasADi AD, which needs every operation in
                      the specific model to be differentiable. That is a per-model runtime property
                      (not knowable from model_type/solver alone), so a caller that has loaded the
                      model should gate the source on it -- e.g. via ``is_circulatory_differentiable``
                      over the cost/operation funcs. All other sources are False.
      * ``description``

    The analytic source, if any, follows exactly ``OpencorParamID.get_gradient`` dispatch (and
    ``AD_GRADIENT_MODEL_TYPES`` / ``fsa_gradient_available`` in the optimisers):
      * ``casadi_python``               -> symbolic CasADi AD
      * ``aadc_python``                 -> AADC tape AD (needs a Matlogica licence at runtime)
      * ``cellml_only`` + ``CVODE_myokit`` -> Myokit CVODES forward sensitivity (FSA)
      * otherwise                       -> finite differences only
    Finite differences is always available. This is the single source of truth these rules were
    previously duplicated from in downstream tools (e.g. CUFLynx); keep it in step with
    get_gradient.

    When ``method`` (the integrator plugin) is given, the analytic source is additionally gated on
    per-integrator suitability (``SOLVER_SCHEMA['ad_suitable_methods']`` /
    ``['fsa_suitable_methods']``): the CasADi AD descriptor is omitted for a casadi_integrator
    method that uses adjoint sensitivity (cvodes/idas, which can't produce a usable AD gradient),
    and FSA is omitted for a non-FSA method. ``method=None`` leaves the source ungated. Issue #298.
    """
    sources = [{
        'value': 'FD', 'label': 'Finite difference', 'do_ad': False,
        'requires_all_differentiable': False,
        'description': ('Central finite differences of the cost; works for any model_type, at '
                        'the cost of extra simulations per gradient.'),
    }]
    if model_type == 'casadi_python':
        casadi_methods = SOLVER_SCHEMA['methods_by_solver']['casadi_integrator']
        ad_suitable = SOLVER_SCHEMA['ad_suitable_methods']['casadi_integrator']
        # Gate only a *known* casadi_integrator method that is AD-unsuitable (cvodes/idas); an
        # unknown or unspecified method leaves AD offered.
        if method is None or method not in casadi_methods or method in ad_suitable:
            sources.append({
                'value': 'AD', 'label': 'Automatic differentiation (CasADi)', 'do_ad': True,
                'requires_all_differentiable': True,
                'description': ('Exact symbolic CasADi gradient. Requires every operation in the '
                                'model to be differentiable.'),
            })
    elif model_type == 'aadc_python':
        aadc_methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']
        aadc_ad_suitable = SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']
        # Gate only a *known* aadc method the tape cannot record (adaptive_rk45); an unknown or
        # unspecified method leaves AD offered, matching the casadi branch above.
        if method is not None and method in aadc_methods and method not in aadc_ad_suitable:
            return sources
        sources.append({
            'value': 'AD', 'label': 'Automatic differentiation (AADC)', 'do_ad': True,
            'requires_all_differentiable': False,
            'description': 'Exact AADC tape gradient (requires a Matlogica AADC licence at runtime).',
        })
    elif model_type == 'cellml_only' and solver == 'CVODE_myokit':
        solver_methods = SOLVER_SCHEMA['methods_by_solver'].get(solver, [])
        fsa_suitable = SOLVER_SCHEMA['fsa_suitable_methods'].get(solver, [])
        if method is None or method not in solver_methods or method in fsa_suitable:
            sources.append({
                'value': 'FSA', 'label': 'Forward sensitivity (Myokit CVODES)', 'do_ad': True,
                'requires_all_differentiable': False,
                'description': ('Myokit CVODES forward-sensitivity gradient; the analytic gradient '
                                'path for stiff cellml_only models.'),
            })
    return sources


# Settings blocks for the non-calibration analysis modes (sensitivity, MCMC, identifiability), in
# the same descriptor shape as a param_id method's `options`, so a tool can auto-populate their
# settings forms too. Each entry gives the enabling top-level flag (`enable_flag`), the
# user_inputs key holding the block (`options_key`), and the option descriptors the mode reads.
# Keep in sync with sensitivityAnalysis.py / paramID.py (MCMC) / identifiabilityAnalysis.py.
ANALYSIS_OPTIONS = {
    'sensitivity_analysis': {
        'label': 'Sobol sensitivity analysis',
        'enable_flag': 'do_sensitivity',
        'options_key': 'sa_options',
        'options': [
            {'name': 'method', 'type': 'enum', 'default': 'sobol', 'required': False,
             'choices': ['sobol', 'local'],
             'description': ('Sensitivity method: global variance-based Sobol indices, or '
                             '"local" derivative-based sensitivities (see gradient_method '
                             'for which backends can produce them).')},
            # Only read by method 'local'. Declared so downstream tools offer the choice
            # rather than hardcoding it, and so FD is something the user picks by name: it
            # costs 2M simulations and its accuracy depends on a step size the analytic arms
            # do not have, so it must never stand in silently for them. The arms carry their
            # own names -- AD / FSA, the same vocabulary as gradient_sources() and the Laplace
            # gradient_source -- because only an explicit name can be offered, disabled, or
            # reported back by a UI; each validates against the backend rather than being
            # silently reinterpreted. 'analytic' remains accepted in code as a legacy spelling
            # of 'auto' but is no longer advertised.
            {'name': 'gradient_method', 'type': 'enum', 'default': 'auto', 'required': False,
             'choices': ['auto', 'AD', 'FSA', 'FD'],
             'description': ('For method "local": how to differentiate. "auto" picks the '
                             "backend's own analytic arm and fails where there is none; "
                             '"AD" is the exact CasADi jacobian (casadi_python); "FSA" is '
                             'Myokit CVODES forward sensitivities (cellml_only + '
                             'CVODE_myokit + do_ad); "FD" is central finite differences, '
                             'which works on any backend that runs a forward simulation, at '
                             '2M simulations for M parameters.')},
            # Only read by method 'local' with gradient_method 'FD'. Declared because it is
            # not a tuning detail: on Lotka-Volterra, moving it from 1e-3 to 1e-2 changes a
            # sensitivity coefficient by up to 48%, since `max` of an oscillating trace is a
            # rough functional. A number that swings the answer that far belongs to the user.
            {'name': 'fd_rel_step', 'type': 'float', 'default': 1e-3, 'required': False,
             'description': ('For method "local" with gradient_method "FD": the finite-'
                             'difference step, relative to each parameter. Too large and it '
                             'measures curvature rather than the derivative; too small and '
                             'solver noise dominates.')},
            # enum, not str: sobol_SA.generate_samples dispatches on exactly these two
            # and raises ValueError on anything else, so a free string only lets a
            # typo through to run time.
            {'name': 'sample_type', 'type': 'enum', 'default': 'saltelli', 'required': False,
             'choices': ['saltelli', 'sobol'],
             'description': 'SALib sampling scheme: saltelli or sobol.'},
            # Real default lives in the fill-in block below (sets 32 when absent), so the schema
            # must advertise it rather than an empty required field (see issue #277).
            {'name': 'num_samples', 'type': 'int', 'default': 32, 'required': False,
             'description': ('Base sample count; the actual number of runs is num_samples*(2M+2) '
                             'for Sobol, where M is the number of parameters.')},
        ],
    },
    'mcmc': {
        'label': 'MCMC posterior sampling',
        'enable_flag': 'do_mcmc',
        'options_key': 'mcmc_options',
        'options': [
            {'name': 'num_steps', 'type': 'int', 'default': 5000, 'required': False,
             'description': 'Number of MCMC steps per walker.'},
            {'name': 'num_walkers', 'type': 'int', 'default': None, 'required': False,
             'description': 'Number of ensemble walkers (defaults to 2 * number of parameters).'},
        ],
    },
    'identifiability_analysis': {
        'label': 'Identifiability analysis',
        'enable_flag': 'do_ia',
        'options_key': 'ia_options',
        'options': [
            {'name': 'method', 'type': 'enum', 'default': 'Laplace', 'required': True,
             'choices': ['Laplace', 'profile_likelihood'],
             'description': 'Identifiability method: Laplace approximation or profile likelihood.'},
            # The source for the Laplace Hessian. 'FD' uses sub_method below (a finite-difference
            # Hessian of the log-posterior). 'AD'/'FSA' build the Fisher information matrix
            # J^T diag(1/std^2) J from the analytic observable sensitivities (CasADi jacobian for
            # casadi_python, Myokit CVODES for cellml_only + CVODE_myokit) -- i.e. the same
            # sources gradient_sources(model_type, solver) advertises for calibration. Which of
            # AD/FSA is actually usable follows from model_type/solver, so a front-end should
            # offer only gradient_sources(...)'s values (plus FD); an unavailable choice raises.
            {'name': 'gradient_source', 'type': 'enum', 'default': 'FD', 'required': False,
             'choices': ['FD', 'AD', 'FSA'],
             'description': ('Source for the Laplace Hessian: FD (finite-difference sub_method), '
                             'AD (exact CasADi, casadi_python), or FSA (Myokit CVODES Fisher '
                             'information, cellml_only + CVODE_myokit). See '
                             'gradient_sources(model_type, solver) for what the current model '
                             'supports.')},
            # enum, not str: utility_funcs.calculate_hessian dispatches on these. Only consulted
            # when gradient_source is 'FD'. 'AD' is a fourth branch there but raises
            # NotImplementedError, so it is deliberately not offered; any other value falls back
            # to plain finite differences with only a printed warning, which a free string invites.
            {'name': 'sub_method', 'type': 'enum', 'default': 'parabola_fit', 'required': False,
             'choices': ['parabola_fit', 'numdifftools_finite_diff'],
             'description': 'Finite-difference Hessian method for the Laplace approximation '
                            '(used when gradient_source is FD).'},
        ],
    },
}


def analysis_options(mode):
    """The option descriptors for a non-calibration analysis mode ('sensitivity_analysis',
    'mcmc', 'identifiability_analysis'); an empty list for an unknown mode."""
    meta = ANALYSIS_OPTIONS.get(mode)
    return meta['options'] if meta else []


def save_dated_user_inputs(inp_data_dict):
    """Best-effort: archive the resolved run config as ``user_inputs_<yymmdd>.yaml``
    in ``resources_dir``, so every run keeps a dated, reproducible record of what
    was run (mirrors the params_for_id / obs_data archival). Never raises — a
    failure here must not abort a run. Only rank 0 writes (avoids MPI clobber)."""
    if rank != 0:
        return
    try:
        resources_dir = inp_data_dict.get('resources_dir')
        if not resources_dir or not os.path.isdir(resources_dir):
            return
        # Keep only yaml-serialisable entries so a stray object never breaks a run.
        safe = {}
        for key, value in inp_data_dict.items():
            try:
                yaml.safe_dump({key: value})
            except Exception:
                continue
            safe[key] = value
        out_path = os.path.join(
            resources_dir, f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"
        )
        with open(out_path, 'w') as f:
            yaml.safe_dump(safe, f, default_flow_style=False, sort_keys=False)
    except Exception:
        pass


def warn_if_casadi_nonzero_pre_time(
    model_type,
    pre_time=None,
    pre_times=None,
    offline_pre_time=None,
    method=None,
):
    """Warn when CasADi AD with an ADJOINT integrator (cvodes/idas) is configured with nonzero
    warmup times. The symbolic bdf / semi_implicit_euler methods are differentiated by reverse
    mode (no adjoint) and support nonzero pre_time, so they are exempt."""
    if model_type != 'casadi_python':
        return
    if method is not None and method not in _CASADI_ADJOINT_METHODS:
        return

    issues = []
    if pre_time is not None and float(pre_time) != 0.0:
        issues.append(f'pre_time={pre_time}')
    if pre_times is not None:
        nonzero = [float(t) for t in pre_times if float(t) != 0.0]
        if nonzero:
            issues.append(f'protocol_info pre_times contains nonzero value(s): {nonzero}')
    if offline_pre_time is not None and float(offline_pre_time) != 0.0:
        issues.append(f'offline_pre_time={offline_pre_time}')

    if issues:
        warnings.warn(
            f'CasADi automatic differentiation with an adjoint integrator '
            f'(solver_info method={method!r}) does not support nonzero pre_time or pre_times: '
            'adjoint sensitivity integration typically fails with CV_TOO_MUCH_WORK. '
            'Set pre_time/pre_times to 0.0, or use a symbolic method '
            '(method="bdf" or "semi_implicit_euler") which supports warmup. '
            f'Affected: {", ".join(issues)}.',
            UserWarning,
            stacklevel=3,
        )
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
src_dir = os.path.join(os.path.dirname(__file__), '..')
param_id_dir = os.path.join(src_dir, 'param_id')
base_dir = os.path.join(src_dir, '..')
operation_funcs_user_dir = os.path.join(base_dir, 'funcs_user')

class scriptFunctionParser(object):
    '''
    Parses scripts with functions into objects (dicts) which holds the functions
    '''

    def __init__(self, operation_funcs_external_path=None, cost_funcs_external_path=None):
        sys.path.append(param_id_dir)
        sys.path.append(operation_funcs_user_dir)
        '''
        Constructor

        ``operation_funcs_external_path`` / ``cost_funcs_external_path`` (issue #303): optional
        paths to external Python files with additional user operation / cost funcs, merged in
        alongside the built-ins by ``get_operation_funcs_dict`` / ``get_cost_funcs_dict`` (and
        ``cost_func_metadata``). ``None``/empty -> only the built-in and funcs_user funcs.
        '''
        self.operation_funcs_external_path = operation_funcs_external_path
        self.cost_funcs_external_path = cost_funcs_external_path

    def get_operation_funcs_dict(self, mode="numpy"):
        import operation_funcs

        return operation_funcs.get_operation_funcs_dict_for_mode(
            mode, external_path=self.operation_funcs_external_path)

    def get_default_user_operation_funcs(self, mode="numpy"):
        """User operations are merged in ``get_operation_funcs_dict``; this is kept for API compatibility."""
        return {}

    def add_user_operation_func(self, operation_funcs_dict, func):
        operation_funcs_dict[func.__name__] = func
        return operation_funcs_dict

    def add_user_cost_func(self, cost_funcs_dict, func):
        cost_funcs_dict[func.__name__] = func
        return cost_funcs_dict

    def get_cost_funcs_dict(self, mode="numpy"):
        import cost_funcs_user

        return cost_funcs_user.get_cost_funcs_dict_for_mode(
            mode, external_path=self.cost_funcs_external_path)

    def cost_func_metadata(self, mode="numpy"):
        """Discoverable cost metadata (see cost_funcs_user.cost_func_metadata), including any
        external costs from ``cost_funcs_external_path``."""
        import cost_funcs_user

        return cost_funcs_user.cost_func_metadata(
            mode, external_path=self.cost_funcs_external_path)

class YamlFileParser(object):
    '''
    Parses Yaml files 
    '''
    def __init__(self):
        '''
        Constructor
        '''
    
    def parse_user_inputs_file(self, inp_data_dict, obs_path_needed=False, do_generation_with_fit_parameters=False):
        
        if inp_data_dict is None:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
                if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                    user_files_dir = os.path.dirname(inp_data_dict["user_inputs_path_override"])
                    with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
                else:
                    print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                    print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                            "you want to use the default user_inputs.yaml location")
                    exit()
            else:
                user_files_dir = ''
        else:
            user_files_dir = ''

        if inp_data_dict is None:
            print('no inp_data_dict provided and user_inputs.yaml not found, exiting')
            exit()
            
        if 'file_prefix' not in inp_data_dict.keys():
            print('file_prefix not found in inp_data_dict, exiting')
            exit()
        else:
            file_prefix = inp_data_dict['file_prefix']

        if 'couple_to_1d' not in inp_data_dict.keys():
            inp_data_dict['couple_to_1d'] = False
        
        if 'param_id_method' not in inp_data_dict.keys():
            inp_data_dict['param_id_method'] = 'genetic_algorithm'

        # cellml_only models get an AD gradient too, via Myokit CVODES forward sensitivity,
        # when run through the Myokit solver with do_ad set.
        _fsa_ad = (inp_data_dict.get('model_type') == 'cellml_only'
                   and inp_data_dict.get('solver', 'CVODE_myokit') == 'CVODE_myokit'
                   and inp_data_dict.get('do_ad'))
        if inp_data_dict.get('param_id_method') == 'sp_minimize' and \
                inp_data_dict.get('model_type') not in ('casadi_python', 'aadc_python') and not _fsa_ad:
            print('Parameter identification with sp_minimize requires model_type to be '
                  '"casadi_python" or "aadc_python", or "cellml_only" with solver '
                  '"CVODE_myokit" and do_ad: true (Myokit CVODES forward sensitivity).')
            exit()

        # multi_start_sp_minimize runs on any model type: it uses the AD gradient for
        # casadi_python (symbolic), aadc_python (tape), and cellml_only + Myokit CVODES FSA
        # (do_ad), and falls back to finite differences for the others.
        if inp_data_dict.get('param_id_method') == 'multi_start_sp_minimize' and \
                inp_data_dict.get('model_type') not in ('casadi_python', 'aadc_python') and not _fsa_ad:
            print('Note: multi_start_sp_minimize with model_type '
                  f'"{inp_data_dict.get("model_type")}" will use finite-difference gradients. '
                  'Set model_type to "casadi_python"/"aadc_python", or use "cellml_only" with '
                  'solver "CVODE_myokit" and do_ad: true, to use automatic differentiation.')

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            inp_data_dict['resources_dir'] = os.path.join(user_files_dir, inp_data_dict['resources_dir'])
        else:
            inp_data_dict['resources_dir'] = os.path.join(root_dir, 'resources')
        if "param_id_output_dir" in inp_data_dict.keys():
            inp_data_dict['param_id_output_dir'] = os.path.join(user_files_dir, inp_data_dict['param_id_output_dir'])
        else:
            inp_data_dict['param_id_output_dir'] = os.path.join(root_dir, 'param_id_output')
        if "generated_models_dir" in inp_data_dict.keys():
            inp_data_dict['generated_models_dir'] = os.path.join(user_files_dir, inp_data_dict['generated_models_dir'])
        else:
            inp_data_dict['generated_models_dir'] = os.path.join(root_dir, 'generated_models')
        
        if obs_path_needed:
            if 'param_id_obs_path' in inp_data_dict.keys():
                inp_data_dict['param_id_obs_path'] = os.path.join(user_files_dir, inp_data_dict['param_id_obs_path'])
                if not os.path.exists(inp_data_dict['param_id_obs_path']):
                    print(f'param_id_obs_path={inp_data_dict["param_id_obs_path"]} does not exist')
                    exit()
            else:
                print(f'param_id_obs_path not defined in user_inputs.yaml')
                print(f'Must run param_id.create_param_id_obs to create the param_id_observables')
                inp_data_dict['param_id_obs_path'] = None

            if 'params_for_id_file' in inp_data_dict.keys():
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['params_for_id_file'])
            else:
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], f'{file_prefix}_params_for_id.csv')
                if not os.path.exists(inp_data_dict['params_for_id_path']):
                    print(f'params_for_id_path={inp_data_dict["params_for_id_path"]} does not exist')
                    print(f'Therefore, you must run param_id.create_params_for_id to define the parameters for identification')
                    inp_data_dict['params_for_id_path'] = None

        if do_generation_with_fit_parameters:
            data_str_addon = re.sub('.json', '', os.path.split(inp_data_dict['param_id_obs_path'])[1])
            inp_data_dict['param_id_output_dir_abs_path'] = os.path.join(inp_data_dict['param_id_output_dir'], 
                                                                         inp_data_dict['param_id_method'] + f'_{file_prefix}_{data_str_addon}')
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], 
                                                                    file_prefix + '_' + data_str_addon)
        else:
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], file_prefix)
        
        os.makedirs(inp_data_dict['generated_models_dir'], exist_ok=True)
        os.makedirs(inp_data_dict['generated_models_subdir'], exist_ok=True)
            
        if 'model_type' not in inp_data_dict.keys():
            inp_data_dict['model_type'] = 'cellml_only'
            
        if inp_data_dict.get('model_type') == 'python_user_defined':
            model_ext = None
            # The "model" is the user's hand-written ODE wrapper in funcs_user/,
            # not a generated file. Default to funcs_user/{file_prefix}_wrapper.py,
            # but allow an explicit override via 'model_wrapper_path'.
            wrapper_path = inp_data_dict.get('model_wrapper_path')
            if not wrapper_path:
                wrapper_path = os.path.join(base_dir, 'funcs_user', f'{file_prefix}_wrapper.py')
            inp_data_dict['model_path'] = wrapper_path
            inp_data_dict['uncalibrated_model_path'] = wrapper_path
        elif inp_data_dict.get('model_type') in ['python', 'casadi_python']:
            model_ext = '.py'
        elif inp_data_dict.get('model_type') == 'cellml_only':
            model_ext = '.cellml'
        elif inp_data_dict.get('model_type') == 'cpp':
            model_ext = '.cpp'
        elif inp_data_dict.get('model_type') == 'aadc_python':
            model_ext = '.py'
        else:
            print(f'Invalid model type: {inp_data_dict.get("model_type")}')
            exit()

        if model_ext is not None:
            inp_data_dict['model_path'] = os.path.join(inp_data_dict['generated_models_subdir'], f'{file_prefix}{model_ext}')

            if do_generation_with_fit_parameters:
                inp_data_dict['uncalibrated_model_path'] = os.path.join(inp_data_dict["generated_models_dir"], file_prefix,
                                                   file_prefix + model_ext)
            else:
                inp_data_dict['uncalibrated_model_path'] = inp_data_dict['model_path']


        if 'dt' not in inp_data_dict.keys():
            inp_data_dict['dt'] = 0.01
        else:
            if ScalarFloat is not None and isinstance(inp_data_dict['dt'], ScalarFloat):
                inp_data_dict['dt'] = float(inp_data_dict['dt'])
            if type(inp_data_dict['dt']) != float:
                print(f'dt must be a float, but is {type(inp_data_dict["dt"])}')
                exit()
            
        if 'pre_time' in inp_data_dict.keys():
            inp_data_dict['pre_time'] = inp_data_dict['pre_time']
        else:
            inp_data_dict['pre_time'] = 0.0

        if 'sim_time' in inp_data_dict.keys():
            inp_data_dict['sim_time'] = inp_data_dict['sim_time']
        else:
            print(f'sim_time not found in inp_data_dict, setting to None so it can be set in protocol_info')
            inp_data_dict['sim_time'] = None

        # Parse and validate the solver parameter
        # Supported solvers: CVODE_opencor (OpenCOR), CVODE_myokit (Myokit), solve_ivp (Python), or casadi_integrator (CasADi)
        
        # Sourced from the module-level SOLVER_SCHEMA (single source of truth).
        _solvers = SOLVER_SCHEMA['solvers_by_model_type']
        _methods = SOLVER_SCHEMA['methods_by_solver']
        valid_cellml_solvers = _solvers['cellml_only']
        valid_cellml_methods = _methods['CVODE_myokit']
        valid_cpp_solvers = _solvers['cpp']  # TODO should this be different to methods?
        valid_cpp_methods = _solvers['cpp']
        valid_python_solvers = _solvers['python']
        valid_solve_ivp_methods = _methods['solve_ivp']
        valid_casadi_solvers = _solvers['casadi_python']
        valid_casadi_solver_plugins = _methods['casadi_integrator']
        valid_aadc_solvers = _solvers.get('aadc_python', [])
        valid_user_defined_solvers = _solvers.get('python_user_defined', [])

        solver_name = inp_data_dict.get('solver_info', {}).get('solver')
        if solver_name is None:
            solver_name = inp_data_dict.get('solver')
        
        # Backward compatibility: 
        if solver_name == 'CVODE':
            solver_name = 'CVODE_myokit' # default to CVODE_myokit for cellml models

        if solver_name is None:
            if inp_data_dict.get('model_type') == 'cellml_only':
                solver_name = 'CVODE_opencor'
            elif inp_data_dict.get('model_type') == 'python':
                solver_name = 'solve_ivp'
            elif inp_data_dict.get('model_type') == 'cpp':
                solver_name = 'CVODE'
            elif inp_data_dict.get('model_type') == 'casadi_python':
                solver_name = 'cvodes'
            elif inp_data_dict.get('model_type') == 'aadc_python':
                solver_name = 'aadc_semi_implicit'
            elif inp_data_dict.get('model_type') == 'python_user_defined':
                solver_name = 'user_defined'
            else:
                print(f'Invalid model type: {inp_data_dict.get("model_type")}')
                exit()
        else:
            valid_aadc_solvers = SOLVER_SCHEMA['solvers_by_model_type'].get('aadc_python', [])
            if (solver_name not in valid_cellml_solvers and
                solver_name not in valid_cpp_solvers and
                solver_name not in valid_python_solvers and
                solver_name not in valid_casadi_solvers and
                solver_name not in valid_aadc_solvers and
                solver_name not in valid_user_defined_solvers):
                print(f'Invalid solver: {solver_name}')
                exit()
        

        if 'solver_info' in inp_data_dict:
            try:
                inp_data_dict['solver_info'] = migrate_legacy_solver_info_keys(
                    solver_name, inp_data_dict['solver_info']
                )
            except ValueError as exc:
                # Same failure mode as validate_solver_info below: the config is
                # wrong and no guess about which value was meant is safe.
                print(exc)
                exit()

        if 'solver_info' not in inp_data_dict.keys():
            inp_data_dict['solver_info'] = _solver_info_default_for(
                inp_data_dict['model_type'], solver_name
            )
        else:
            defaults = _solver_info_default_for(inp_data_dict['model_type'], solver_name)
            if inp_data_dict.get('model_type') == 'casadi_python':
                if 'max_num_steps' not in inp_data_dict['solver_info']:
                    inp_data_dict['solver_info']['max_num_steps'] = defaults.get('max_num_steps', 5000)
            elif inp_data_dict.get('model_type') == 'python':
                if 'max_step' not in inp_data_dict['solver_info']:
                    inp_data_dict['solver_info']['max_step'] = defaults.get('max_step', 0.001)
            elif inp_data_dict.get('model_type') == 'aadc_python':
                pass  # AADC solver handles its own defaults
            elif inp_data_dict.get('model_type') == 'python_user_defined':
                pass  # user wrapper handles its own integration
            elif ('MaximumNumberOfSteps' in defaults
                  and 'MaximumNumberOfSteps' not in inp_data_dict['solver_info']):
                inp_data_dict['solver_info']['MaximumNumberOfSteps'] = defaults['MaximumNumberOfSteps']
        if 'solver' not in inp_data_dict['solver_info'].keys():
            inp_data_dict['solver_info']['solver'] = solver_name
        elif inp_data_dict.get('model_type') == 'cpp':
            if solver_name.startswith('RK4'):
                inp_data_dict['solver_info']['solver'] = 'RK4'
                solver_name = 'RK4'
            elif solver_name.startswith('CVODE'):
                inp_data_dict['solver_info']['solver'] = 'CVODE'
                solver_name = 'CVODE'
            elif solver_name.startswith('PETSC'):
                inp_data_dict['solver_info']['solver'] = 'PETSC'
                solver_name = 'PETSC'

        solver_info = dict(inp_data_dict['solver_info'])
        if 'solver' not in solver_info:
            solver_info['solver'] = solver_name

        dt_solver = solver_info.get('dt_solver')
        if dt_solver is None:
            dt_solver = solver_info.get('MaximumStep')
        if dt_solver is None:
            dt_solver = solver_info.get('max_step_size')
        if dt_solver is None:
            dt_solver = solver_info.get('max_step')
        if dt_solver is not None:
            solver_info['dt_solver'] = dt_solver
        if solver_info.get('solver', '').startswith('CVODE') and dt_solver is not None:
            solver_info['MaximumStep'] = dt_solver

        inp_data_dict['solver_info'] = solver_info

        if 'solver' in inp_data_dict:
            del inp_data_dict['solver']

        if 'method' in inp_data_dict.get('solver_info', {}):
            solver_method = inp_data_dict['solver_info']['method']
        else:
            if inp_data_dict.get('model_type') == 'cpp':
                if solver_name.startswith('CVODE'):
                    solver_method = 'CVODE'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('RK4'):
                    solver_method = 'RK4'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('PETSC'):
                    solver_method = 'PETSC'
                    inp_data_dict['solver_info']['method'] = solver_method # TODO Bea: add specific solver to be used within PETSC (CN / BDF1 / BDF2 / ...)
                else:
                    print(f'solver set {solver_name} not compatible with model_type cpp : change this in the user_inputs.yaml file')
            elif solver_name in valid_user_defined_solvers:
                # The user wrapper is integrated by solve_ivp; default to RK45.
                solver_method = 'RK45'
                inp_data_dict['solver_info']['method'] = solver_method
            else:
                if solver_name.startswith('CVODE'):
                    solver_method = 'CVODE'
                    inp_data_dict['solver_info']['method'] = solver_method
                elif solver_name.startswith('casadi'):
                    print('Method not set in solver_options, which should be set for solver casadi_integrator, '
                          'using default method cvodes')
                    solver_method = 'cvodes'
                    inp_data_dict['solver_info']['method'] = solver_method
                else:
                    print('Method not set in solver_options, which should be set for solver solve_ivp,'
                        'using default method RK45')
                    solver_method = 'RK45'
                    inp_data_dict['solver_info']['method'] = solver_method

        if (solver_name not in valid_cellml_solvers
            and solver_name not in valid_python_solvers
            and solver_name not in valid_cpp_solvers
            and solver_name not in valid_casadi_solvers
            and solver_name not in valid_aadc_solvers
            and solver_name not in valid_user_defined_solvers):
            print(f'Invalid solver: {solver_name}')
            print(f'Valid CellML solvers: {valid_cellml_solvers}')
            print(f'Valid Python solvers: {valid_python_solvers}')
            print(f'Valid Cpp solvers: {valid_cpp_solvers}')
            print(f'Valid CasADi solvers: {valid_casadi_solvers}')
            print(f'Valid AADC solvers: {valid_aadc_solvers}')
            print(f'Valid user-defined solvers: {valid_user_defined_solvers}')
            exit()
        
        
        # Validate solver-model compatibility
        # CellML and CasADi solvers cannot be used with Python models
        if inp_data_dict.get('model_type') == 'python' and solver_name not in valid_python_solvers:
                print(f'Solver {solver_method} cannot be used with Python models (model_type="python")')
                print(f'Use {valid_python_solvers} for Python models')
                exit()

        # solve_ivp methods can only be used with Python models (generated or user-defined)
        if solver_method in valid_solve_ivp_methods:
            if inp_data_dict.get('model_type') not in ['python', 'python_user_defined', None]:
                print(f'solve_ivp method {solver_method} requires model_type to be "python"')
                print('Use CVODE_opencor (or legacy CVODE) or CVODE_myokit for CellML models')
                print('Use CVODE or RK4 or PETSC for Cpp models')
                exit()

        if solver_name in valid_cellml_solvers and solver_method not in valid_cellml_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_cellml_methods} for CellML models')
            exit()
        
        if solver_name in valid_cpp_solvers and solver_method not in valid_cpp_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_cpp_methods} for Cpp models')
            exit()
        if solver_name in valid_python_solvers and solver_method not in valid_solve_ivp_methods:
            print(f'solver method {solver_method} not compatible with solver {solver_name}, use {valid_solve_ivp_methods} for Python models')
            exit()

        # CellML and Python solvers cannot be used with CasADi Python models
        if inp_data_dict.get('model_type') == 'casadi_python' and solver_name not in valid_casadi_solvers:
                print(f'Solver {solver_method} cannot be used with CasADi Python models (model_type="casadi_python")')
                print(f'Use {valid_casadi_solvers} for CasADi Python models')
                exit()

        # AADC solvers can only be used with AADC Python models
        if inp_data_dict.get('model_type') == 'aadc_python' and solver_name not in valid_aadc_solvers:
                print(f'Solver {solver_name} cannot be used with AADC Python models')
                print(f'Use {valid_aadc_solvers} for AADC Python models')
                exit()

        # CasADi solvers can only be used with CasADi Python models.
        # aadc_python is exempted here so that a stale config carrying the removed AADC
        # method 'bdf' falls through to validate_solver_info below, which names the methods
        # aadc_semi_implicit actually accepts, rather than being rejected with a misleading
        # "requires model_type casadi_python" message.
        if solver_method in valid_casadi_solver_plugins:
            if inp_data_dict.get('model_type') not in ['casadi_python', 'aadc_python', None]:
                print(f'CasADi solver {solver_method} requires model_type to be "casadi_python"')
                print('Use CVODE_opencor (or legacy CVODE) or CVODE_myokit for CellML models')
                print('Use CVODE or RK4 or PETSC for Cpp models')
                exit()

        try:
            validate_solver_info(solver_name, inp_data_dict['solver_info'])
        except ValueError as exc:
            print(exc)
            exit()

        warn_if_casadi_nonzero_pre_time(
            inp_data_dict.get('model_type'),
            pre_time=inp_data_dict.get('pre_time'),
            method=(inp_data_dict.get('solver_info') or {}).get('method'),
        )

        if 'DEBUG' in inp_data_dict.keys(): 
            if inp_data_dict['DEBUG']:
                # For backwards compatibility, still set ga_options if debug_ga_options exists
                if 'debug_ga_options' in inp_data_dict.keys():
                    inp_data_dict['ga_options'] = inp_data_dict['debug_ga_options']
                if 'debug_mcmc_options' in inp_data_dict.keys():
                    inp_data_dict['mcmc_options'] = inp_data_dict['debug_mcmc_options']
            else:
                pass
        else:
            inp_data_dict['DEBUG'] = False

        if 'external_modules_dir' not in inp_data_dict.keys() or inp_data_dict['external_modules_dir'] is None:
            inp_data_dict['external_modules_dir'] = None
        else:
            # check if it is an absolute path
            if not os.path.isabs(inp_data_dict['external_modules_dir']):
                inp_data_dict['external_modules_dir'] = os.path.join(user_files_dir, inp_data_dict['external_modules_dir'])
            else:
                inp_data_dict['external_modules_dir'] = inp_data_dict['external_modules_dir']
            # check if external_modules_dir is a valid directory
            if not os.path.exists(inp_data_dict['external_modules_dir']):
                print(f'external_modules_dir={inp_data_dict["external_modules_dir"]} does not exist')
                exit()
        
        # for sensitivity analysis and parameter identification
        if not 'sa_options' in inp_data_dict.keys():
            inp_data_dict['sa_options'] = None

        if inp_data_dict['sa_options'] is None:
            inp_data_dict['sa_options'] = {
                'method': 'sobol',
                'num_samples': 32,
                'sample_type': 'saltelli',
                'output_dir': os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')
            }
        else:
            if 'output_dir' not in inp_data_dict['sa_options'].keys():
                inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')  
            else:
                if not os.path.isabs(inp_data_dict['sa_options']['output_dir']):
                    inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', inp_data_dict['sa_options']['output_dir']) 
            
            if not os.path.exists(inp_data_dict['sa_options']['output_dir']):
                os.makedirs(inp_data_dict['sa_options']['output_dir'], exist_ok=True)
            
            if 'method' not in inp_data_dict['sa_options'].keys():
                print('No method specified for sensitivity analysis, setting to sobol by default')
                inp_data_dict['sa_options']['method'] = 'sobol'
            if 'num_samples' not in inp_data_dict['sa_options'].keys():
                print('No num_samples specified for sensitivity analysis, setting to 32 by default')
                inp_data_dict['sa_options']['num_samples'] = 32
            if 'sample_type' not in inp_data_dict['sa_options'].keys():
                print('No sample_type specified for sensitivity analysis, setting to saltelli by default')
                inp_data_dict['sa_options']['sample_type'] = 'saltelli'
            
        if 'do_ia' not in inp_data_dict.keys():
            inp_data_dict['do_ia'] = False
        
        if 'ia_options' not in inp_data_dict.keys():
            inp_data_dict['ia_options'] = {
                'method': 'Laplace'
            }
        else:
            if 'method' not in inp_data_dict['ia_options'].keys():
                print('No method specified for identifiability analysis, setting to Laplace by default')
                inp_data_dict['ia_options']['method'] = 'Laplace'
        
        # Parse optimiser_options - this is the new unified way to specify options
        # Handle backwards compatibility: if ga_options or debug_ga_options is specified, merge into optimiser_options
        if 'optimiser_options' not in inp_data_dict.keys():
            inp_data_dict['optimiser_options'] = {}
        else:
            # Ensure optimiser_options is a dictionary
            if inp_data_dict['optimiser_options'] is None:
                inp_data_dict['optimiser_options'] = {}
            
        # Merge ga_options into optimiser_options for backwards compatibility
        
        # Backwards compatibility: convert ga_options to optimiser_options
        # Only copy entries that don't already exist in optimiser_options to avoid duplicates
        # Note: If DEBUG is True, ga_options may have been set to debug_ga_options above,
        # but we still want to merge the original ga_options first (if it exists and DEBUG is False)
        # or merge debug_ga_options with higher precedence when DEBUG is True
        if not inp_data_dict['DEBUG']:
            # When DEBUG is False, merge ga_options normally
            if 'ga_options' in inp_data_dict.keys() and inp_data_dict['ga_options'] is not None:
                ga_opts = inp_data_dict['ga_options']
                if isinstance(ga_opts, dict):
                    for key, value in ga_opts.items():
                        # Only add if not already in optimiser_options
                        if key not in inp_data_dict['optimiser_options']:
                            inp_data_dict['optimiser_options'][key] = value
                        # Warn if there's a conflict (same key, different value)
                        elif inp_data_dict['optimiser_options'][key] != value:
                            print(f'Warning: ga_options["{key}"] conflicts with optimiser_options["{key}"]. '
                                  f'Using optimiser_options value: {inp_data_dict["optimiser_options"][key]}')
        
        # Handle debug_ga_options for backwards compatibility
        # When DEBUG is True, debug_ga_options should override optimiser_options
        if inp_data_dict['DEBUG']:
            if 'debug_ga_options' in inp_data_dict.keys() and inp_data_dict['debug_ga_options'] is not None:
                debug_ga_opts = inp_data_dict['debug_ga_options']
                if isinstance(debug_ga_opts, dict):
                    for key, value in debug_ga_opts.items():
                        # Debug options override optimiser_options (they take precedence)
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_ga_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value
        
        # Handle debug_optimiser_options (new preferred way)
        # Only apply when DEBUG is True to avoid overriding production runs
        if inp_data_dict['DEBUG']:
            if 'debug_optimiser_options' in inp_data_dict.keys() and inp_data_dict['debug_optimiser_options'] is not None:
                debug_opts = inp_data_dict['debug_optimiser_options']
                if isinstance(debug_opts, dict):
                    for key, value in debug_opts.items():
                        if key in inp_data_dict['optimiser_options']:
                            if inp_data_dict['optimiser_options'][key] != value:
                                print(f'Note: debug_optimiser_options["{key}"] overriding optimiser_options["{key}"] '
                                      f'({inp_data_dict["optimiser_options"][key]} -> {value})')
                        inp_data_dict['optimiser_options'][key] = value

        # for generation only
    
        inp_data_dict['vessels_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], file_prefix + '_vessel_array.csv')
        inp_data_dict['parameters_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['input_param_file'])

        if inp_data_dict.get('model_type') == 'cpp' and inp_data_dict.get('couple_to_1d'):
            file_prefix_0d = file_prefix + '_0d'
            file_prefix_1d = file_prefix + '_1d'

            vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
            idx_last = vessels_csv_abs_path.rfind(file_prefix)
            vessel_filename_0d = vessels_csv_abs_path[:idx_last] + file_prefix_0d + vessels_csv_abs_path[idx_last+len(file_prefix):]
            vessel_filename_1d = vessels_csv_abs_path[:idx_last] + file_prefix_1d + vessels_csv_abs_path[idx_last+len(file_prefix):]

            inp_data_dict['file_prefix_0d'] = file_prefix_0d
            inp_data_dict['file_prefix_1d'] = file_prefix_1d
            inp_data_dict['vessels_0d_csv_abs_path'] = vessel_filename_0d
            inp_data_dict['vessels_1d_csv_abs_path'] = vessel_filename_1d

        # Archive a dated copy of the resolved config for reproducibility.
        save_dated_user_inputs(inp_data_dict)

        return inp_data_dict


# Keys always allowed in solver_info (framework metadata, not passed to integrators).
_FRAMEWORK_SOLVER_INFO_KEYS = frozenset({'solver', 'method', 'dt_solver'})

# Integrator-specific keys that may appear in solver_info for each backend. Derived from
# SOLVER_INFO_FIELDS (the schema surfaced to tools) so the validation and the advertised settings
# cannot drift apart -- add a field there and it is both offered to the UI and accepted here.
_SOLVER_INTEGRATOR_KEYS = {
    solver: frozenset(field['name'] for field in fields)
    for solver, fields in SOLVER_INFO_FIELDS.items()
}


def _warn_renamed_solver_info_key(solver_name, old_key, new_key):
    """Tell the user which key the value was moved to, and to rename it.

    Migration used to be silent, so a config could keep a key that had quietly
    stopped doing anything and nothing said which key had replaced it.
    """
    print(
        f'WARNING: solver_info key {old_key!r} is not used by solver '
        f'{solver_name!r}; its value was applied to {new_key!r} instead. '
        f'Rename it in your user_inputs to silence this.'
    )


def _raise_duplicate_solver_info_key(solver_name, old_key, new_key, solver_info):
    """One setting, specified twice, under two names.

    Not a warning: picking a winner would silently discard the other value, and
    there is no way to tell which one the user meant -- if the two disagree, one
    of them is what they think the run is using. Refuse and make them delete one.
    """
    raise ValueError(
        f'solver_info sets both {old_key!r} ({solver_info[old_key]!r}) and '
        f'{new_key!r} ({solver_info[new_key]!r}) for solver {solver_name!r}. '
        f'These are the same setting: {old_key!r} is the legacy name for '
        f'{new_key!r}, which is the one this solver uses. Remove {old_key!r} '
        f'(keeping {new_key!r}) so there is one value, not two.'
    )


def _warn_dropped_solver_info_key(solver_name, key, reason):
    print(
        f'WARNING: solver_info key {key!r} is not supported by solver '
        f'{solver_name!r} and was ignored. {reason}'
    )


def migrate_legacy_solver_info_keys(solver_name, solver_info):
    """
    Map legacy CVODE-style solver_info keys to backend-specific names and drop
    keys that the selected integrator does not accept.

    Every rename and every drop warns, naming the key to use instead (or saying
    that there is none), so a setting can never stop taking effect silently.
    Migrating rather than rejecting keeps configs written for another backend --
    or for an older CA -- working.

    Raises ValueError when a config sets both names for one setting: that is not
    a stale key to migrate but a contradiction, and no choice between the two
    values can be made on the user's behalf.
    """
    solver_info = dict(solver_info)

    def migrate(old_key, new_key, fallback_key=None):
        """Move ``old_key`` onto ``new_key`` (unless already set), then drop it."""
        if old_key not in solver_info and fallback_key not in solver_info:
            return
        if new_key in solver_info and old_key in solver_info:
            # One setting under two names. Silently preferring either would hide
            # the other value from a user who believes it is in effect.
            _raise_duplicate_solver_info_key(solver_name, old_key, new_key, solver_info)
        if new_key not in solver_info:
            source = old_key if old_key in solver_info else fallback_key
            solver_info[new_key] = solver_info[source]
            if source == old_key:
                _warn_renamed_solver_info_key(solver_name, old_key, new_key)
        solver_info.pop(old_key, None)

    if solver_name == 'solve_ivp':
        migrate('MaximumStep', 'max_step', fallback_key='dt_solver')
        if 'MaximumNumberOfSteps' in solver_info:
            _warn_dropped_solver_info_key(
                solver_name, 'MaximumNumberOfSteps',
                'scipy.integrate.solve_ivp has no step-count limit.',
            )
            solver_info.pop('MaximumNumberOfSteps', None)
    elif solver_name == 'casadi_integrator':
        migrate('MaximumStep', 'max_step_size', fallback_key='dt_solver')
        migrate('MaximumNumberOfSteps', 'max_num_steps')
    elif solver_name == 'CVODE_myokit':
        # No equivalent to migrate onto: myokit.Simulation exposes only
        # set_max_step_size / set_min_step_size / set_tolerance.
        if 'MaximumNumberOfSteps' in solver_info:
            _warn_dropped_solver_info_key(
                solver_name, 'MaximumNumberOfSteps',
                "Myokit's integrator has no maximum-step-count setting; use "
                "'MaximumStep' to bound the step size, or 'rtol'/'atol' to "
                'control accuracy.',
            )
            solver_info.pop('MaximumNumberOfSteps', None)

    return solver_info


def validate_solver_info(solver_name, solver_info):
    """
    Validate solver_info keys against the selected solver backend.

    Raises ValueError listing unsupported keys and the allowed keys for that solver.
    """
    if solver_name not in _SOLVER_INTEGRATOR_KEYS:
        raise ValueError(
            f'Cannot validate solver_info for unknown solver {solver_name!r}. '
            f'Known solvers: {sorted(_SOLVER_INTEGRATOR_KEYS)}'
        )

    allowed = _FRAMEWORK_SOLVER_INFO_KEYS | _SOLVER_INTEGRATOR_KEYS[solver_name]
    unsupported = sorted(
        key for key in solver_info.keys()
        if key not in allowed
    )
    if not unsupported:
        return

    integrator_keys = sorted(_SOLVER_INTEGRATOR_KEYS[solver_name])
    framework_keys = sorted(_FRAMEWORK_SOLVER_INFO_KEYS)
    hints = []
    if solver_name == 'casadi_integrator':
        if 'MaximumStep' in unsupported:
            hints.append(
                f'for solver {solver_name!r}, use max_step_size instead of MaximumStep'
            )
        if 'MaximumNumberOfSteps' in unsupported:
            hints.append(
                f'for solver {solver_name!r}, use max_num_steps instead of MaximumNumberOfSteps'
            )
    elif solver_name == 'solve_ivp':
        if 'MaximumStep' in unsupported:
            hints.append(
                f'for solver {solver_name!r}, use max_step instead of MaximumStep'
            )
        if 'MaximumNumberOfSteps' in unsupported:
            hints.append(
                f'for solver {solver_name!r}, MaximumNumberOfSteps is not supported'
            )

    hint_text = f' Hint: {"; ".join(hints)}.' if hints else ''
    raise ValueError(
        f'solver_info contains key(s) not supported by solver {solver_name!r}: {unsupported}. '
        f'Allowed framework keys: {framework_keys}. '
        f'Allowed integrator keys for {solver_name}: {integrator_keys}.{hint_text}'
    )


def _solver_info_default_for(model_type, solver_name):
    """``get_solver_info_default`` narrowed to the keys ``solver_name`` accepts.

    The defaults are per model_type, but a model_type can host backends with
    different settings: cellml_only covers both CVODE_opencor (which takes
    MaximumNumberOfSteps) and CVODE_myokit (which has no such knob). Seeding the
    whole default set would put back exactly what
    migrate_legacy_solver_info_keys just removed, and validate_solver_info would
    then reject CA's own default.
    """
    defaults = get_solver_info_default(model_type)
    allowed = _SOLVER_INTEGRATOR_KEYS.get(solver_name)
    if allowed is None:
        return defaults
    allowed = allowed | _FRAMEWORK_SOLVER_INFO_KEYS
    return {k: v for k, v in defaults.items() if k in allowed}


def get_solver_info_default(model_type):
    if model_type == 'cellml_only':
        return {
            'solver': 'CVODE_opencor',
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    if model_type == 'python':
        return {
            'solver': 'solve_ivp',
            'method': 'RK45',
            'max_step': 0.001,
            'rtol': 1e-8,
            'atol': 1e-8,
        }
    if model_type == 'cpp':
        return {
            'solver': 'CVODE',
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
            'rtol': 1e-8,
            'atol': 1e-8
        }
    if model_type == 'casadi_python':
        return {
            'solver': 'casadi_integrator',
            'method': 'cvodes',
            'max_step_size': 0.001,
            'max_num_steps': 5000,
            'reltol': 1e-8,
            'abstol': 1e-10,
        }
    if model_type == 'aadc_python':
        return {
            'solver': 'aadc_semi_implicit',
            'method': 'adaptive_rk45',
            'tol': 1e-8,
            'threads': 4,
        }
    if model_type == 'python_user_defined':
        return {
            'solver': 'user_defined',
            'method': 'RK45',
            'max_step': 0.001,
            'rtol': 1e-8,
            'atol': 1e-8,
        }
    raise ValueError(f'Invalid model type: {model_type}')

class CSVFileParser(object):
    '''
    Parses CSV files
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if( has_header ):
            csv_dataframe = pd.read_csv(filename, dtype=str, na_filter=False)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None, na_filter=False)

        csv_dataframe = csv_dataframe.rename(columns=lambda x: x.strip())
        # Ensure object dtype so list-like assignments are allowed (pandas >=2.0 uses StringArray)
        csv_dataframe = csv_dataframe.astype(object)
        for II in range(csv_dataframe.shape[0]):
            for column_index, column_name in enumerate(csv_dataframe.columns):
                entry = csv_dataframe.iat[II, column_index]
                if type(entry) is not str:
                    sub_entries = []
                else:
                    sub_entries = entry.split()

                if column_name in ['vessel_name', 'inp_vessels', 'out_vessels']:
                    if sub_entries == []:
                        new_entry = []
                        pass
                    else:
                        new_entry = [sub_entry.strip() for sub_entry in sub_entries]
                else:
                    if sub_entries == []:
                        new_entry = []
                    else:
                        new_entry = sub_entries[0].strip()

                # Use iat to avoid pandas trying to broadcast list-like values.
                csv_dataframe.iat[II, column_index] = new_entry

        # for column_name in csv_dataframe.columns:
        #     if column_name == 'vessel_name':
        #         continue
        #     csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()
    
        return csv_dataframe

    def get_data_as_dataframe(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if (has_header):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        for column_name in csv_dataframe.columns:
            csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()

        return csv_dataframe

    def get_data_as_nparray(self,filename,has_header=True):
        '''
        Returns the data in the CSV file as a numpy array
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''

        csv_dataframe = self.get_data_as_dataframe(filename, has_header)
    
        csv_np_array = csv_dataframe.to_numpy()
        dtypes = []
        for column in list(csv_dataframe.columns):
            dtypes.append((column,'<U80'))
            
        csv_np_array = np.array(list(zip(*csv_np_array.T)), dtype=dtypes)
    
        return csv_np_array

    def get_data_as_dictionary(self,filename):
        '''
        Returns the data in the CSV file as a Python dictionary
        :param filename: filename of CSV file
        '''

        csv_dataframe = self.get_data_as_dataframe(filename).T
        csv_dictionary = csv_dataframe.to_dict()
        
        return list(csv_dictionary.values())


    def get_param_id_params_as_lists_of_tuples(self, param_id_dir):

        param_names = []

        with open(os.path.join(os.path.join(param_id_dir, 'param_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_names.append(row)


        # get date identifier of the parameter id
        date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

        param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))
        param_name_and_val = []
        
        for name_or_list, val in zip(param_names, param_vals):
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    param_name_and_val.append((name, val))
            else:
                param_name_and_val.append((name, val))

        return param_name_and_val, date_id


class JSONFileParser(object):
    '''
    Parses json files
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def json_to_dataframe(self, json_path):
        with open(json_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        df = pd.DataFrame(json_obj)
        return df

    @staticmethod
    def _is_json_module_file(file):
        # macOS writes AppleDouble '._<name>.json' sidecar files on non-native partitions; they
        # match '.json' but are binary and blow up json.load, so skip them here (issue #83).
        return file.endswith('.json') and not file.startswith('._')

    def json_to_dataframe_with_user_dir(self, json_dir, json_user_dir, external_modules_dir):
        dfs = [self.json_to_dataframe(os.path.join(json_dir, file)) \
                for file in os.listdir(json_dir) if self._is_json_module_file(file)]
        user_module_dfs = [self.json_to_dataframe(os.path.join(json_user_dir, file)) \
                for file in os.listdir(json_user_dir) if self._is_json_module_file(file)]
        if external_modules_dir is not None:
            external_module_dfs = [self.json_to_dataframe(os.path.join(external_modules_dir, file)) \
                    for file in os.listdir(external_modules_dir) if self._is_json_module_file(file)]
        else:
            external_module_dfs = []
            
        df = None
        for json_df in dfs:
            if df is None:
                df = json_df
            else:
                # concatenate dataframes, ignore index to reset the index
                # so that it is not duplicated
                df = pd.concat([df, json_df], ignore_index=True)

        for user_module_df in user_module_dfs:
            df = pd.concat([df, user_module_df], ignore_index=True)
        for external_module_df in external_module_dfs:
            df = pd.concat([df, external_module_df], ignore_index=True)
        return df
    
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        '''
        with open(filename, 'r') as f:
            json_obj = json.load(f)
        df = pd.DataFrame(json_obj)
        return df

    def append_module_config_info_to_vessel_df(self, vessel_df, module_df):
        # add columns to vessel_df

        add_on_lists = {column:[] for column in module_df.columns[2:]}
        for vessel_tup in vessel_df.itertuples():
            vessel_type = vessel_tup.vessel_type
            BC_type = vessel_tup.BC_type
            if len(BC_type) <1 or len(vessel_type) <1:
                print('You have an empty entry in your vessel array, exiting')
                exit()
            this_vessel_module_df = module_df.loc[((module_df["vessel_type"] == vessel_type)
                                                   & (module_df["BC_type"] == BC_type))].squeeze()
            if this_vessel_module_df.empty:
                print(f'combination of vessel_type = {vessel_type} and BC_type = {BC_type} doesn\'t exist, check module_config.json',
                        'for this combination')
                exit()
            for column in add_on_lists:
                # deepcopy to make sure that the lists for different vessel same module are not linked
                val = this_vessel_module_df[column]
                is_na = False
                try:
                    mask = pd.isna(val)
                    if isinstance(mask, (np.bool_, bool)):
                        is_na = bool(mask)
                    else:
                        # array-like: consider NaN only if all entries are NaN
                        is_na = bool(np.all(mask))
                except Exception:
                    is_na = False

                if is_na:
                    add_on_lists[column].append("None")
                else:
                    add_on_lists[column].append(copy.deepcopy(val))

        for column in add_on_lists:
            vessel_df[column] = add_on_lists[column]


def validate_params_to_change(protocol_info):
    """
    Raise ValueError if params_to_change rows are not aligned with sim_times.

    Every key must have one experiment row per entry in sim_times, and each row
    must have one value per subexperiment (len(sim_times[exp_idx])).
    """
    ptc = protocol_info.get("params_to_change") or {}
    if not ptc:
        return

    sim_times = protocol_info.get("sim_times")
    if sim_times is None:
        raise ValueError("protocol_info missing required key 'sim_times'")

    n_exp = len(sim_times)
    pre_times = protocol_info.get("pre_times")
    if pre_times is not None and len(pre_times) != n_exp:
        raise ValueError(
            f"pre_times length ({len(pre_times)}) does not match "
            f"num_experiments ({n_exp})"
        )

    errors = []
    for key, rows in sorted(ptc.items()):
        if not isinstance(rows, (list, tuple)):
            errors.append(
                f"  {key}: expected list of experiment rows, got {type(rows).__name__}"
            )
            continue
        if len(rows) != n_exp:
            errors.append(
                f"  {key}: {len(rows)} experiment row(s), expected {n_exp}"
            )
            continue
        for exp_idx, pair in enumerate(rows):
            n_sub = len(sim_times[exp_idx])
            if not isinstance(pair, (list, tuple)):
                errors.append(
                    f"  {key}[{exp_idx}]: expected list, got {type(pair).__name__}"
                )
            elif len(pair) != n_sub:
                errors.append(
                    f"  {key}[{exp_idx}]: {len(pair)} sub value(s), expected {n_sub}"
                )

    if errors:
        raise ValueError(
            "params_to_change shape mismatch:\n" + "\n".join(errors)
        )


class ObsAndParamDataParser(object):
    def __init__(self, modifier_funcs_external_path=None):
        # Optional external file of user modifier functions (issue #383), threaded from the
        # `modifier_funcs_external_path` config key -- mirrors operation/cost funcs. Modifier
        # entries validate against the merged registry, and the path is recorded on the built
        # param_id_info so the expansion and resolve steps load the same registry later.
        self.modifier_funcs_external_path = modifier_funcs_external_path

    def parse_obs_data_json(
        self,
        param_id_obs_path=None,
        obs_data_dict=None,
        pre_time=None,
        sim_time=None,
        model_type=None,
        method=None,
    ):
        """
        Loads the ground truth observation data from the JSON file and returns 
        the core data structures: gt_df, protocol_info, and prediction_info.
        """
        
        if param_id_obs_path is not None:
            with open(param_id_obs_path, encoding='utf-8-sig') as rf:
                json_obj = json.load(rf)
        elif obs_data_dict is not None:
            json_obj = obs_data_dict
        else:
            print("No obs data path or obs data dict provided, exiting")
            return None

        gt_df, protocol_info, prediction_info = None, None, None
        REQUIRED = "REQUIRED"

        def _is_missing_scalar(val):
            if val is None:
                return True
            try:
                is_na = pd.isna(val)
            except Exception:
                return False
            return isinstance(is_na, (bool, np.bool_)) and bool(is_na)

        def _hydrate_series_data_items(data_items):
            """
            Normalise series entries before schema validation.

            Handles three legacy/current formats:
            1. ``data_type: "timeseries"`` → renamed to ``"series"``.
            2. Top-level ``t_path`` + ``vm_path`` / ``im_path`` (old format,
               variable name used to pick which signal is the observable).
            3. Top-level ``t_path`` + ``value_path`` (current format, no
               ambiguity — ``value_path`` is always the observable).

            When ``value`` is missing it is loaded from ``value_path`` (and
            ``obs_dt`` from ``t_path`` if omitted). ``std`` must always be set
            in the JSON: either one positive scalar (applied to every sample)
            or a vector with the same length as the series.

            Raises ValueError if a series item specifies embedded ``value`` and
            ``t_path`` / ``value_path`` at the same time.
            """
            if not data_items:
                return data_items

            def _series_length(item, y_arr=None):
                if y_arr is not None:
                    return len(y_arr)
                val = item.get("value")
                if isinstance(val, (list, tuple, np.ndarray)):
                    return len(val)
                return 0

            def _normalize_series_std(item, y_arr=None):
                var = item.get("variable", "<unknown>")
                if "std" not in item or _is_missing_scalar(item.get("std")):
                    raise ValueError(
                        f"Series data item {var!r} requires 'std' in the JSON "
                        f"(one positive scalar applied to all samples, or a "
                        f"vector with the same length as the series)."
                    )
                n = _series_length(item, y_arr)
                if n < 1:
                    raise ValueError(
                        f"Series data item {var!r}: cannot set 'std' without "
                        f"series 'value' or loaded .npy data."
                    )
                std_raw = item["std"]
                if isinstance(std_raw, (int, float, np.integer, np.floating)):
                    std_scalar = float(std_raw)
                    if std_scalar <= 0.0 or not np.isfinite(std_scalar):
                        raise ValueError(
                            f"Series data item {var!r}: scalar 'std' must be "
                            f"finite and > 0, got {std_raw!r}."
                        )
                    item["std"] = [std_scalar] * n
                    return

                if isinstance(std_raw, (list, tuple, np.ndarray)):
                    std_arr = np.asarray(std_raw, dtype=np.float64).ravel()
                    if std_arr.size == 1:
                        std_scalar = float(std_arr[0])
                        if std_scalar <= 0.0 or not np.isfinite(std_scalar):
                            raise ValueError(
                                f"Series data item {var!r}: scalar 'std' must "
                                f"be finite and > 0, got {std_scalar!r}."
                            )
                        item["std"] = [std_scalar] * n
                        return
                    if std_arr.size != n:
                        raise ValueError(
                            f"Series data item {var!r}: 'std' length "
                            f"({std_arr.size}) does not match series length "
                            f"({n})."
                        )
                    if np.any(std_arr <= 0.0) or not np.all(np.isfinite(std_arr)):
                        raise ValueError(
                            f"Series data item {var!r}: every 'std' entry must "
                            f"be finite and > 0."
                        )
                    item["std"] = std_arr.tolist()
                    return

                raise ValueError(
                    f"Series data item {var!r}: 'std' must be a scalar or list, "
                    f"got {type(std_raw).__name__}."
                )

            def _has_embedded_series_value(item):
                if "value" not in item:
                    return False
                val = item.get("value")
                if _is_missing_scalar(val):
                    return False
                if isinstance(val, (list, tuple, np.ndarray)):
                    return len(val) > 0
                return True

            hydrated = []
            for raw in data_items:
                item = copy.deepcopy(raw)
                dtype = item.get("data_type")
                if dtype == "timeseries":
                    item["data_type"] = "series"
                    dtype = "series"
                if item.get("plot_type") == "timeseries":
                    item["plot_type"] = "series"
                if dtype != "series":
                    hydrated.append(item)
                    continue

                t_path = item.get("t_path")
                value_path = item.get("value_path")

                # Legacy: top-level vm_path / im_path; pick observable by variable name.
                if value_path is None:
                    vm_path = item.pop("vm_path", None)
                    im_path = item.pop("im_path", None)
                    if vm_path or im_path:
                        var = str(item.get("variable", ""))
                        if "I_tot" in var or var.endswith("/I_tot_pA"):
                            value_path = im_path or vm_path
                        else:
                            value_path = vm_path or im_path
                        if value_path:
                            item["value_path"] = value_path

                # Also handle legacy paths buried in source dict.
                src = item.get("source")
                if isinstance(src, dict):
                    if t_path is None:
                        t_path = src.pop("t_path", None)
                        if t_path:
                            item["t_path"] = t_path
                    if value_path is None:
                        vp = src.pop("value_path", None)
                        if vp is None:
                            vm_p = src.pop("vm_path", None)
                            im_p = src.pop("im_path", None)
                            var = str(item.get("variable", ""))
                            if "I_tot" in var or var.endswith("/I_tot_pA"):
                                vp = im_p or vm_p
                            else:
                                vp = vm_p or im_p
                        if vp:
                            value_path = vp
                            item["value_path"] = value_path
                    # Flatten description string back to source if dict is now empty
                    desc = src.get("description")
                    if desc and len(src) == 1:
                        item["source"] = desc
                    elif not src:
                        item.pop("source", None)

                if _has_embedded_series_value(item) and (t_path or value_path):
                    var = item.get("variable", "<unknown>")
                    raise ValueError(
                        f"Series data item {var!r} specifies both embedded 'value' "
                        f"and 't_path'/'value_path' (.npy files). Use one source only: "
                        f"either embed 'value' in the JSON or provide 't_path' and "
                        f"'value_path', not both."
                    )

                need_value = (
                    "value" not in item
                    or _is_missing_scalar(item.get("value"))
                )
                y_arr = None
                if need_value and t_path and value_path:
                    t_arr = np.load(t_path)
                    y_arr = np.asarray(np.load(value_path), dtype=np.float64)
                    item["value"] = y_arr.tolist()
                    if "obs_dt" not in item or _is_missing_scalar(item.get("obs_dt")):
                        item["obs_dt"] = (
                            float(np.mean(np.diff(t_arr))) if len(t_arr) > 1 else 1.0
                        )
                elif _has_embedded_series_value(item):
                    y_arr = np.asarray(item["value"], dtype=np.float64)

                _normalize_series_std(item, y_arr)

                hydrated.append(item)
            return hydrated

        # --- Case 1: Simple list of data items ---
        if type(json_obj) == list:
            gt_df = pd.DataFrame(json_obj)
            protocol_info = {"pre_times": [pre_time], 
                             "sim_times": [[sim_time]],
                             "params_to_change": {}}
            prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
            

        # --- Case 2: Dictionary structure ---
        elif type(json_obj) == dict:
            # Load Data Items (gt_df)
            if 'data_items' in json_obj.keys() or 'data_item' in json_obj.keys():
                data_items = json_obj.get('data_items', json_obj.get('data_item', []))
                data_items = _hydrate_series_data_items(data_items)
                gt_df = pd.DataFrame(data_items)
            else:
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")

            # Load Protocol Info
            if 'protocol_info' in json_obj.keys():
                protocol_info = copy.deepcopy(json_obj['protocol_info'])
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()
                protocol_info = {"pre_times": [pre_time], "sim_times": [[sim_time]], "params_to_change": {}}

            protocol_schema = {
                "pre_times": {"types": (list, tuple, np.ndarray), "default": [pre_time] if pre_time is not None else REQUIRED},
                "sim_times": {"types": (list, tuple, np.ndarray), "default": [[sim_time]] if sim_time is not None else REQUIRED},
                "params_to_change": {"types": (dict,), "default": {}},
                "offline_pre_time": {"types": (float, int), "default": None},
                "experiment_labels": {"types": (list, tuple, np.ndarray), "default": None},
                "experiment_ids": {"types": (list, tuple, np.ndarray), "default": None},
                "experiment_colors": {"types": (list, tuple, np.ndarray), "default": None},
                "comment": {"types": (str,), "default": None},
                "protocol_traces": {"types": (dict,), "default": {}},
                # The same waveforms written as Myokit-style events rather than
                # point tables; expanded into protocol_traces below.
                "protocol_shapes": {"types": (dict,), "default": {}},
            }

            unknown_protocol_keys = sorted(set(protocol_info.keys()) - set(protocol_schema.keys()))
            if len(unknown_protocol_keys) > 0:
                raise ValueError(
                    f"Unknown protocol_info keys not in schema: {unknown_protocol_keys}"
                )

            missing_protocol_required = []
            protocol_type_errors = []
            for key, rules in protocol_schema.items():
                allowed = rules["types"]
                default = rules["default"]

                if key not in protocol_info or _is_missing_scalar(protocol_info[key]):
                    if default == REQUIRED:
                        missing_protocol_required.append(key)
                    else:
                        protocol_info[key] = copy.deepcopy(default)
                        continue

                if protocol_info[key] is not None and not isinstance(protocol_info[key], allowed):
                    protocol_type_errors.append(
                        f"protocol_info['{key}']: expected {allowed}, got {type(protocol_info[key])}"
                    )

            if len(missing_protocol_required) > 0:
                raise ValueError(
                    f"Missing required protocol_info keys: {sorted(missing_protocol_required)}"
                )
            if len(protocol_type_errors) > 0:
                raise ValueError(
                    "Invalid protocol_info value types:\n" + "\n".join(protocol_type_errors)
                )

            validate_params_to_change(protocol_info)
            # Shapes become traces here, once, so every consumer downstream --
            # solver helpers, plotting, anything added later -- keeps seeing only
            # protocol_traces and needs no knowledge of shapes.
            materialise_shapes(protocol_info)
            validate_trace_references(protocol_info)

            # Load Prediction Info
            if 'prediction_items' in json_obj.keys():
                prediction_items = json_obj['prediction_items']
                if not isinstance(prediction_items, (list, tuple)):
                    raise ValueError(
                        f"prediction_items must be a list of dict entries, got {type(prediction_items)}"
                    )

                prediction_entry_schema = {
                    "variable": {"types": (str,), "default": REQUIRED},
                    "unit": {"types": (str,), "default": REQUIRED},
                    "name_for_plotting": {"types": (str,), "default": lambda entry: entry["variable"]},
                    "experiment_idx": {"types": (int, np.integer), "default": 0},
                }

                prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
                for entry_idx, raw_entry in enumerate(prediction_items):
                    if not isinstance(raw_entry, dict):
                        raise ValueError(
                            f"prediction_items[{entry_idx}] must be a dict, got {type(raw_entry)}"
                        )
                    entry = copy.deepcopy(raw_entry)

                    unknown_pred_keys = sorted(set(entry.keys()) - set(prediction_entry_schema.keys()))
                    if len(unknown_pred_keys) > 0:
                        raise ValueError(
                            f"Unknown keys in prediction_items[{entry_idx}] not in schema: {unknown_pred_keys}"
                        )

                    missing_pred_required = []
                    pred_type_errors = []
                    for key, rules in prediction_entry_schema.items():
                        allowed = rules["types"]
                        default = rules["default"]

                        if key not in entry or _is_missing_scalar(entry[key]):
                            if default == REQUIRED:
                                missing_pred_required.append(key)
                                continue
                            entry[key] = default(entry) if callable(default) else copy.deepcopy(default)

                        if not isinstance(entry[key], allowed):
                            pred_type_errors.append(
                                f"prediction_items[{entry_idx}]['{key}']: expected {allowed}, got {type(entry[key])}"
                            )

                    if len(missing_pred_required) > 0:
                        raise ValueError(
                            f"Missing required keys in prediction_items[{entry_idx}]: {sorted(missing_pred_required)}"
                        )
                    if len(pred_type_errors) > 0:
                        raise ValueError(
                            "Invalid prediction_items value types:\n" + "\n".join(pred_type_errors)
                        )

                    prediction_info['names'].append(entry['variable'])
                    prediction_info['units'].append(entry['unit'])
                    prediction_info['names_for_plotting'].append(entry['name_for_plotting'])
                    prediction_info['experiment_idxs'].append(entry['experiment_idx'])
            else:
                prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
            
        else:
            print(f"Error: unknown data type for imported json object of {type(json_obj)}")
            return None

        # Fill common optional fields so downstream processing can rely on defaults.
        if gt_df is not None:
            schema = {
                "variable": {"types": (str,), "default": REQUIRED},
                "name_for_plotting": {"types": (str,), "default": lambda df: df["variable"]},
                "data_type": {"types": (str,), "default": REQUIRED},
                "unit": {"types": (str,), "default": REQUIRED},
                "weight": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": 1.0},
                "operands": {"types": (list, tuple, np.ndarray), "default": REQUIRED},
                "operation": {"types": (str,), "default": None},
                "operation_kwargs": {"types": (dict,), "default": lambda df: [{} for _ in range(len(df))]},
                # Extra keyword arguments for the data_item's cost_type func (issue #84), the
                # cost-side counterpart of operation_kwargs. std and weight are supplied by CA
                # from the fields above and are rejected here -- see param_id.cost_kwargs.
                "cost_kwargs": {"types": (dict,), "default": lambda df: [{} for _ in range(len(df))]},
                "value": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": REQUIRED},
                "std": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": REQUIRED},
                "experiment_idx": {"types": (int, np.integer), "default": 0},
                "subexperiment_idx": {"types": (int, np.integer), "default": 0},
                "plot_type": {"types": (str,), "default": None},
                "plot_color": {"types": (str,), "default": None},
                "comment": {"types": (str,), "default": None},
                "cost_type": {"types": (str,), "default": "MSE"},
                "obs_type": {"types": (str,), "default": None},
                "frequencies": {"types": (list, tuple, np.ndarray, int, float, np.integer, np.floating), "default": None},
                # If omitted, phase weighting should follow the same weighting as amplitude.
                "phase_weight": {"types": (int, float, np.integer, np.floating, list, np.ndarray), "default": lambda df: df["weight"]},
                "phase": {"types": (list, tuple, np.ndarray, int, float, np.integer, np.floating), "default": None},
                "prob_dist_params": {"types": (dict,), "default": None},
                "obs_dt": {"types": (int, float, np.integer, np.floating), "default": None},
                "dt": {"types": (int, float, np.integer, np.floating), "default": None},
                "sample_rate": {"types": (int, float, np.integer, np.floating), "default": None},
                "species": {"types": (str,), "default": None},
                "location": {"types": (str,), "default": None},
                "source": {"types": (str, dict), "default": None},
                "t_path": {"types": (str,), "default": None},
                "value_path": {"types": (str,), "default": None},
            }

            unknown_cols = sorted(set(gt_df.columns) - set(schema.keys()))
            if len(unknown_cols) > 0:
                raise ValueError(
                    f"Unknown data_item keys not in schema: {unknown_cols}"
                )

            type_errors = []
            missing_required_cols = []
            # No data items at all is a valid obs_data: a protocol-only file says
            # how to drive the model without yet saying what to measure, which is
            # what an obs_data generated from a model's own protocol looks like
            # before its targets are added. There is nothing to validate, and the
            # column defaults below are derived from other columns -- so on an
            # empty frame they raised KeyError: 'variable' instead.
            for col, rules in ({} if len(gt_df) == 0 else schema).items():
                allowed = rules["types"]
                default = rules["default"]

                if col not in gt_df.columns:
                    if default == REQUIRED:
                        missing_required_cols.append(col)
                        continue
                    if callable(default):
                        gt_df[col] = pd.Series(default(gt_df), index=gt_df.index)
                    else:
                        gt_df[col] = default
                else:
                    if default != REQUIRED:
                        missing_mask = gt_df[col].apply(_is_missing_scalar)
                        if missing_mask.any():
                            if callable(default):
                                default_series = pd.Series(default(gt_df), index=gt_df.index)
                                gt_df[col] = gt_df[col].where(~missing_mask, default_series)
                            else:
                                gt_df[col] = gt_df[col].where(~missing_mask, default)

                for row_idx, val in gt_df[col].items():
                    if _is_missing_scalar(val):
                        continue
                    if not isinstance(val, allowed):
                        type_errors.append(
                            f"row {row_idx}, column '{col}': expected {allowed}, got {type(val)}"
                        )

            if len(missing_required_cols) > 0:
                raise ValueError(
                    f"Missing required data_item keys: {sorted(missing_required_cols)}"
                )

            if len(type_errors) > 0:
                raise ValueError(
                    "Invalid data_item value types:\n" + "\n".join(type_errors)
                )

        warn_if_casadi_nonzero_pre_time(
            model_type,
            pre_time=pre_time,
            pre_times=protocol_info.get('pre_times') if protocol_info is not None else None,
            offline_pre_time=protocol_info.get('offline_pre_time') if protocol_info is not None else None,
            method=method,
        )

        return {
            "gt_df": gt_df, 
            "protocol_info": protocol_info, 
            "prediction_info": prediction_info
        }

    def process_obs_info(self, gt_df, output_dir, dt):
        """
        Generates the detailed obs_info dictionary, including names, units, 
        plotting defaults, operations, and kwargs from the ground truth dataframe.
        """
        obs_info = {}
        
        # --- Simple Array Generation ---
        N = gt_df.shape[0]
        
        obs_info["obs_names"] = gt_df["variable"].tolist()
        obs_info["data_types"] = gt_df["data_type"].tolist()
        obs_info["units"] = gt_df["unit"].tolist()
        obs_info["experiment_idxs"] = [gt_df.iloc[II].get("experiment_idx", 0) for II in range(N)]
        obs_info["subexperiment_idxs"] = [gt_df.iloc[II].get("subexperiment_idx", 0) for II in range(N)]

        # --- Plotting Colors ---
        possible_colors = ['b', 'g', 'c', 'm', 'y', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange']
        obs_info["plot_colors"] = [gt_df.iloc[II].get("plot_color", possible_colors[II % len(possible_colors)]) 
                                        for II in range(N)]
        
        # --- Plotting Type Defaults (Logic preserved) ---
        obs_info["plot_type"] = []
        warning_printed = False
        for II in range(N):
            if "plot_type" not in gt_df.iloc[II].keys():
                if gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "series":
                    obs_info["plot_type"].append("series")
                elif gt_df.iloc[II]["data_type"] == "frequency":
                    obs_info["plot_type"].append("frequency")
                elif gt_df.iloc[II]["data_type"] == "plot_dist":
                    obs_info["plot_type"].append("horizontal")
                else:
                    print(f'data type {gt_df.iloc[II]["data_type"]} not recognised')
            else:
                obs_info["plot_type"].append(gt_df.iloc[II]["plot_type"])
                if obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    obs_info["plot_type"][II] = None

        # --- Operations (Mapping obs_type to operation) ---
        obs_info["operations"] = []
        obs_info["operands"] = []
        obs_info["operation_kwargs"] = [gt_df.iloc[II].get("operation_kwargs", {}) for II in range(N)]
        obs_info["cost_kwargs"] = [gt_df.iloc[II].get("cost_kwargs", {}) for II in range(N)]
        obs_info["freqs"] = [gt_df.iloc[II].get("frequencies") for II in range(N)]
        obs_info["names_for_plotting"] = [gt_df.iloc[II].get("name_for_plotting", obs_info["obs_names"][II]) for II in range(N)]

        for II in range(N):
            op = gt_df.iloc[II].get("operation")
            obs_type = gt_df.iloc[II].get("obs_type")
            operands = gt_df.iloc[II].get("operands")
            # Filter out empty operands, should tolerate empty operands exists
            if isinstance(operands, (list, tuple, np.ndarray)):
                operands = [op for op in list(operands) if op]
                
            if op in ["Null", "None", "null", "none", "", "nan", np.nan, None]:
                if obs_type in ["series", "frequency"]:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
                elif obs_type in ["min", "max", "mean"]: 
                    obs_info["operations"].append(obs_type)
                    obs_info["operands"].append([gt_df.iloc[II]["variable"]])
                else:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
            else:
                obs_info["operations"].append(op)
                obs_info["operands"].append(operands)

        # --- Weights and Cost Types ---
        weights = gt_df["weight"].to_numpy()
        data_types = np.array(obs_info["data_types"])
        
        obs_info["num_obs"] = N
        obs_info["weight_const_vec"] = weights[data_types == "constant"]
        obs_info["weight_series_vec"] = weights[data_types == "series"]
        obs_info["weight_amp_vec"] = weights[data_types == "frequency"]
        obs_info["weight_prob_dist_vec"] = weights[data_types == "prob_dist"]

        phase_weights = gt_df.apply(
            lambda row: row["phase_weight"] if row.get("phase_weight") is not None else row["weight"],
            axis=1,
        )
        obs_info["weight_phase_vec"] = phase_weights[data_types == "frequency"].to_numpy()

        obs_info["cost_type"] = [gt_df.iloc[II].get("cost_type", "MSE") for II in range(N)]

        obs_info = self.get_ground_truth_values(gt_df, obs_info, output_dir, dt)
        
        return obs_info
    
    def get_ground_truth_values(self, gt_df, obs_info, output_dir, dt):

        # _______ First we access data for constant values

        # TODO make all of the below lists instead of arrays? So we can have different sized entries.

        ground_truth_const = np.array([gt_df.iloc[II]["value"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "constant"])

        # _______ Then for time series
        ground_truth_series = [np.array(gt_df.iloc[II]["value"]) for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "series"]

        # _______ Then for frequency series
        ground_truth_amp = np.array([gt_df.iloc[II]["value"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "frequency"])

        # then for ground truth probability distributions
        ground_truth_prob_dist_params = np.array([gt_df.iloc[II]["prob_dist_params"] for II in range(gt_df.shape[0])
                                            if gt_df.iloc[II]["data_type"] == "prob_dist"])


        # _______ and the phase of the freq data
        ground_truth_phase_list = []
        for II in range(gt_df.shape[0]):
            if gt_df.iloc[II]["data_type"] == "frequency":
                if "phase" not in gt_df.iloc[II].keys():
                    ground_truth_phase_list.append(None)
                else:
                    ground_truth_phase_list.append(gt_df.iloc[II]["phase"])
        ground_truth_phase = np.array(ground_truth_phase_list)

        # get the dt for the series data
        dt_list = []
        for II in range(gt_df.shape[0]):
            if gt_df.iloc[II]["data_type"] == "series":
                if "obs_dt" not in gt_df.iloc[II].keys():
                    print("dt not found in obs_data.json for series data, exiting")
                    exit()
                dt_list.append(gt_df.iloc[II]["obs_dt"])
        
        obs_info["obs_dt"] = np.array(dt_list)
        
        if len(obs_info["obs_dt"]) > 0:
            if min(obs_info["obs_dt"]) < dt:
                print("one of the dt in obs_data.json is less than the dt in user_inputs.yaml, the output timestep"
                    "defined in user_inputs.yaml must be less than the smallest dt for your data. Exiting")
                exit()

        # The std for the different observables
        obs_info["std_const_vec"] = np.array([gt_df.iloc[II]["std"] for II in range(gt_df.shape[0])
                                       if gt_df.iloc[II]["data_type"] == "constant"])

        obs_info["std_series_vec"] = [np.array(gt_df.iloc[II]["std"]) for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "series"]

        obs_info["std_amp_vec"] = np.array([gt_df.iloc[II]["std"] for II in range(gt_df.shape[0])
                                        if gt_df.iloc[II]["data_type"] == "frequency"])

        # if len(ground_truth_series) > 0:
            # TODO what if we have ground truths of different size or sample rate?
            # ground_truth_series = np.stack(ground_truth_series)
            # removed because we have data of different sizes

        if len(ground_truth_amp) > 0:
            ground_truth_amp = np.stack(ground_truth_amp)

        if len(ground_truth_phase) > 0:
            ground_truth_phase = np.stack(ground_truth_phase)

        if rank == 0:
            np.save(os.path.join(output_dir, 'ground_truth_const.npy'), ground_truth_const)
            if len(ground_truth_series) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_series.npy'), 
                        np.array(ground_truth_series, dtype=object), allow_pickle=True)
            if len(ground_truth_amp) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_amp.npy'), ground_truth_amp)
            if len(ground_truth_phase) > 0:
                np.save(os.path.join(output_dir, 'ground_truth_phase.npy'), ground_truth_phase)

        obs_info["ground_truth_const"] = ground_truth_const
        obs_info["ground_truth_prob_dist_params"] = ground_truth_prob_dist_params
        obs_info["ground_truth_series"] = ground_truth_series
        obs_info["ground_truth_amp"] = ground_truth_amp
        obs_info["ground_truth_phase"] = ground_truth_phase

        # create a mapping between const_idx and the obs_idx
        const_count = 0
        series_count = 0
        freq_count = 0
        prob_dist_count = 0
        obs_info["const_idx_to_obs_idx"] = []
        obs_info["series_idx_to_obs_idx"] = []
        obs_info["freq_idx_to_obs_idx"] = []
        obs_info["prob_dist_idx_to_obs_idx"] = []
        for obs_idx in range(obs_info["num_obs"]):
            if obs_info["data_types"][obs_idx] == "constant":
                obs_info["const_idx_to_obs_idx"].append(obs_idx)
                const_count += 1
            elif obs_info["data_types"][obs_idx] == "series":
                obs_info["series_idx_to_obs_idx"].append(obs_idx)
                series_count += 1
            elif obs_info["data_types"][obs_idx] == "frequency":
                obs_info["freq_idx_to_obs_idx"].append(obs_idx)
                freq_count += 1
            elif obs_info["data_types"][obs_idx] == "prob_dist":
                obs_info["prob_dist_idx_to_obs_idx"].append(obs_idx)
                prob_dist_count += 1

        return obs_info

    def process_protocol_and_weights(self, gt_df, protocol_info, dt):
        """
        Calculates time totals, validates protocol labels/colors, and generates 
        the scaled weight maps for experiment/subexperiment cost calculation.
        """
        protocol = protocol_info
        df = gt_df
        
        # --- Protocol Info Preprocessing ---
        protocol['num_experiments'] = len(protocol["sim_times"])
        protocol['num_sub_per_exp'] = [len(protocol["sim_times"][exp_idx]) for exp_idx in range(protocol["num_experiments"])]
        protocol['num_sub_total'] = sum(protocol['num_sub_per_exp'])

        validate_params_to_change(protocol)
        
        protocol["total_sim_times_per_exp"] = []
        protocol["tSims_per_exp"] = []
        protocol["num_steps_total_per_exp"] = []

        for exp_idx in range(protocol['num_experiments']):
            total_sim_time = np.sum(protocol["sim_times"][exp_idx])
            num_steps_total = int(total_sim_time / dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            
            protocol["total_sim_times_per_exp"].append(total_sim_time)
            protocol["tSims_per_exp"].append(tSim_per_exp)
            protocol["num_steps_total_per_exp"].append(num_steps_total)
            
        # --- Protocol Info Validation ---
        N_exp = protocol['num_experiments']
        
        if "experiment_colors" not in protocol or protocol["experiment_colors"] is None:
            protocol["experiment_colors"] = ['r'] * N_exp
        elif len(protocol["experiment_colors"]) != N_exp:
            print('Error: experiment_colors length does not match num_experiments, exiting')
            exit()
            
        if "experiment_labels" not in protocol or protocol["experiment_labels"] is None:
            protocol["experiment_labels"] = [None] * N_exp
        elif len(protocol["experiment_labels"]) != N_exp:
            print('Error: experiment_labels length does not match num_experiments, exiting')
            exit()
        
        if "experiment_ids" not in protocol or protocol["experiment_ids"] is None:
            protocol["experiment_ids"] = [None] * N_exp
        elif len(protocol["experiment_ids"]) != N_exp:
            print('Error: experiment_ids length does not match num_experiments, exiting')
            exit()

        # --- Weight Mapping Initialization ---
        
        # Ensure experiment_idx and subexperiment_idx exist in the DataFrame
        # IMPORTANT: These columns must be added safely if they don't exist
        df["experiment_idx"] = df.apply(lambda row: row.get("experiment_idx", 0), axis=1)
        df["subexperiment_idx"] = df.apply(lambda row: row.get("subexperiment_idx", 0), axis=1)

        # Initialize nested lists for weight maps (one list per data type)
        const_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        series_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        amp_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        phase_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        prob_dist_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]

        
        # --- Calculate Scaled Weight Maps ---
        
        for exp_idx in range(N_exp):
            for this_sub_idx in range(protocol['num_sub_per_exp'][exp_idx]):
                
                # Mask to find observations belonging to the current experiment/subexperiment
                mask = (df["experiment_idx"] == exp_idx) & (df["subexperiment_idx"] == this_sub_idx)
                
                # Iterate over all possible data types
                for data_type, weight_map in [
                    ("constant", const_map), ("series", series_map), ("frequency", amp_map), ("prob_dist", prob_dist_map)
                ]:
                    # Create the full weight vector (Weight if matched, 0.0 otherwise)
                    full_weights = np.where(mask & (df["data_type"] == data_type), df["weight"], 0.0)
                    weight_map[exp_idx][this_sub_idx] = full_weights
                
                # Handle phase map separately
                freq_mask = mask & (df["data_type"] == "frequency")
                # Use "phase_weight" if present, otherwise use "weight", or 0.0
                phase_weights = np.where(
                    freq_mask,
                    df.apply(
                        lambda row: row["phase_weight"] if row.get("phase_weight") is not None else row["weight"],
                        axis=1,
                    ),
                    0.0,
                )
                phase_map[exp_idx][this_sub_idx] = phase_weights

        # --- Store Final Maps in protocol_info ---
        protocol["scaled_weight_const_from_exp_sub"] = const_map
        protocol["scaled_weight_series_from_exp_sub"] = series_map
        protocol["scaled_weight_amp_from_exp_sub"] = amp_map
        protocol["scaled_weight_phase_from_exp_sub"] = phase_map
        protocol["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        
        return protocol
    
    @staticmethod
    def _qname_to_vessel_and_param(qname):
        """Split a 'component/param' qname. rsplit so a component containing '/' still works."""
        if '/' not in qname:
            raise ValueError(
                f"params_for_id target {qname!r} is not a 'component/param' name. Targets are "
                f"full qualified names, e.g. 'aortic_root/C' or 'global/q_lv_init'.")
        vessel, param = qname.rsplit('/', 1)
        return vessel.strip(), param.strip()

    @classmethod
    def params_for_id_csv_to_json(cls, csv_text_or_path):
        """Convert a legacy params_for_id CSV into the canonical JSON structure.

        Pure: text (or a path) in, dict out. No model, no solver, no simulation helper -- so a
        front-end can show a user's existing CSV as JSON in an editor without loading anything.
        The mapping is documented in the tutorial as well as implemented here, because a tool
        without circulatory_autogen on sys.path has to be able to reproduce it.

        Per row:
            vessel_name='a b', param_name='C'  ->  targets: ['a/C', 'b/C']
            min/max/param_type/name_for_plotting/comment  ->  same keys
            prior + prior_mean/prior_std/...   ->  prior + prior_params: {...}
            unbounded                          ->  unbounded (bool)
            (none)                             ->  name, defaulting to the first target
        """
        import io

        if isinstance(csv_text_or_path, str) and ('\n' in csv_text_or_path
                                                  or ',' in csv_text_or_path.split('\n')[0]) \
                and not os.path.exists(csv_text_or_path):
            handle = io.StringIO(csv_text_or_path)
        else:
            handle = csv_text_or_path

        df = pd.read_csv(handle, dtype=str)
        df = df.rename(columns=lambda c: c.strip())
        for column in df.columns:
            df[column] = df[column].apply(lambda v: v.strip() if isinstance(v, str) else v)

        def _present(value):
            if value is None:
                return False
            if isinstance(value, float) and np.isnan(value):
                return False
            return str(value).strip() != ''

        params = []
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            vessels = [v for v in str(row.get('vessel_name', '') or '').split() if v]
            param_name = str(row.get('param_name', '') or '').strip()
            if not vessels or not param_name:
                raise ValueError(
                    f'params_for_id CSV row {idx}: both vessel_name and param_name are required '
                    f'(got vessel_name={row.get("vessel_name")!r}, param_name={param_name!r}).')

            entry = {'targets': [f'{v}/{param_name}' for v in vessels]}
            entry['name'] = entry['targets'][0]

            for key in ('param_type', 'name_for_plotting', 'comment', 'prior'):
                if key in df.columns and _present(row.get(key)):
                    entry[key] = str(row[key]).strip()
            for key in ('min', 'max'):
                if key in df.columns and _present(row.get(key)):
                    entry[key] = str(row[key]).strip()
            if PARAM_UNBOUNDED_COLUMN in df.columns and _present(row.get(PARAM_UNBOUNDED_COLUMN)):
                entry[PARAM_UNBOUNDED_COLUMN] = _truthy_flag(row[PARAM_UNBOUNDED_COLUMN])

            prior_params = {name: str(row[name]).strip()
                            for name in PARAMS_FOR_ID_CSV_PRIOR_COLUMNS
                            if name in df.columns and _present(row.get(name))}
            if prior_params:
                entry['prior_params'] = prior_params

            params.append(entry)

        return {'version': PARAMS_FOR_ID_JSON_VERSION, 'defaults': {}, 'params': params}

    @classmethod
    def resolve_params_for_id_doc(cls, doc, modifier_funcs=None):
        """Validate a params_for_id document and fold `defaults` into each entry.

        Resolution is a shallow per-key override -- an entry's own key wins over `defaults`, which
        wins over whatever circulatory_autogen derives later (the prior machinery's default_expr).
        `prior_params` merges per key too, so a defaults block setting prior_std does not wipe an
        entry's prior_mean.

        ``modifier_funcs`` is the registry modifier entries validate against (operation names,
        declared inputs); defaults to the built-in + funcs_user registry. Pass
        ``get_modifier_funcs(external_path)`` to also accept externally-defined functions.
        """
        if modifier_funcs is None:
            modifier_funcs = get_modifier_funcs(None)
        if not isinstance(doc, dict):
            raise ValueError(
                f'params_for_id JSON must be an object with a "params" list, got '
                f'{type(doc).__name__}.')

        version = doc.get('version', PARAMS_FOR_ID_JSON_VERSION)
        if int(version) != PARAMS_FOR_ID_JSON_VERSION:
            raise ValueError(
                f'params_for_id JSON version {version} is not supported by this version of '
                f'circulatory_autogen (expected {PARAMS_FOR_ID_JSON_VERSION}).')

        params = doc.get('params')
        if not isinstance(params, list) or not params:
            raise ValueError('params_for_id JSON needs a non-empty "params" list.')

        defaults = doc.get('defaults') or {}
        if not isinstance(defaults, dict):
            raise ValueError(
                f'params_for_id "defaults" must be an object, got {type(defaults).__name__}.')
        unknown_defaults = set(defaults) - PARAMS_FOR_ID_ENTRY_KEYS
        if unknown_defaults:
            raise ValueError(
                f'params_for_id "defaults" has unknown key(s) {sorted(unknown_defaults)}. '
                f'Valid keys are the entry keys: {sorted(PARAMS_FOR_ID_ENTRY_KEYS)}.')

        resolved = []
        seen_names = {}
        for idx, raw in enumerate(params):
            if not isinstance(raw, dict):
                raise ValueError(
                    f'params_for_id entry {idx} must be an object, got {type(raw).__name__}.')
            unknown = set(raw) - PARAMS_FOR_ID_ENTRY_KEYS
            if unknown:
                raise ValueError(
                    f'params_for_id entry {idx} has unknown key(s) {sorted(unknown)}. Valid '
                    f'keys: {sorted(PARAMS_FOR_ID_ENTRY_KEYS)}.')

            entry = {k: v for k, v in defaults.items() if k != 'prior_params'}
            entry.update({k: v for k, v in raw.items() if k != 'prior_params'})
            merged_prior = dict(defaults.get('prior_params') or {})
            merged_prior.update(raw.get('prior_params') or {})
            if merged_prior:
                entry['prior_params'] = merged_prior

            has_targets = entry.get('targets') is not None
            has_modifies = entry.get('modifies') is not None
            if has_targets and has_modifies:
                raise ValueError(
                    f'params_for_id entry {idx} sets both "targets" and "modifies". An entry is '
                    f'either a parameter (targets) or a modifier of parameters (modifies), not '
                    f'both.')
            if not has_targets and not has_modifies:
                raise ValueError(
                    f'params_for_id entry {idx} needs a non-empty "targets" list of '
                    f'component/param names (or "modifies" for a modifier entry).')

            key = 'modifies' if has_modifies else 'targets'
            names = entry.get(key)
            if isinstance(names, str):
                names = [names]
            if not names or not isinstance(names, list):
                raise ValueError(
                    f'params_for_id entry {idx} needs a non-empty "{key}" list of '
                    f'component/param names.')
            names = [str(t).strip() for t in names]
            for qname in names:
                cls._qname_to_vessel_and_param(qname)
            entry[key] = names

            if has_modifies:
                operation = entry.get('operation') or DEFAULT_PARAM_MODIFIER_OPERATION
                if operation not in modifier_funcs:
                    raise ValueError(
                        f'params_for_id entry {idx} has unknown operation {operation!r}. '
                        f'Registered modifier functions: {sorted(modifier_funcs)} (built-ins '
                        f'plus funcs_user/modifier_funcs_user.py and '
                        f'modifier_funcs_external_path).')
                entry['operation'] = operation
                entry['inputs'] = cls._validate_modifier_inputs(
                    idx, operation, modifier_funcs[operation], entry.get('inputs'))
                # A multiplier range that straddles zero flips the sign of every target
                # somewhere inside it. That is legal arithmetic and almost never intended.
                try:
                    lo, hi = float(entry.get('min')), float(entry.get('max'))
                except (TypeError, ValueError):
                    lo = hi = None
                if lo is not None and operation == 'scale' and lo <= 0 < hi:
                    warnings.warn(
                        f'params_for_id entry {idx} ({entry.get("name", "?")}) is a scale '
                        f'modifier with min={lo} and max={hi}, a range crossing zero. Every '
                        f'target changes sign inside it; scale bounds are multipliers, not '
                        f'physical values.')
            elif entry.get('operation') is not None:
                raise ValueError(
                    f'params_for_id entry {idx} sets "operation" but has no "modifies". '
                    f'operation only applies to a modifier entry.')
            elif entry.get('inputs') is not None:
                raise ValueError(
                    f'params_for_id entry {idx} sets "inputs" but has no "modifies". '
                    f'inputs supply a modifier function\'s extra model constants.')

            entry.setdefault('name', entry[key][0])
            name = str(entry['name'])
            if name in seen_names:
                raise ValueError(
                    f'params_for_id entry {idx} reuses the name {name!r}, already used by entry '
                    f'{seen_names[name]}. Entry names identify a parameter and must be unique.')
            seen_names[name] = idx
            entry['name'] = name
            resolved.append(entry)

        cls._validate_modifier_relationships(resolved)
        return resolved

    @classmethod
    def _validate_modifier_inputs(cls, idx, operation, fn, raw_inputs):
        """Normalise and validate a modifier entry's ``inputs`` against the function's
        declaration.

        The function declares each input's name and type on the decorator
        (``@modifier_func(inputs={'subtract': 'list'})``); the entry supplies the model
        qname(s) whose *default* values the function will receive: a single qname string for
        ``'float'``, a non-empty list of qnames for ``'list'``. Everything is checked here, at
        parse time, so a typo'd input never reaches a calibration as a silently-absent kwarg.
        """
        declared = dict(getattr(fn, 'modifier_inputs', {}) or {})
        raw_inputs = dict(raw_inputs or {})

        unknown = set(raw_inputs) - set(declared)
        if unknown:
            raise ValueError(
                f'params_for_id entry {idx}: operation {operation!r} does not take input(s) '
                f'{sorted(unknown)}. Declared inputs: {declared or "none"}.')
        missing = set(declared) - set(raw_inputs)
        if missing:
            raise ValueError(
                f'params_for_id entry {idx}: operation {operation!r} requires input(s) '
                f'{sorted(missing)} ({ {k: declared[k] for k in sorted(missing)} }), naming '
                f'the model constant(s) whose default values the function receives.')

        normalised = {}
        for name, kind in declared.items():
            value = raw_inputs[name]
            if kind == 'float':
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f'params_for_id entry {idx}: input {name!r} of operation '
                        f'{operation!r} is type "float" and takes a single component/param '
                        f'qname string, got {value!r}.')
                qnames = [value.strip()]
                normalised[name] = qnames[0]
            else:  # 'list'
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or not value:
                    raise ValueError(
                        f'params_for_id entry {idx}: input {name!r} of operation '
                        f'{operation!r} is type "list" and takes a non-empty list of '
                        f'component/param qnames, got {value!r}.')
                qnames = [str(q).strip() for q in value]
                normalised[name] = qnames
            for qname in qnames:
                cls._qname_to_vessel_and_param(qname)
        return normalised

    @staticmethod
    def _validate_modifier_relationships(entries):
        """Cross-entry rules that only make sense once every entry is known.

        Both are refusals rather than warnings because both produce a calibration that runs,
        converges and means nothing.
        """
        free_owner = {}
        for entry in entries:
            for qname in entry.get('targets', []):
                free_owner.setdefault(qname, entry['name'])

        modifier_names = {e['name'] for e in entries if e.get('modifies')}

        # 3. Two modifiers on the same parameter multiply: p = theta_1 * theta_2 * baseline, so
        #    only the product is identifiable and each factor alone is meaningless -- the same
        #    flat ridge as rule 1, reached a different way.
        modified_by = {}
        for entry in entries:
            for qname in entry.get('modifies', []):
                if qname in modified_by:
                    raise ValueError(
                        f"params_for_id: '{qname}' is modified by both "
                        f"'{modified_by[qname]}' and '{entry['name']}'. Two modifiers on one "
                        f"parameter multiply, so only their product is identifiable and neither "
                        f"factor means anything on its own. Combine them into one modifier.")
                modified_by[qname] = entry['name']

        for entry in entries:
            modifies = entry.get('modifies')
            if not modifies:
                continue
            for qname in modifies:
                # 1. A modified parameter must not also be calibrated freely. (theta, p) and
                #    (theta*k, p/k) give an identical cost, so the optimiser wanders a flat
                #    ridge and both reported values are meaningless.
                if qname in free_owner:
                    raise ValueError(
                        f"params_for_id: '{qname}' is modified by '{entry['name']}' and is also "
                        f"a free parameter in entry '{free_owner[qname]}'. That is structurally "
                        f"unidentifiable -- scaling the modifier and dividing the free parameter "
                        f"by the same factor gives an identical cost, so neither value means "
                        f"anything. Remove one of the two entries.")
                # 2. One level, no chains: a modifier of a modifier has no defined baseline.
                if qname in modifier_names:
                    raise ValueError(
                        f"params_for_id: '{entry['name']}' modifies '{qname}', which is itself a "
                        f"modifier. Modifiers apply to model parameters only -- chains are not "
                        f"supported.")

    def get_param_id_info(self, params_for_id_path, idxs_to_ignore= None):
    
        if not params_for_id_path:
            print(f'params_for_id_path cannot be None, exiting')
            return None

        # One code path behind the front door: a .csv is converted to the JSON structure on read,
        # and everything downstream sees only resolved JSON entries.
        if str(params_for_id_path).lower().endswith('.json'):
            with open(params_for_id_path, 'r') as f:
                doc = json.load(f)
        else:
            doc = self.params_for_id_csv_to_json(params_for_id_path)
        entries = self.resolve_params_for_id_doc(
            doc, modifier_funcs=get_modifier_funcs(self.modifier_funcs_external_path))
        return self._build_param_id_info_from_entries(entries, idxs_to_ignore=idxs_to_ignore)

    def get_param_id_info_from_entries(self, params_for_id_entries, idxs_to_ignore=None):
        """
        Build param_id_info from a list/dict of parameter entries.
        Each entry should include: vessel_name, param_name, min, max.
        """
        if params_for_id_entries is None:
            print('params_for_id_entries cannot be None, exiting')
            return None

        # Allow callers to pass a dict wrapper
        if isinstance(params_for_id_entries, dict):
            if "params_for_id_path" in params_for_id_entries:
                return self.get_param_id_info(params_for_id_entries["params_for_id_path"],
                                              idxs_to_ignore=idxs_to_ignore)
            if "params" in params_for_id_entries:
                params_for_id_entries = params_for_id_entries["params"]

        if not isinstance(params_for_id_entries, list):
            raise ValueError("params_for_id_entries must be a list of dicts or include params_for_id_path")

        input_params = pd.DataFrame(params_for_id_entries)
        return self._build_param_id_info_from_df(input_params, idxs_to_ignore=idxs_to_ignore)

    def _build_param_id_info_from_entries(self, entries, idxs_to_ignore=None):
        """Build param_id_info from resolved params_for_id entries (the canonical JSON shape).

        This is the single builder; the CSV and the programmatic dict API both convert to entries
        first. Output shape is unchanged from the CSV-driven builder it replaces -- param_names is
        still a list of qname lists, one per calibrated variable -- so every consumer, including
        #376's grouped set_param_vals and the Sobol split, keeps working untouched.
        """
        if not entries:
            raise ValueError("No parameter entries provided")

        if idxs_to_ignore is not None:
            ignore = set(idxs_to_ignore)
            entries = [e for i, e in enumerate(entries) if i not in ignore]
            if not entries:
                raise ValueError("No parameter entries provided")

        N_params = len(entries)
        param_id_info = {}
        # A modifier looks exactly like a grouped row to every consumer: one variable to the
        # sampler and to the optimiser, N parameters to set. The only difference is what value
        # each of the N receives, which is resolved when the values are expanded (see
        # expand_modifier_param_vals) rather than here.
        param_id_info["param_names"] = [
            list(e["modifies"]) if e.get("modifies") else list(e["targets"]) for e in entries]

        # Simplified names for the generator, per target rather than per entry. Deciding the whole
        # row from the first vessel meant a row mixing 'global' with named vessels emitted one gen
        # name and dropped the rest, while param_names kept all of them, so the two positional
        # lists stopped describing the same parameters (#350).
        param_names_for_gen = []
        for entry, qnames in zip(entries, param_id_info["param_names"]):
            gen = []
            for qname in qnames:
                vessel, param = self._qname_to_vessel_and_param(qname)
                gen.append(param if vessel == 'global' else f'{param}_{vessel}')
            param_names_for_gen.append(gen)

        def _numeric(key):
            out = np.empty(N_params, dtype=float)
            for II, entry in enumerate(entries):
                raw = entry.get(key)
                try:
                    out[II] = float(raw) if raw is not None and str(raw).strip() != '' else np.nan
                except (TypeError, ValueError):
                    out[II] = np.nan
            return out

        param_id_info["param_mins"] = _numeric("min")
        param_id_info["param_maxs"] = _numeric("max")

        def _plot_name(entry):
            raw = entry.get("name_for_plotting")
            if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
                # A modifier falls back to its own name, not to one of the parameters it
                # modifies -- theta is its own quantity (dimensionless for scale) and labelling
                # it with a target's name would misreport what was calibrated.
                return entry["name"] if entry.get("modifies") else entry["targets"][0]
            return raw

        param_id_info["param_names_for_plotting"] = np.array(
            [_plot_name(entry) for entry in entries])

        param_id_info["param_prior_types"] = np.array([
            normalise_prior_type(entries[II].get("prior"), row_idx=II) for II in range(N_params)
        ])

        # normalise_prior_params takes anything with .get(), so the entry's prior_params mapping
        # is passed straight in -- the hyper-parameters live under their own key in JSON, but the
        # validation that owns which prior takes which is unchanged.
        # min/max travel with the hyper-parameters: normalise_prior_params checks a centre
        # declared `within_bounds` against the row's own range, and every prior is truncated to
        # [min, max], so a mean outside it describes a peak the sampler can never reach. Passing
        # prior_params alone silently disabled that check (#365).
        param_id_info["param_prior_params"] = [
            normalise_prior_params(
                param_id_info["param_prior_types"][II],
                {**dict(entries[II].get("prior_params") or {}),
                 'min': entries[II].get('min'), 'max': entries[II].get('max')},
                row_idx=II)
            for II in range(N_params)
        ]

        param_id_info["param_unbounded"] = np.array([
            _truthy_flag(entries[II].get(PARAM_UNBOUNDED_COLUMN)) for II in range(N_params)
        ])
        for II in range(N_params):
            if not param_id_info["param_unbounded"][II]:
                if not np.isfinite(param_id_info["param_mins"][II]) or \
                        not np.isfinite(param_id_info["param_maxs"][II]):
                    raise ValueError(
                        f"params_for_id row {II}: min and max are required unless "
                        f"'{PARAM_UNBOUNDED_COLUMN}' is set.")
                continue
            lo, hi = derive_bounds_from_prior(
                param_id_info["param_prior_types"][II],
                param_id_info["param_prior_params"][II],
                row_idx=II,
            )
            param_id_info["param_mins"][II] = lo
            param_id_info["param_maxs"][II] = hi

        param_id_info["param_names_for_gen"] = param_names_for_gen
        param_id_info["param_entry_names"] = [e["name"] for e in entries]

        # One display label per calibrated variable. A grouped row joins its qnames; a modifier
        # uses its own name. Downstream (the SALib problem, plots) reads this rather than
        # rebuilding it, so the two cannot drift.
        param_id_info["param_labels"] = [
            str(param_id_info["param_names_for_plotting"][II]) if entries[II].get("modifies")
            else '+'.join(param_id_info["param_names"][II])
            for II in range(N_params)
        ]

        # The modifier record downstream tools code against. `index` is a position in
        # param_names/param_labels and is computed *after* idxs_to_ignore filtering, so it is
        # always valid for the param_id_info it ships with -- but match on `name` if you carry a
        # modifier between two different builds. `baselines` is filled in once a simulation
        # helper is available (resolve_modifier_baselines); it is None until then rather than
        # absent, so a consumer can tell "not resolved yet" from "no baseline".
        param_id_info["modifiers"] = [
            {
                "index": II,
                "name": entries[II]["name"],
                "operation": entries[II].get("operation", DEFAULT_PARAM_MODIFIER_OPERATION),
                "targets": list(entries[II]["modifies"]),
                "baselines": None,
                # The model constants the modifier function's declared inputs name (qnames);
                # their default values land in `resolved_inputs` alongside the baselines.
                "inputs": dict(entries[II].get("inputs") or {}),
                "resolved_inputs": None,
                # Per-target affine coefficients of p_i = a_i*theta + b_i, probed at resolve
                # time: a_i is the gradient chain-rule weight, and inverting at the first
                # target's baseline gives theta's starting value.
                "affine": None,
            }
            for II in range(N_params) if entries[II].get("modifies")
        ]
        # Recorded so resolve_modifier_baselines / expand_modifier_param_vals load the same
        # registry (including external functions) that the entries were validated against.
        param_id_info["modifier_funcs_external_path"] = getattr(
            self, 'modifier_funcs_external_path', None)

        return param_id_info

    def _build_param_id_info_from_df(self, input_params, idxs_to_ignore=None):
        """Adapter for the programmatic entries API (vessel_name / param_name dicts).

        The documented in-code form of params_for_id is a list of
        {vessel_name, param_name, min, max, name_for_plotting}, and vessel_name may be a list
        sharing one calibrated value. That is converted to canonical entries here so it goes
        through the same builder as the CSV and the JSON -- one code path, three front doors.
        """
        if input_params is None or getattr(input_params, 'empty', False):
            raise ValueError("No parameter entries provided")

        required_cols = {"vessel_name", "param_name", "min", "max"}
        missing = required_cols - set(input_params.columns)
        if missing:
            raise ValueError(f"params_for_id is missing required columns: {sorted(list(missing))}")

        def _vessels(val):
            if isinstance(val, list):
                return [str(v).strip() for v in val]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return []
            return [v.strip() for v in str(val).strip().split() if v.strip()]

        def _present(value):
            if value is None:
                return False
            if isinstance(value, float) and np.isnan(value):
                return False
            return str(value).strip() != ''

        entries = []
        for II in range(input_params.shape[0]):
            row = input_params.iloc[II]
            vessels = _vessels(row["vessel_name"])
            param_name = str(row["param_name"]).strip()
            if not vessels or not param_name:
                raise ValueError(
                    f'params_for_id entry {II}: both vessel_name and param_name are required.')
            entry = {'targets': [f'{v}/{param_name}' for v in vessels]}
            entry['name'] = entry['targets'][0]
            for key in ('param_type', 'name_for_plotting', 'comment', 'prior', 'min', 'max'):
                if key in input_params.columns and _present(row.get(key)):
                    entry[key] = row[key]
            if PARAM_UNBOUNDED_COLUMN in input_params.columns \
                    and _present(row.get(PARAM_UNBOUNDED_COLUMN)):
                entry[PARAM_UNBOUNDED_COLUMN] = _truthy_flag(row[PARAM_UNBOUNDED_COLUMN])
            prior_params = {name: row[name] for name in PARAMS_FOR_ID_CSV_PRIOR_COLUMNS
                            if name in input_params.columns and _present(row.get(name))}
            if prior_params:
                entry['prior_params'] = prior_params
            entries.append(entry)

        # Names come from the first target here, so duplicates are possible in a way the JSON
        # front door forbids; de-duplicate positionally rather than refusing a form that has
        # always been legal.
        seen = {}
        for entry in entries:
            base = entry['name']
            if base in seen:
                seen[base] += 1
                entry['name'] = f'{base}#{seen[base]}'
            else:
                seen[base] = 0

        return self._build_param_id_info_from_entries(entries, idxs_to_ignore=idxs_to_ignore)

    def save_param_names(self, param_id_info, output_dir):
        """
        Saves the generated parameter names and generator names to CSV files.
        Requires the dictionary returned by _process_param_info.
        """
        if rank == 0:
            # 1. Save param_names (vessel_name/param_name format)
            param_names_path = os.path.join(output_dir, 'param_names.csv')
            with open(param_names_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names"])
            
            # 2. Save param_names_for_gen (simplified format)
            param_gen_path = os.path.join(output_dir, 'param_names_for_gen.csv')
            with open(param_gen_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names_for_gen"])

            # 3. Save the modifier records (see save_param_modifiers). At this point the
            #    baselines may still be None -- parsing happens before a simulation helper
            #    exists -- so the calibration run re-saves once they are resolved.
            save_param_modifiers(param_id_info, output_dir)
        return




