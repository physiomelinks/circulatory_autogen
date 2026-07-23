import pathlib
import warnings

import pytest

from parsers.PrimitiveParsers import (
    YamlFileParser,
    migrate_legacy_solver_info_keys,
    validate_solver_info,
    warn_if_casadi_nonzero_pre_time,
    PARAM_ID_METHODS,
    valid_param_id_methods,
    param_id_method_options,
    SOLVER_SCHEMA,
    SOLVER_INFO_FIELDS,
    solver_info_fields,
    gradient_sources,
    ANALYSIS_OPTIONS,
    analysis_options,
    _SOLVER_INTEGRATOR_KEYS,
    _CASADI_ADJOINT_METHODS,
)

# The descriptor shape shared by optimiser_options, solver_info fields, and analysis options.
_DESCRIPTOR_TYPES = {'int', 'float', 'bool', 'str', 'dict', 'enum'}


def _assert_descriptors_well_formed(context, options, valid_types=_DESCRIPTOR_TYPES):
    """Every option/field descriptor must carry the fields a settings UI relies on."""
    assert isinstance(options, list) and options, f'{context}: must be a non-empty list'
    seen = set()
    for opt in options:
        key = opt.get('name')
        assert key and key not in seen, f'{context}: missing/duplicate name {key!r}'
        seen.add(key)
        assert opt.get('type') in valid_types, f'{context}.{key}: bad type {opt.get("type")!r}'
        assert isinstance(opt.get('required'), bool), f'{context}.{key}: required must be bool'
        assert 'default' in opt, f'{context}.{key}: needs a default (None if none)'
        assert opt.get('description'), f'{context}.{key}: needs a description'
        if opt['type'] == 'enum':
            assert opt.get('default') in opt.get('choices', []), \
                f'{context}.{key}: enum default not in choices'
    return seen


def test_param_id_methods_schema_matches_dispatch():
    """PARAM_ID_METHODS is the discoverable list of calibration methods surfaced to downstream
    tools (e.g. the CUFLynx settings UI), so it must stay in sync with the param_id_method
    dispatch in OpencorParamID.run(). If a method is added/removed there, update this set."""
    assert set(PARAM_ID_METHODS.keys()) == {
        'genetic_algorithm', 'CMA-ES', 'bayesian', 'sp_minimize', 'multi_start_sp_minimize'
    }
    for name, meta in PARAM_ID_METHODS.items():
        assert meta.get('label') and meta.get('description')
        assert isinstance(meta.get('gradient_based'), bool)
    # aliases are surfaced by valid_param_id_methods (the dispatch accepts CMAES / cmaes for CMA-ES)
    assert set(valid_param_id_methods()) >= set(PARAM_ID_METHODS.keys()) | {'CMAES', 'cmaes'}


def test_param_id_method_options_are_well_formed():
    """Every method exposes its optimiser_options settings so a tool can auto-populate a settings
    form. Each option descriptor must carry the fields the UI relies on, with consistent types."""
    for name, meta in PARAM_ID_METHODS.items():
        _assert_descriptors_well_formed(name, meta.get('options'))
    # aliases resolve to the same options as their canonical method
    assert param_id_method_options('CMAES') == param_id_method_options('CMA-ES')
    assert param_id_method_options('not_a_method') == []


def test_param_id_method_options_match_optimiser_reads():
    """The advertised options must be the ones the optimiser classes actually read from
    optimiser_options -- otherwise a tool would offer settings that do nothing (or omit real
    ones). Guards against PARAM_ID_METHODS drifting from optimisers.py."""
    def names(method):
        return {opt['name'] for opt in param_id_method_options(method)}

    # Keys each optimiser reads from optimiser_options (see param_id/optimisers.py).
    assert names('genetic_algorithm') == {'num_calls_to_function', 'cost_convergence',
                                          'max_patience'}
    assert names('CMA-ES') == {'num_calls_to_function', 'sigma0', 'cost_convergence',
                               'max_patience'}
    assert names('bayesian') == {'num_calls_to_function'}
    assert names('sp_minimize') == {'cost_convergence'}
    assert names('multi_start_sp_minimize') == {
        'num_starts', 'start_sampling', 'include_init_point', 'seed', 'fd_step',
        'no_new_starts_on_convergence', 'convergence_cluster_tol_frac', 'cost_convergence'}
    # multi-start is a superset of sp_minimize's gradient-descent settings
    assert names('sp_minimize') <= names('multi_start_sp_minimize')


def test_solver_info_fields_schema_well_formed():
    """SOLVER_INFO_FIELDS lets a tool auto-populate the solver settings form; every solver that
    can be selected must have a well-formed field list, and it must be exposed on SOLVER_SCHEMA."""
    for solver, fields in SOLVER_INFO_FIELDS.items():
        _assert_descriptors_well_formed(solver, fields)
    # every solver offered by SOLVER_SCHEMA has a solver_info field list
    all_solvers = {s for solvers in SOLVER_SCHEMA['solvers_by_model_type'].values() for s in solvers}
    assert all_solvers <= set(SOLVER_INFO_FIELDS), \
        f'solvers without solver_info fields: {all_solvers - set(SOLVER_INFO_FIELDS)}'
    assert SOLVER_SCHEMA['solver_info_fields_by_solver'] is SOLVER_INFO_FIELDS
    assert solver_info_fields('CVODE_myokit') and solver_info_fields('not_a_solver') == []


def test_solver_integrator_keys_derived_from_schema():
    """_SOLVER_INTEGRATOR_KEYS (used by validate_solver_info) is derived from SOLVER_INFO_FIELDS,
    so the accepted keys and the advertised settings cannot drift. Locks the exact key sets that
    validation enforces today -- if a field is added to the schema, update these sets too."""
    for solver, fields in SOLVER_INFO_FIELDS.items():
        assert _SOLVER_INTEGRATOR_KEYS[solver] == {f['name'] for f in fields}
    cvode = {'MaximumStep', 'MaximumNumberOfSteps', 'rtol', 'atol'}
    assert _SOLVER_INTEGRATOR_KEYS['CVODE_opencor'] == cvode
    assert _SOLVER_INTEGRATOR_KEYS['CVODE_myokit'] == cvode
    assert _SOLVER_INTEGRATOR_KEYS['solve_ivp'] == {
        'rtol', 'atol', 'max_step', 'vectorized', 'dense_output', 'jac'}
    assert _SOLVER_INTEGRATOR_KEYS['casadi_integrator'] == {
        'reltol', 'abstol', 'rtol', 'atol', 'max_num_steps', 'max_step_size', 'max_step',
        'options'}
    assert _SOLVER_INTEGRATOR_KEYS['aadc_semi_implicit'] == {'tol', 'threads'}


def test_schema_settings_are_actually_read_by_the_code():
    """Every setting the schemas advertise must be read somewhere in src/.

    The schemas are CUFLynx's contract -- it builds its settings forms by reading them -- so a
    setting no code consumes becomes a control the user can change with no effect and no way to
    tell. The sibling tests above cannot catch that: _SOLVER_INTEGRATOR_KEYS is *derived from*
    SOLVER_INFO_FIELDS, so they compare the schema against a copy of itself. A phantom
    'gradient_method' on aadc_semi_implicit passed them for precisely that reason (AD vs FD is
    chosen by the do_ad flag, and the AD backend follows from model_type -- nothing ever read
    it). Check the schema against the source instead.

    The search is deliberately repo-wide rather than per-solver-file, because a setting is not
    always consumed by its own helper: CVODE_myokit's MaximumNumberOfSteps is read in
    protocol_runner.py, not myokit_helper.py. PrimitiveParsers.py is excluded because that is
    where the schema declares the names in the first place.
    """
    src_dir = pathlib.Path(__file__).resolve().parent.parent / 'src'
    corpus = '\n'.join(
        path.read_text(errors='ignore')
        for path in src_dir.rglob('*.py')
        if path.name != 'PrimitiveParsers.py' and 'obsolete' not in path.parts
    )

    def never_read(names):
        return [n for n in names if f'"{n}"' not in corpus and f"'{n}'" not in corpus]

    unread_solver = {
        solver: never_read([f['name'] for f in fields])
        for solver, fields in SOLVER_INFO_FIELDS.items()
    }
    unread_solver = {k: v for k, v in unread_solver.items() if v}
    assert not unread_solver, (
        f'solver_info settings advertised to CUFLynx but read nowhere in src/: {unread_solver}. '
        'Either wire the setting up or remove it from SOLVER_INFO_FIELDS.')

    unread_method = {
        method: never_read([o['name'] for o in spec.get('options', [])])
        for method, spec in PARAM_ID_METHODS.items()
    }
    unread_method = {k: v for k, v in unread_method.items() if v}
    assert not unread_method, (
        f'optimiser options advertised to CUFLynx but read nowhere in src/: {unread_method}. '
        'Either wire the option up or remove it from PARAM_ID_METHODS.')


def test_analysis_options_schema_well_formed():
    """The non-calibration analysis modes (sensitivity, MCMC, identifiability) expose their option
    blocks the same way, so a tool can auto-populate their settings forms too."""
    assert set(ANALYSIS_OPTIONS) == {'sensitivity_analysis', 'mcmc', 'identifiability_analysis'}
    for mode, meta in ANALYSIS_OPTIONS.items():
        assert meta.get('label') and meta.get('enable_flag') and meta.get('options_key')
        _assert_descriptors_well_formed(mode, meta.get('options'))
    # option names the analysis code actually reads (sensitivityAnalysis.py / paramID.py / IA)
    def names(mode):
        return {o['name'] for o in analysis_options(mode)}
    assert names('sensitivity_analysis') == {'method', 'sample_type', 'num_samples'}
    assert names('mcmc') == {'num_steps', 'num_walkers'}
    assert names('identifiability_analysis') == {'method', 'sub_method'}
    assert analysis_options('not_a_mode') == []
    # the enabling flags match the documented user_inputs feature flags
    assert {m['enable_flag'] for m in ANALYSIS_OPTIONS.values()} == {
        'do_sensitivity', 'do_mcmc', 'do_ia'}


def _option(mode, name):
    return next(o for o in analysis_options(mode) if o['name'] == name)


def test_closed_set_analysis_options_are_enums_with_choices():
    """Every option whose consumer dispatches on a fixed set of values must be declared
    'enum' with those values in 'choices' -- not a free 'str'.

    The schema is what front-ends build their settings forms from, so a free string
    becomes a text box for what is really a menu: the user can type something that
    only fails once the run is under way, and a GUI cannot offer a dropdown without
    hardcoding the list (which then drifts from CA).

    Choices are pinned to the dispatch sites, so adding a branch there without
    updating the schema fails here:
      * sample_type -> sobolSA._generate_samples (raises ValueError otherwise)
      * sub_method  -> utility_funcs.calculate_hessian
      * method      -> sensitivityAnalysis / identifiabilityAnalysis
    """
    expected = {
        ('sensitivity_analysis', 'method'): ['sobol', 'local'],
        ('sensitivity_analysis', 'sample_type'): ['saltelli', 'sobol'],
        ('identifiability_analysis', 'method'): ['Laplace', 'profile_likelihood'],
        # 'AD' is a branch in calculate_hessian but raises NotImplementedError, so it
        # is deliberately absent -- offering it would let a user pick a guaranteed crash.
        ('identifiability_analysis', 'sub_method'): ['parabola_fit', 'numdifftools_finite_diff'],
    }
    for (mode, name), choices in expected.items():
        opt = _option(mode, name)
        assert opt['type'] == 'enum', f'{mode}.{name} should be enum, got {opt["type"]!r}'
        assert opt['choices'] == choices, f'{mode}.{name} choices drifted from the dispatch'
        assert opt['default'] in opt['choices'], f'{mode}.{name} default not selectable'


def test_sample_type_choices_match_sobolsa_dispatch():
    """Guards the pairing directly: each declared sample_type must be a branch in
    sobol_SA.generate_samples, and an unknown one must still raise.

    Reads the source rather than importing sobolSA, which pulls in SALib/mpi4py and
    would make a pure schema test depend on the analysis stack being installed.
    """
    import re
    from pathlib import Path

    src_file = Path(__file__).resolve().parents[1] / 'src' / 'sensitivity_analysis' / 'sobolSA.py'
    src = src_file.read_text()
    body = src[src.index('def generate_samples'):]
    body = body[:body.index('\n    def ', 1)]  # just this method

    for choice in _option('sensitivity_analysis', 'sample_type')['choices']:
        assert re.search(rf'sample_type"?\'?\]?\s*==\s*[\'"]{choice}[\'"]', body), \
            f'sample_type {choice!r} is offered but generate_samples does not dispatch on it'
    assert 'raise ValueError' in body, 'generate_samples should still reject an unknown sample_type'


def test_cost_func_metadata_discovers_builtins():
    """The obs-data editor discovers valid cost_type values + flags at runtime (costs are a
    user-extensible registry, not a static schema)."""
    from funcs_user.cost_funcs_user import cost_func_metadata
    meta = cost_func_metadata()
    # built-in costs are all present
    assert {'gaussian_MLE', 'MSE', 'AE', 'multimodal_gaussian', 'additive', 'norm_additive'} \
        <= set(meta)
    for name, flags in meta.items():
        assert set(flags) == {'is_MLE', 'is_combiner', 'differentiable'}
        assert all(isinstance(v, bool) for v in flags.values())
    assert meta['gaussian_MLE']['is_MLE'] and meta['gaussian_MLE']['differentiable']
    assert meta['additive']['is_combiner']
    assert not meta['MSE']['is_MLE']


def test_cost_registry_excludes_organisational_accessors():
    """The organisational helpers in cost_funcs_user (register/build/get accessors and the
    cost_func_metadata accessor) must not be registered as selectable cost functions -- otherwise
    a bogus 'cost_func_metadata' cost shows up and even self-references in its own output (#259)."""
    from funcs_user.cost_funcs_user import get_cost_funcs_dict_for_mode, cost_func_metadata
    costs = get_cost_funcs_dict_for_mode("numpy")
    for accessor in ('cost_func_metadata', 'get_cost_funcs_dict_for_mode',
                     'build_cost_funcs_dict', 'register_cost_funcs'):
        assert accessor not in costs, f'{accessor} should not be a registered cost function'
    # the real costs are still there
    assert {'gaussian_MLE', 'MSE', 'AE'} <= set(costs)
    # and the metadata accessor no longer lists itself
    assert 'cost_func_metadata' not in cost_func_metadata()


def test_statically_defaulted_options_advertise_their_default():
    """If the code substitutes a fixed value when an option is absent, that value belongs in the
    schema's `default` -- a `None` default renders as a blank required field in front-ends even
    though CA supplies the value at run time (#277). Pins the options with a known static default;
    genuinely-required options (GA's num_calls_to_function, which raises if absent) stay None."""
    def opt(options, name):
        return next(o for o in options if o['name'] == name)

    num_samples = opt(analysis_options('sensitivity_analysis'), 'num_samples')
    assert num_samples['default'] == 32 and num_samples['required'] is False

    # bayesian falls back to a 10000-call budget; CMA-ES already advertised the same default.
    for method in ('bayesian', 'CMA-ES'):
        ncalls = opt(param_id_method_options(method), 'num_calls_to_function')
        assert ncalls['default'] == 10000 and ncalls['required'] is False

    # the genetic algorithm has no fallback (it raises if the key is missing), so None/required
    # is the correct, honest descriptor -- do not give it a phantom default.
    ga = opt(param_id_method_options('genetic_algorithm'), 'num_calls_to_function')
    assert ga['default'] is None and ga['required'] is True


def test_casadi_integrator_rejects_maximum_step_keys():
    with pytest.raises(ValueError, match="MaximumStep"):
        validate_solver_info('casadi_integrator', {
            'solver': 'casadi_integrator',
            'method': 'cvodes',
            'MaximumStep': 0.001,
        })

    with pytest.raises(ValueError, match="MaximumNumberOfSteps"):
        validate_solver_info('casadi_integrator', {
            'solver': 'casadi_integrator',
            'method': 'cvodes',
            'MaximumNumberOfSteps': 5000,
        })


def test_casadi_integrator_accepts_cvodes_options():
    validate_solver_info('casadi_integrator', {
        'solver': 'casadi_integrator',
        'method': 'cvodes',
        'max_step_size': 0.0001,
        'max_num_steps': 50000,
        'reltol': 1e-8,
        'abstol': 1e-10,
    })


def test_casadi_integrator_accepts_bdf_max_step():
    """The symbolic bdf method reads solver_info['max_step'] (internal sub-step cap), distinct
    from max_step_size. It must validate -- previously it was rejected as an unsupported key."""
    validate_solver_info('casadi_integrator', {
        'solver': 'casadi_integrator',
        'method': 'bdf',
        'max_step': 0.0005,
        'max_step_size': 0.001,
    })
    assert any(f['name'] == 'max_step' for f in solver_info_fields('casadi_integrator'))


def test_cellml_solver_accepts_maximum_step_keys():
    validate_solver_info('CVODE_myokit', {
        'solver': 'CVODE_myokit',
        'method': 'CVODE',
        'MaximumStep': 0.001,
        'MaximumNumberOfSteps': 5000,
    })


def test_cpp_rk4_accepts_maximum_number_of_steps():
    validate_solver_info('RK4', {
        'solver': 'RK4',
        'method': 'RK4',
        'MaximumStep': 0.001,
        'MaximumNumberOfSteps': 5000,
    })


def test_solve_ivp_rejects_maximum_step_keys():
    with pytest.raises(ValueError, match="MaximumStep"):
        validate_solver_info('solve_ivp', {
            'solver': 'solve_ivp',
            'method': 'BDF',
            'MaximumStep': 0.001,
        })


def test_migrate_legacy_solver_info_keys_for_solve_ivp():
    migrated = migrate_legacy_solver_info_keys('solve_ivp', {
        'MaximumStep': 0.0001,
        'MaximumNumberOfSteps': 5000,
        'method': 'BDF',
    })
    assert migrated == {'method': 'BDF', 'max_step': 0.0001}
    validate_solver_info('solve_ivp', {'solver': 'solve_ivp', **migrated})


def test_migrate_legacy_solver_info_keys_for_casadi_integrator():
    migrated = migrate_legacy_solver_info_keys('casadi_integrator', {
        'MaximumStep': 0.0001,
        'MaximumNumberOfSteps': 5000,
        'method': 'cvodes',
    })
    assert migrated == {
        'method': 'cvodes',
        'max_step_size': 0.0001,
        'max_num_steps': 5000,
    }
    validate_solver_info('casadi_integrator', {'solver': 'casadi_integrator', **migrated})


def test_parse_user_inputs_migrates_legacy_keys_for_python_model():
    parsed = YamlFileParser().parse_user_inputs_file({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'python',
        'solver': 'solve_ivp',
        'solver_info': {
            'MaximumStep': 0.0001,
            'MaximumNumberOfSteps': 5000,
            'method': 'BDF',
        },
        'dt': 0.01,
        'pre_time': 0.0,
        'sim_time': 1.0,
    }, obs_path_needed=False)
    assert parsed['solver_info']['max_step'] == 0.0001
    assert 'MaximumStep' not in parsed['solver_info']
    assert 'MaximumNumberOfSteps' not in parsed['solver_info']


def test_parse_user_inputs_warns_for_casadi_nonzero_pre_time():
    with pytest.warns(UserWarning, match='does not support nonzero pre_time'):
        YamlFileParser().parse_user_inputs_file({
            'file_prefix': '3compartment_nonstiff',
            'input_param_file': '3compartment_nonstiff_parameters.csv',
            'model_type': 'casadi_python',
            'solver': 'casadi_integrator',
            'solver_info': {'method': 'cvodes', 'max_num_steps': 50000},
            'dt': 0.01,
            'pre_time': 0.5,
            'sim_time': 0.3,
        }, obs_path_needed=False)


def test_warn_if_casadi_nonzero_pre_time_ignores_other_model_types():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warn_if_casadi_nonzero_pre_time('python', pre_time=0.5)
    assert len(caught) == 0


def test_sa_method_choices_each_have_a_dispatch_handler():
    """Every sensitivity_analysis 'method' choice in the schema must have a matching
    run_<method>_sensitivity handler on SensitivityAnalysis, and the run dispatcher must derive
    its valid set from the schema (not a hardcoded list). This locks the schema <-> dispatch
    correspondence the run message now relies on, so adding a method to the schema without a
    handler (or vice versa) fails here."""
    from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis, sa_method_choices

    choices = sa_method_choices()
    assert choices, "no sensitivity_analysis method choices found in the schema"
    # sa_method_choices reads straight from the schema accessor.
    assert choices == _option('sensitivity_analysis', 'method')['choices']
    for method in choices:
        assert hasattr(SensitivityAnalysis, f'run_{method}_sensitivity'), (
            f"schema advertises sa method {method!r} but SensitivityAnalysis has no "
            f"run_{method}_sensitivity handler")


def test_gradient_sources_well_formed_and_match_get_gradient_dispatch():
    """gradient_sources(model_type, solver) must advertise exactly the sources CA can actually
    produce, keyed to the top-level do_ad flag, so a front-end can build a gradient menu without
    hand-mirroring CA's rules. Pinned to OpencorParamID.get_gradient's dispatch: the AD-capable
    model types are AD_GRADIENT_MODEL_TYPES, and cellml_only+CVODE_myokit gets FSA.
    """
    from param_id.optimisers import AD_GRADIENT_MODEL_TYPES

    _REQUIRED_KEYS = {'value', 'label', 'do_ad', 'requires_all_differentiable', 'description'}

    def _check_shape(srcs):
        assert srcs, 'at least finite differences must always be offered'
        for s in srcs:
            assert set(s) == _REQUIRED_KEYS, s
            assert s['value'] in {'FD', 'AD', 'FSA'}
            assert isinstance(s['do_ad'], bool)
            assert isinstance(s['requires_all_differentiable'], bool)
        # Finite differences is always present, and is the only do_ad=False source.
        fd = [s for s in srcs if s['value'] == 'FD']
        assert len(fd) == 1 and fd[0]['do_ad'] is False
        assert [s for s in srcs if not s['do_ad']] == fd
        # Only CasADi AD needs all-differentiable.
        for s in srcs:
            assert s['requires_all_differentiable'] == (
                s['value'] == 'AD' and 'CasADi' in s['label'])

    # Every AD-capable model type offers an AD source with do_ad=True (matching get_gradient).
    for mt in AD_GRADIENT_MODEL_TYPES:
        srcs = gradient_sources(mt)
        _check_shape(srcs)
        ad = [s for s in srcs if s['value'] == 'AD']
        assert len(ad) == 1 and ad[0]['do_ad'] is True, mt

    # cellml_only gets the Myokit CVODES FSA source only with the Myokit solver.
    myokit = gradient_sources('cellml_only', 'CVODE_myokit')
    _check_shape(myokit)
    fsa = [s for s in myokit if s['value'] == 'FSA']
    assert len(fsa) == 1 and fsa[0]['do_ad'] is True

    # No analytic source for cellml_only under a non-FSA solver, or for non-AD model types.
    for mt, sv in [('cellml_only', 'CVODE_opencor'), ('python', 'solve_ivp'), ('cpp', 'RK4')]:
        srcs = gradient_sources(mt, sv)
        _check_shape(srcs)
        assert [s['value'] for s in srcs] == ['FD'], (mt, sv)


def test_per_integrator_ad_fsa_and_default_method_schema():
    """ad_suitable_methods / fsa_suitable_methods / default_method_by_solver let a front-end gate
    its Gradient menu on the selected integrator (issue #298). Lock their shape and, crucially,
    that ad_suitable_methods is derived from _CASADI_ADJOINT_METHODS (so the flag and the
    adjoint-warning cannot drift) and that the referenced methods actually exist in the schema."""
    ad = SOLVER_SCHEMA['ad_suitable_methods']
    fsa = SOLVER_SCHEMA['fsa_suitable_methods']
    default_method = SOLVER_SCHEMA['default_method_by_solver']

    casadi_methods = SOLVER_SCHEMA['methods_by_solver']['casadi_integrator']
    # AD-suitable = every casadi_integrator method that is NOT an adjoint-sensitivity method.
    assert ad['casadi_integrator'] == [m for m in casadi_methods
                                       if m not in _CASADI_ADJOINT_METHODS]
    assert ad['casadi_integrator'] == ['collocation', 'rk', 'semi_implicit_euler', 'bdf']
    # the adjoint methods are exactly the ones excluded
    assert set(casadi_methods) - set(ad['casadi_integrator']) == set(_CASADI_ADJOINT_METHODS)

    # fsa_suitable methods exist in the schema for their solver.
    for solver, methods in fsa.items():
        assert solver in SOLVER_SCHEMA['methods_by_solver'], solver
        assert set(methods) <= set(SOLVER_SCHEMA['methods_by_solver'][solver]), (solver, methods)
    assert fsa['CVODE_myokit'] == ['CVODE']

    # a default method must be one of that solver's methods, and AD-suitable where AD applies.
    for solver, m in default_method.items():
        assert m in SOLVER_SCHEMA['methods_by_solver'][solver], (solver, m)
    assert default_method['casadi_integrator'] == 'bdf'
    assert default_method['casadi_integrator'] in ad['casadi_integrator']


def test_gradient_sources_gates_analytic_source_on_integrator_method():
    """gradient_sources(..., method=...) drops the analytic source when the selected integrator
    cannot produce it -- CasADi AD for the adjoint methods (cvodes/idas), FSA for a non-CVODE
    method -- and keeps it otherwise. This is the per-integrator gate downstream tools apply."""
    # CasADi AD: offered for the symbolic/AD-suitable methods, gone for the adjoint ones.
    for m in SOLVER_SCHEMA['ad_suitable_methods']['casadi_integrator']:
        assert 'AD' in [s['value'] for s in gradient_sources('casadi_python', method=m)], m
    for m in _CASADI_ADJOINT_METHODS:
        assert [s['value'] for s in gradient_sources('casadi_python', method=m)] == ['FD'], m
    # method=None (or an unknown method) leaves AD offered.
    assert 'AD' in [s['value'] for s in gradient_sources('casadi_python')]
    assert 'AD' in [s['value'] for s in gradient_sources('casadi_python', method='not_a_method')]

    # Myokit FSA: offered for the CVODE method (the only, and FSA-suitable, method).
    assert 'FSA' in [s['value'] for s in
                     gradient_sources('cellml_only', 'CVODE_myokit', method='CVODE')]
    assert 'FSA' in [s['value'] for s in gradient_sources('cellml_only', 'CVODE_myokit')]

    # AADC AD has no per-integrator gate (its tape method is independent of this menu).
    assert 'AD' in [s['value'] for s in gradient_sources('aadc_python', method='semi_implicit')]
