"""
protocol_runner.py — standalone user-facing protocol runner.

ProtocolRunner is the independently callable class for running CellML model
protocols.  It handles model loading, solver setup, and result assembly,
delegating the core simulation loop to ProtocolExecutor.

Usage::

    runner = ProtocolRunner('/path/to/model.cellml', solver='CVODE_myokit')
    t_list, res_list, sim_times = runner.run_protocols(protocol_info=protocol_info)
"""

import json
import os
import sys

import numpy as np

from libcuflynx.utilities.paths import default_user_inputs_dir
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
from libcuflynx.protocol_runners.predictions import join_segments


def _solver_accepts(solver, key):
    """Whether ``solver``'s backend honours the solver_info ``key``.

    Imported lazily so this module keeps working if the parsers package is
    unavailable; an unknown solver is assumed to accept the key, which preserves
    the previous behaviour rather than silently dropping a setting.
    """
    try:
        from libcuflynx.parsers.PrimitiveParsers import solver_info_fields
    except Exception:
        return True
    fields = solver_info_fields(solver)
    return True if not fields else any(f['name'] == key for f in fields)


class ProtocolRunner:
    """Standalone, user-facing runner for model protocols.

    Creates its own [`SimulationHelper`][solver_wrappers.python_solver_helper.SimulationHelper]
    (defaulting to Myokit / CVODE) and wraps ``ProtocolExecutor`` for the
    multi-experiment / sub-experiment simulation loop. Useful for re-simulating
    a model with calibrated parameters outside the param-id loop.

    Args:
        model_path: Path to the model file.
        inp_data_dict: Configuration dict containing at least ``dt`` and
            ``solver_info``. If None, loaded from
            ``user_run_files/user_inputs.yaml`` (respecting
            ``user_inputs_path_override`` if set).
        solver: Solver identifier passed to
            [`get_simulation_helper`][solver_wrappers.get_simulation_helper],
            e.g. ``'CVODE_myokit'`` or ``'CVODE_opencor'``.
        model_type: Backend family (``'cellml'`` / ``'python'`` /
            ``'casadi_python'``); falls back to ``inp_data_dict['model_type']``.
    """

    def __init__(self, model_path, inp_data_dict=None, solver='CVODE_myokit', model_type=None):
        if inp_data_dict is None:
            import yaml
            with open(os.path.join(default_user_inputs_dir(), 'user_inputs.yaml'), 'r') as fh:
                inp_data_dict = yaml.load(fh, Loader=yaml.FullLoader)
            override = inp_data_dict.get('user_inputs_path_override')
            if override:
                if os.path.exists(override):
                    with open(override, 'r') as fh:
                        inp_data_dict = yaml.load(fh, Loader=yaml.FullLoader)
                else:
                    print(
                        f"User inputs file not found at {override}\n"
                        "Check user_inputs_path_override in user_inputs.yaml "
                        "and set it to False to use the default location."
                    )
                    raise FileNotFoundError(override)

        self.inp_data_dict = inp_data_dict
        self.solver = solver
        # model_type selects the backend family (cellml / python / casadi_python).
        # Falls back to inp_data_dict['model_type'] when not passed explicitly.
        self.model_type = model_type if model_type is not None else inp_data_dict.get('model_type')

        self.dt = inp_data_dict['dt']
        solver_info = inp_data_dict.get('solver_info', {})
        self.MaximumStep = solver_info.get('MaximumStep', 0.001)
        self.MaximumNumberOfSteps = solver_info.get('MaximumNumberOfSteps', 5000)

        # Preserve all solver_info keys (e.g. rtol/atol/method) and only fill in
        # MaximumStep / MaximumNumberOfSteps defaults — previously the extra keys
        # were silently dropped, so tight tolerances passed by callers were ignored.
        #
        # The MaximumNumberOfSteps default is conditional: this relays solver_info
        # to the backend rather than consuming it, so seeding it unconditionally
        # put the key back for solvers whose helper ignores it (CVODE_myokit) --
        # undoing migrate_legacy_solver_info_keys and re-creating a setting that
        # looks live but does nothing.
        full_solver_info = dict(solver_info)
        if _solver_accepts(solver, 'MaximumNumberOfSteps'):
            full_solver_info.setdefault('MaximumNumberOfSteps', self.MaximumNumberOfSteps)
        full_solver_info.setdefault('MaximumStep', self.MaximumStep)
        # sim_time=1.0 is a placeholder — overridden per protocol_info in run_protocols
        self.sim_helper = get_simulation_helper(
            model_path=model_path,
            solver=solver,
            model_type=self.model_type,
            dt=self.dt,
            sim_time=1.0,
            solver_info=full_solver_info,
            pre_time=0.0,
        )
        self._executor = ProtocolExecutor(self.sim_helper)
        self.variable_names = self.sim_helper.get_all_variable_names()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_variable_names(self):
        """Return the variable names of the model (used for downstream plotting)."""
        return self.variable_names

    def get_var2idx_dict(self):
        """Return a mapping from variable name to index in the result list."""
        return {name: idx for idx, name in enumerate(self.variable_names)}

    def _apply_protocol_info(self, protocol_info):
        """Hand ``protocol_info`` to the helper before running it.

        A params_to_change entry naming a trace (or a shape) needs the helper to know the
        protocol. Guarded on identity because for the Myokit backend this binds the traced
        inputs and recreates the simulation, which is not something to redo on every call.
        The result rows are ordered by the rebuilt model's variables, so ``variable_names`` is
        re-read; left stale, every variable silently reads as its neighbour.
        """
        if hasattr(self.sim_helper, 'set_protocol_info'):
            if getattr(self, '_applied_protocol_info', None) is not protocol_info:
                self.sim_helper.set_protocol_info(protocol_info)
                self._applied_protocol_info = protocol_info
                self.variable_names = self.sim_helper.get_all_variable_names()

    def run_protocols(self, model_path=None, protocol_info=None,
                      id_param_names=None, id_param_vals=None,
                      result_variables=None, return_segments=False):
        """Run the protocol defined by ``protocol_info`` and return result arrays.

        Args:
            model_path: Not used (the model was loaded at construction time); kept so
                existing positional calls keep working.
            protocol_info: Protocol descriptor. If None, read from
                ``inp_data_dict['param_id_obs_path']``.
            id_param_names: Parameter names to set (same format as param_id).
            id_param_vals: Values to apply before each experiment (e.g.
                calibrated parameters).
            result_variables: The variables to record, as for ``get_results``; None records
                every variable, in ``get_variable_names()`` order.
            return_segments: Also return the per-sub-experiment results.

        Returns:
            tuple: ``(t_list, res_list, sim_times)`` where ``t_list`` is the time
            vector per experiment (pre_time removed); ``res_list[exp][var]`` is
            the full time-series for that variable joined across sub-experiments
            (``join_segments``); and ``sim_times`` is the ``sim_times`` from
            ``protocol_info``. With ``return_segments``, a fourth element
            ``{(exp, sub): results}`` holds each sub-experiment's own results.
        """
        if protocol_info is None:
            obs_path = self.inp_data_dict['param_id_obs_path']
            with open(obs_path, encoding='utf-8-sig') as fh:
                json_obj = json.load(fh)
            protocol_info = json_obj['protocol_info']

        self._apply_protocol_info(protocol_info)

        sim_times = protocol_info['sim_times']
        pre_times = protocol_info['pre_times']
        num_experiments = len(sim_times)

        if num_experiments != len(pre_times):
            raise ValueError('pre_times and sim_times must have the same length')

        num_sub_per_exp = protocol_info.get(
            'num_sub_per_exp',
            [len(sim_times[i]) for i in range(num_experiments)],
        )

        print(f'Running experiments — dt={self.dt}, MaximumStep={self.MaximumStep}')

        success, results_by_sub, _, t_by_exp = self._executor.run_protocol(
            protocol_info,
            result_variables=result_variables,
            id_param_names=id_param_names,
            id_param_vals=id_param_vals,
        )
        if not success:
            raise RuntimeError('Protocol simulation failed.')

        t_list = []
        res_list = []
        for exp_idx in range(num_experiments):
            t_list.append(t_by_exp[exp_idx])
            subs = range(num_sub_per_exp[exp_idx])
            results = [results_by_sub.get((exp_idx, sub_idx)) for sub_idx in subs]
            if result_variables is not None:
                # get_results gives one list per requested name list; flattened in order
                results = [None if r is None else [v for entry in r for v in entry]
                           for r in results]
            res_list.append(join_segments(results, sim_times[exp_idx], self.dt))
            print(
                f'Experiment {exp_idx} completed. '
                f'Remaining: {num_experiments - exp_idx - 1}'
            )

        if return_segments:
            return t_list, res_list, sim_times, results_by_sub
        return t_list, res_list, sim_times
