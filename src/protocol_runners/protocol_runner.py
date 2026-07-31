"""
protocol_runner.py — standalone user-facing protocol runner.

ProtocolRunner is the independently callable class for running CellML model
protocols.  It handles model loading, solver setup, and result assembly,
delegating the core simulation loop to ProtocolExecutor.

Usage::

    runner = ProtocolRunner('/path/to/model.cellml', solver='CVODE_myokit')
    t_list, res_list, sim_times = runner.run_protocols(model_path, protocol_info)
"""

import json
import os
import sys

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(__file__), '..')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_ROOT_DIR = os.path.join(os.path.dirname(__file__), '../..')
_USER_INPUTS_DIR = os.path.join(_ROOT_DIR, 'user_run_files')

from solver_wrappers import get_simulation_helper
from protocol_runners.protocol_executor import ProtocolExecutor


def _solver_accepts(solver, key):
    """Whether ``solver``'s backend honours the solver_info ``key``.

    Imported lazily so this module keeps working if the parsers package is
    unavailable; an unknown solver is assumed to accept the key, which preserves
    the previous behaviour rather than silently dropping a setting.
    """
    try:
        from parsers.PrimitiveParsers import solver_info_fields
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
        model_type: Backend family (``'cellml_only'`` / ``'python'`` /
            ``'casadi_python'``); falls back to ``inp_data_dict['model_type']``.
    """

    def __init__(self, model_path, inp_data_dict=None, solver='CVODE_myokit', model_type=None):
        if inp_data_dict is None:
            import yaml
            with open(os.path.join(_USER_INPUTS_DIR, 'user_inputs.yaml'), 'r') as fh:
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
        # model_type selects the backend family (cellml_only / python / casadi_python).
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

    def run_protocols(self, model_path, protocol_info=None,
                      id_param_names=None, id_param_vals=None):
        """Run the protocol defined by ``protocol_info`` and return result arrays.

        Args:
            model_path: Path to the model. Kept for API compatibility; the model
                was loaded at construction time and this argument is not re-used.
            protocol_info: Protocol descriptor. If None, read from
                ``inp_data_dict['param_id_obs_path']``.
            id_param_names: Parameter names to set (same format as param_id).
            id_param_vals: Values to apply before each experiment (e.g.
                calibrated parameters).

        Returns:
            tuple: ``(t_list, res_list, sim_times)`` where ``t_list`` is the time
            vector per experiment (pre_time removed); ``res_list[exp][var]`` is
            the full time-series for that variable concatenated across
            sub-experiments; and ``sim_times`` is the ``sim_times`` from
            ``protocol_info``.
        """
        if protocol_info is None:
            obs_path = self.inp_data_dict['param_id_obs_path']
            with open(obs_path, encoding='utf-8-sig') as fh:
                json_obj = json.load(fh)
            protocol_info = json_obj['protocol_info']

        # Hand the protocol to the helper before running it. Without this a
        # params_to_change entry naming a trace (or a shape) failed here with
        # "protocol_traces not found in protocol_info" -- the helper had never
        # been told the protocol it was executing. Guarded on identity because
        # for the Myokit backend this rebinds `pace` and recreates the
        # simulation, which is not something to redo on every call.
        if hasattr(self.sim_helper, 'set_protocol_info'):
            if getattr(self, '_applied_protocol_info', None) is not protocol_info:
                self.sim_helper.set_protocol_info(protocol_info)
                self._applied_protocol_info = protocol_info
                # Binding `pace` rebuilds the simulation, and the result rows are
                # ordered by the rebuilt model's variables -- so the map captured
                # in __init__ no longer indexes what get_results returns. Left
                # stale, every variable silently reads as its neighbour.
                self.variable_names = self.sim_helper.get_all_variable_names()

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
            result_variables=None,
            id_param_names=id_param_names,
            id_param_vals=id_param_vals,
        )
        if not success:
            raise RuntimeError('Protocol simulation failed.')

        t_list = []
        res_list = []
        for exp_idx in range(num_experiments):
            t_list.append(t_by_exp[exp_idx])

            res_vec = None
            for sub_idx in range(num_sub_per_exp[exp_idx]):
                sub_res = results_by_sub.get((exp_idx, sub_idx))
                if sub_res is None:
                    continue

                if sub_idx == 0:
                    res_vec = list(sub_res)
                    for var_idx in range(len(res_vec)):
                        if not hasattr(res_vec[var_idx], '__len__'):
                            n = len(t_by_exp[exp_idx]) if t_by_exp[exp_idx] is not None else 1
                            res_vec[var_idx] = np.ones(n) * res_vec[var_idx]
                else:
                    for var_idx in range(len(res_vec)):
                        new_data = sub_res[var_idx]
                        if not hasattr(new_data, '__len__'):
                            n_sub = round(sim_times[exp_idx][sub_idx] / self.dt)
                            new_data = np.ones(n_sub) * new_data
                        else:
                            new_data = new_data[1:]
                        res_vec[var_idx] = np.concatenate([res_vec[var_idx], new_data])

            res_list.append(res_vec)
            print(
                f'Experiment {exp_idx} completed. '
                f'Remaining: {num_experiments - exp_idx - 1}'
            )

        return t_list, res_list, sim_times
