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


class ProtocolRunner:
    """
    Standalone user-facing class for running CellML model protocols.

    Creates its own SimulationHelper (defaulting to myokit / CVODE) and
    wraps ProtocolExecutor for the simulation loop.

    Parameters
    ----------
    model_path : str
        Path to the CellML model file.
    inp_data_dict : dict, optional
        Configuration dict containing at least ``dt`` and ``solver_info``.
        If None, loaded from ``user_run_files/user_inputs.yaml`` (respecting
        ``user_inputs_path_override`` if set).
    solver : str, default 'CVODE_myokit'
        Solver identifier passed to get_simulation_helper.
        E.g. ``'CVODE_myokit'``, ``'CVODE_opencor'``.
    """

    def __init__(self, model_path, inp_data_dict=None, solver='CVODE_myokit'):
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

        self.dt = inp_data_dict['dt']
        solver_info = inp_data_dict.get('solver_info', {})
        self.MaximumStep = solver_info.get('MaximumStep', 0.001)
        self.MaximumNumberOfSteps = solver_info.get('MaximumNumberOfSteps', 5000)

        full_solver_info = {
            'MaximumNumberOfSteps': self.MaximumNumberOfSteps,
            'MaximumStep': self.MaximumStep,
        }
        # sim_time=1.0 is a placeholder — overridden per protocol_info in run_protocols
        self.sim_helper = get_simulation_helper(
            model_path=model_path,
            solver=solver,
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
        """
        Run the protocol defined by *protocol_info* and return result arrays.

        Parameters
        ----------
        model_path : str
            Path to the CellML model.  Kept for API compatibility; the model
            was loaded at construction time and this argument is not re-used.
        protocol_info : dict, optional
            If None, read from ``inp_data_dict['param_id_obs_path']``.
        id_param_names : list, optional
            Parameter names for identification (same format as param_id).
        id_param_vals : array-like, optional
            Values to apply before each experiment (e.g. calibrated parameters).

        Returns
        -------
        t_list : list[np.ndarray]
            Time vector per experiment with pre_time removed.
        res_list : list[list[np.ndarray]]
            Result arrays per experiment.  ``res_list[exp_idx][var_idx]``
            is the full time-series for that variable, concatenated across
            all sub-experiments.
        sim_times : list[list[float]]
            The sim_times from protocol_info.
        """
        if protocol_info is None:
            obs_path = self.inp_data_dict['param_id_obs_path']
            with open(obs_path, encoding='utf-8-sig') as fh:
                json_obj = json.load(fh)
            protocol_info = json_obj['protocol_info']

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
