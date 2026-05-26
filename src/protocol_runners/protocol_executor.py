"""
protocol_executor.py — shared core protocol simulation loop.

ProtocolExecutor owns the multi-experiment / multi-subexperiment loop that is
common to ProtocolRunner, OpencorParamID, and sobol_SA.  It accepts any
SimulationHelper (myokit, opencor, python) that has already been constructed by
the caller.

For standalone use (without a pre-built SimulationHelper) use ProtocolRunner
from protocol_runners.protocol_runner instead.
"""

import numpy as np


class ProtocolExecutor:
    """
    Core multi-experiment / multi-subexperiment protocol simulation loop.

    Parameters
    ----------
    sim_helper : SimulationHelper
        Any solver-wrapper instance (myokit, opencor, python, casadi).
        The caller retains ownership; ProtocolExecutor does not close it.
    """

    def __init__(self, sim_helper):
        self.sim_helper = sim_helper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_protocol(self, protocol_info,
                     id_param_names=None, id_param_vals=None,
                     result_variables=None,
                     extra_result_variables=None,
                     exp_indices=None,
                     continue_on_failure=False,
                     reset_after_experiment=True):
        """
        Run the multi-experiment / multi-subexperiment protocol loop.

        Parameters
        ----------
        protocol_info : dict
            Must contain:
              - 'sim_times'       list[list[float]]  — sim duration per (exp, sub)
              - 'pre_times'       list[float]        — pre-simulation time per exp
              - 'params_to_change' dict              — {param_name: [[val_exp0_sub0, ...], ...]}
            May also contain pre-computed keys added by process_protocol_and_weights:
              - 'num_experiments'  int
              - 'num_sub_per_exp'  list[int]
            Both are derived from sim_times when absent.
        id_param_names : list, optional
            Parameter names set once before each experiment (e.g. ID candidates).
        id_param_vals : list, optional
            Values matching id_param_names.
        result_variables : list, optional
            Variables to retrieve per sub-experiment via get_results().
            ``None`` → collect all variables via get_all_results(flatten=True).
        extra_result_variables : list, optional
            A second variable set collected in the same pass (e.g. pred_names).
        exp_indices : iterable, optional
            Only run these experiment indices.  Other slots in the output
            will be absent from results_by_sub / extra_by_sub, and None in
            t_by_exp.
        continue_on_failure : bool, default False
            If True, record None for failed sub-experiments and continue
            rather than returning early.  If False (default), return
            immediately on the first simulation failure.
        reset_after_experiment : bool, default True
            If True (default), call sim_helper.reset_and_clear() after the
            final sub-experiment of each experiment.  Pass False for AD
            (automatic differentiation) mode where the solver state must be
            preserved across experiments.

        Returns
        -------
        success : bool
            False only when a simulation failed AND continue_on_failure is False.
        results_by_sub : dict
            Mapping ``(exp_idx, sub_idx)`` → result of get_results() or
            get_all_results(flatten=True).  None for failed sub-experiments
            when continue_on_failure is True.
        extra_by_sub : dict
            Same structure for extra_result_variables.  Empty dict when
            extra_result_variables is None.
        t_by_exp : list[np.ndarray | None]
            Concatenated, pre_time-shifted time vector per experiment index.
            None for skipped or failed experiments.
        """
        sim_times = protocol_info['sim_times']
        pre_times = protocol_info['pre_times']
        params_to_change = protocol_info.get('params_to_change', {})

        num_experiments = protocol_info.get('num_experiments', len(sim_times))
        num_sub_per_exp = protocol_info.get(
            'num_sub_per_exp',
            [len(sim_times[i]) for i in range(num_experiments)],
        )

        run_set = (
            set(range(num_experiments))
            if exp_indices is None
            else set(exp_indices)
        )

        results_by_sub = {}
        extra_by_sub = {}
        t_by_exp = [None] * num_experiments

        for exp_idx in range(num_experiments):
            if exp_idx not in run_set:
                continue

            if id_param_names is not None and id_param_vals is not None:
                self.sim_helper.set_param_vals(id_param_names, id_param_vals)
            self.sim_helper.reset_states()

            current_time = 0
            t_vec = None

            for sub_idx in range(num_sub_per_exp[exp_idx]):
                sim_time = sim_times[exp_idx][sub_idx]

                if sub_idx == 0:
                    self.sim_helper.update_times(
                        self.sim_helper.dt, 0.0, sim_time, pre_times[exp_idx]
                    )
                    current_time += pre_times[exp_idx]
                else:
                    self.sim_helper.update_times(
                        self.sim_helper.dt, current_time, sim_time, 0.0
                    )

                if params_to_change:
                    self.sim_helper.set_param_vals(
                        list(params_to_change.keys()),
                        [params_to_change[k][exp_idx][sub_idx]
                         for k in params_to_change],
                    )

                success = self.sim_helper.run()
                is_last_sub = (sub_idx == num_sub_per_exp[exp_idx] - 1)

                if success:
                    current_time += sim_time

                    if result_variables is None:
                        results_by_sub[(exp_idx, sub_idx)] = (
                            self.sim_helper.get_all_results(flatten=True)
                        )
                    else:
                        results_by_sub[(exp_idx, sub_idx)] = (
                            self.sim_helper.get_results(result_variables)
                        )

                    if extra_result_variables is not None:
                        extra_by_sub[(exp_idx, sub_idx)] = (
                            self.sim_helper.get_results(extra_result_variables)
                        )

                    # Accumulate time vector (concatenate sub-experiment segments)
                    t_seg = np.array(self.sim_helper.tSim)
                    if t_vec is None:
                        t_vec = t_seg
                    else:
                        t_vec = np.concatenate([t_vec, t_seg[1:]])

                    if is_last_sub and reset_after_experiment:
                        self.sim_helper.reset_and_clear()

                else:
                    if is_last_sub and reset_after_experiment:
                        self.sim_helper.reset_and_clear()

                    if continue_on_failure:
                        results_by_sub[(exp_idx, sub_idx)] = None
                        if extra_result_variables is not None:
                            extra_by_sub[(exp_idx, sub_idx)] = None
                    else:
                        return False, results_by_sub, extra_by_sub, t_by_exp

            if t_vec is not None:
                t_by_exp[exp_idx] = t_vec - pre_times[exp_idx]

        return True, results_by_sub, extra_by_sub, t_by_exp
