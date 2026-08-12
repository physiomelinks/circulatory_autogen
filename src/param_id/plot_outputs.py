"""
Post-calibration plotting for CVS0D parameter identification.

Each figure type produced by ``plot_outputs`` lives in its own method on
:class:`ParamIDPlotOutputs`.
"""

from __future__ import annotations

import os
import re
from sys import exit
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class ParamIDPlotOutputs:
    """
    Build and save post-calibration figures for parameter identification.

    ``plot_outputs`` runs, in order: best-fit reconstruction pages
    (:meth:`plot_reconstruction_pages`), vector saves (:meth:`save_error_vectors`),
    percent / std error bar pages (:meth:`plot_percent_error_bar_pages`,
    :meth:`plot_std_error_bar_pages`),     protocol-parameter time courses
    (:meth:`plot_protocol_params_to_change`: one figure per ``params_to_change`` key,
    subplots wrapped **3 per row**), and console error summary
    (:meth:`print_observable_errors`). Orphan reconstruction figures are closed
    with :meth:`finalize_reconstruction_if_unsaved`.

    Parameters
    ----------
    client :
        :class:`CVS0DParamID` instance holding ``obs_info``, ``protocol_info``,
        ``param_id``, ``gt_df``, paths, ``dt``, and ``rank``.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def plot_outputs(self) -> None:
        print("plotting best observables")
        if getattr(self.client.param_id, "emulates_features", False):
            self.save_emulator_outputs()
            return
        phase = self._uses_phase()
        list_of_obs_dicts, list_of_all_series = self._fetch_best_fit_data()
        tSim_per_sub_count, sim_time_tot_per_exp, n_steps_per_sub_count = (
            self._compute_subexperiment_time_axes()
        )
        percent_error_vec, std_error_vec, phase_error_vec = (
            self.plot_reconstruction_pages(
                phase,
                list_of_obs_dicts,
                list_of_all_series,
                tSim_per_sub_count,
                sim_time_tot_per_exp,
                n_steps_per_sub_count,
            )
        )
        self.save_error_vectors(percent_error_vec, std_error_vec)
        obs_names_for_plot = self._observable_names_for_error_plots()
        self.plot_percent_error_bar_pages(obs_names_for_plot, percent_error_vec)
        self.plot_std_error_bar_pages(obs_names_for_plot, std_error_vec)
        self.plot_protocol_params_to_change()
        self.print_observable_errors(
            phase, percent_error_vec, phase_error_vec
        )

    def save_emulator_outputs(self) -> None:
        """Error vectors and error-bar pages for a run that used an emulator.

        An emulator predicts the scalar features and not the traces, so the
        reconstruction pages cannot be drawn -- but the *errors* are a comparison
        of feature against ground truth, and those are exactly what it has. Losing
        them along with the traces would leave a finished calibration with nothing
        to show for itself, which is what happened before this existed (#333).

        Writes the same ``percent_error_vec`` / ``error_vec_names`` files an
        ordinary run writes, so every consumer of them -- CA's own plots, and the
        tools that read the outputs directory -- is unchanged.
        """
        percent_error_vec, std_error_vec = self.emulator_error_vectors()
        self.save_error_vectors(percent_error_vec, std_error_vec)
        obs_names_for_plot = self._observable_names_for_error_plots()
        self.plot_percent_error_bar_pages(obs_names_for_plot, percent_error_vec)
        self.plot_std_error_bar_pages(obs_names_for_plot, std_error_vec)
        print(
            "This run used an emulator, so the reconstruction plots (which need the "
            "simulated traces) were not drawn; the observable errors above are the "
            "emulator's own features against the ground truth."
        )

    def emulator_error_vectors(self):
        """Percent and std error per observable, from the emulator's features.

        Only ``constant`` observables exist on this path -- CA refuses an emulator
        study whose obs_data holds anything else -- so this is the constant branch
        of ``plot_reconstruction_pages`` with the plotting removed, and it must
        stay identical to it: the two numbers are the same quantity, and a run
        should not report a different error because it drew fewer pictures.
        """
        obs_info = self.client.obs_info
        param_id = self.client.param_id
        num_obs = obs_info["num_obs"]
        percent_error_vec = np.zeros((num_obs,))
        std_error_vec = np.zeros((num_obs,))

        _, operands_list = param_id.get_cost_and_obs_from_params(param_id.best_param_vals)
        if not operands_list:
            print("WARNING: the emulator produced no observables; errors not saved")
            return percent_error_vec, std_error_vec

        # One evaluation per segment, then each observable read from its own --
        # a data_item names the experiment and sub-experiment it belongs to, and
        # scoring it against another segment is scoring the wrong thing.
        num_sub_per_exp = self.client.protocol_info["num_sub_per_exp"]
        const_by_segment = {}
        for const_idx, obs_idx in enumerate(obs_info["const_idx_to_obs_idx"]):
            exp = int(obs_info["experiment_idxs"][obs_idx])
            sub = int(obs_info["subexperiment_idxs"][obs_idx])
            flat = sum(num_sub_per_exp[:exp]) + sub
            if flat >= len(operands_list) or operands_list[flat] is None:
                continue
            if flat not in const_by_segment:
                const_by_segment[flat] = np.asarray(
                    param_id.get_obs_output_dict(operands_list[flat])["const"], dtype=float
                )
            consts = const_by_segment[flat]
            if const_idx >= len(consts):
                continue
            ground_truth = obs_info["ground_truth_const"][const_idx]
            percent_error_vec[obs_idx] = (
                100 * (consts[const_idx] - ground_truth) / (ground_truth + 1e-10)
            )
            std_error_vec[obs_idx] = (
                consts[const_idx] - ground_truth
            ) / obs_info["std_const_vec"][const_idx]
        return percent_error_vec, std_error_vec

    def _uses_phase(self) -> bool:
        gtp = self.client.obs_info["ground_truth_phase"]
        if len(gtp) == 0:
            return False
        if gtp.all() == None:
            return False
        return True

    def _fetch_best_fit_data(self):
        obs_info = self.client.obs_info
        param_id = self.client.param_id
        model_type = self.client.model_type
        best = param_id.best_param_vals

        if model_type == "casadi_python":
            _cost = param_id.get_cost_ca(best)
            return param_id.get_obs_ca(best, get_all_series=True)

        _, best_fit_operands_list = param_id.get_cost_and_obs_from_params(best)
        list_of_obs_dicts = []
        list_of_all_series = []
        for obs in best_fit_operands_list:
            obs_dict, all_series = param_id.get_obs_output_dict(
                obs, get_all_series=True
            )
            list_of_obs_dicts.append(obs_dict)
            list_of_all_series.append(all_series)
        return list_of_obs_dicts, list_of_all_series

    def _compute_subexperiment_time_axes(self):
        """Build per-subexperiment time grids used for reconstruction overlays."""
        protocol_info = self.client.protocol_info
        dt = self.client.dt

        subexp_count = -1
        tSim_per_sub_count = []
        sim_time_tot_per_exp = []
        n_steps_per_sub_count = []

        for exp_idx in range(protocol_info["num_experiments"]):
            subexp_count += 1
            sim_time_tot_per_exp.append(np.sum(protocol_info["sim_times"][exp_idx]))
            n_steps_tot = int(sim_time_tot_per_exp[exp_idx] / dt)
            n_steps_per_sub_count = n_steps_per_sub_count + [
                int(protocol_info["sim_times"][exp_idx][II] / dt)
                for II in range(protocol_info["num_sub_per_exp"][exp_idx])
            ]
            np.linspace(0.0, np.sum(protocol_info["sim_times"][exp_idx]), n_steps_tot + 1)
            tSim_per_sub_count.append(
                np.linspace(
                    0.0,
                    protocol_info["sim_times"][exp_idx][0],
                    n_steps_per_sub_count[subexp_count] + 1,
                )
            )
            start_time_sum = protocol_info["sim_times"][exp_idx][0]

            for II in range(1, protocol_info["num_sub_per_exp"][exp_idx]):
                subexp_count += 1
                tSim_per_sub_count.append(
                    np.linspace(
                        start_time_sum,
                        start_time_sum + protocol_info["sim_times"][exp_idx][II],
                        n_steps_per_sub_count[subexp_count] + 1,
                    )
                )
                start_time_sum += protocol_info["sim_times"][exp_idx][II]

        return tSim_per_sub_count, sim_time_tot_per_exp, n_steps_per_sub_count

    def plot_reconstruction_pages(
        self,
        phase,
        list_of_obs_dicts,
        list_of_all_series,
        tSim_per_sub_count,
        sim_time_tot_per_exp,
        n_steps_per_sub_count,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce ``reconstruct_*`` and optional ``phase_reconstruct_*`` figures;
        accumulate percent / std / phase error vectors.

        Each unique (observable, experiment) corresponds to one saved reconstruction file.
        """
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        no_conv = 1.0

        obs_info = self.client.obs_info
        protocol_info = self.client.protocol_info
        plot_dir = self.client.plot_dir
        prefix = self.client.file_name_prefix
        obs_stub = self.client.param_id_obs_file_prefix

        obs_tuples_unique = []
        for idx, obs_name in enumerate(obs_info["obs_names"]):
            tup = (obs_name, obs_info["experiment_idxs"][idx])
            if tup not in obs_tuples_unique:
                obs_tuples_unique.append(tup)

        percent_error_vec = np.zeros((obs_info["num_obs"],))
        phase_error_vec = np.zeros((obs_info["num_obs"],))
        std_error_vec = np.zeros((obs_info["num_obs"],))

        plot_idx = 0
        fig, axs = plt.subplots(squeeze=False)
        axs = axs[0, 0]
        if phase:
            fig_phase, axs_phase = plt.subplots(squeeze=False)
            axs_phase = axs_phase[0, 0]
        else:
            fig_phase = axs_phase = None

        plot_saved = False

        for unique_obs_count in range(len(obs_tuples_unique)):
            this_obs_waveform_plotted = False
            const_idx = -1
            series_idx = -1
            freq_idx = -1
            prob_dist_idx = -1

            for II in range(obs_info["num_obs"]):
                if obs_info["data_types"][II] == "constant":
                    const_idx += 1
                elif obs_info["data_types"][II] == "series":
                    series_idx += 1
                elif obs_info["data_types"][II] == "frequency":
                    freq_idx += 1
                elif obs_info["data_types"][II] == "prob_dist":
                    prob_dist_idx += 1

                if (
                    obs_info["obs_names"][II],
                    obs_info["experiment_idxs"][II],
                ) != obs_tuples_unique[unique_obs_count]:
                    continue

                exp_idx = obs_info["experiment_idxs"][II]
                this_sub_idx = obs_info["subexperiment_idxs"][II]
                subexp_count = int(
                    np.sum(protocol_info["num_sub_per_exp"][:exp_idx]) + this_sub_idx
                )

                series_per_sub = list_of_all_series[subexp_count]

                best_fit_obs_const = list_of_obs_dicts[subexp_count]["const"]
                best_fit_obs_series = list_of_obs_dicts[subexp_count]["series"]
                best_fit_obs_amp = list_of_obs_dicts[subexp_count]["amp"]
                best_fit_obs_phase = list_of_obs_dicts[subexp_count]["phase"]
                best_fit_obs_prob_dist = list_of_obs_dicts[subexp_count][
                    "val_for_prob_dist"
                ]

                if len(obs_info["ground_truth_series"]) > 0:
                    if obs_info["obs_dt"][series_idx] == self.client.dt:
                        min(
                            obs_info["ground_truth_series"][series_idx].shape[0],
                            len(best_fit_obs_series[0]),
                        )

                obs_name_for_plot = obs_info["names_for_plotting"][II]
                if obs_name_for_plot.count("_") > 1:
                    print(
                        f'obs_data variable "{obs_name_for_plot}" has too many underscores',
                        'for plotting a label. Include a "name_for_plotting" key in the ',
                        "obs_data json file entry",
                    )
                    exit()

                unit = obs_info["units"][II]
                if unit == "m3_per_s":
                    conversion = m3_to_cm3
                    unit_label = "[cm^3/s]"
                elif unit == "m_per_s":
                    conversion = no_conv
                    unit_label = "[m/s]"
                elif unit == "m3":
                    conversion = m3_to_cm3
                    unit_label = "[cm^3]"
                elif unit == "J_per_m3":
                    conversion = Pa_to_kPa
                    unit_label = "[kPa]"
                else:
                    conversion = 1.0
                    unit_label = f"[{unit}]"

                if obs_info["data_types"][II] == "series":
                    axs.set_ylabel(f"${obs_name_for_plot}$ ${unit_label}$", fontsize=18)

                if not this_obs_waveform_plotted:
                    axs.set_ylabel(f"${obs_name_for_plot}$ ${unit_label}$", fontsize=18)
                    if obs_info["data_types"][II] != "frequency":
                        for temp_sub_idx in range(
                            protocol_info["num_sub_per_exp"][exp_idx]
                        ):
                            temp_subexp_count = int(
                                np.sum(protocol_info["num_sub_per_exp"][:exp_idx])
                                + temp_sub_idx
                            )
                            temp_series_per_sub = list_of_all_series[temp_subexp_count]
                            if temp_sub_idx == 0:
                                axs.plot(
                                    tSim_per_sub_count[temp_subexp_count],
                                    conversion * temp_series_per_sub[II][:],
                                    color=protocol_info["experiment_colors"][exp_idx],
                                    label="output",
                                )
                            else:
                                axs.plot(
                                    tSim_per_sub_count[temp_subexp_count],
                                    conversion * temp_series_per_sub[II][:],
                                    color=protocol_info["experiment_colors"][exp_idx],
                                )
                        axs.set_xlim(0.0, sim_time_tot_per_exp[exp_idx])
                        axs.set_xlabel("Time [$s$]", fontsize=18)
                    else:
                        axs.plot(
                            obs_info["freqs"][II],
                            conversion * best_fit_obs_amp[freq_idx],
                            color=protocol_info["experiment_colors"][exp_idx],
                            marker="v",
                            linestyle="",
                            label="model output",
                        )
                        if phase:
                            axs_phase.plot(
                                obs_info["freqs"][II],
                                best_fit_obs_phase[freq_idx],
                                color=protocol_info["experiment_colors"][exp_idx],
                                marker="v",
                                linestyle="",
                                label="model output",
                            )
                            axs_phase.set_ylabel(
                                f"${obs_name_for_plot}$ phase", fontsize=18
                            )
                        axs.set_xlim(0.0, obs_info["freqs"][II][-1])
                        axs.set_xlabel("frequency [$Hz$]", fontsize=18)
                    this_obs_waveform_plotted = True

                dt = self.client.dt

                if obs_info["data_types"][II] == "constant":
                    pt = obs_info["plot_type"][II]
                    if pt == "horizontal":
                        const_plot_gt = obs_info["ground_truth_const"][const_idx] * np.ones(
                            (n_steps_per_sub_count[subexp_count] + 1,)
                        )
                        const_plot_bf = best_fit_obs_const[const_idx] * np.ones(
                            (n_steps_per_sub_count[subexp_count] + 1,)
                        )
                        axs.plot(
                            tSim_per_sub_count[subexp_count],
                            conversion * const_plot_gt,
                            color=obs_info["plot_colors"][II],
                            linestyle="--",
                            label=f'{obs_info["operations"][II]} gt',
                        )
                        axs.plot(
                            tSim_per_sub_count[subexp_count],
                            conversion * const_plot_bf,
                            color=obs_info["plot_colors"][II],
                            linestyle="-",
                            label=f'{obs_info["operations"][II]} output',
                        )
                    elif pt == "horizontal_from_min":
                        min_val = np.min(series_per_sub[II])
                        const_plot_gt = (
                            min_val + obs_info["ground_truth_const"][const_idx]
                        ) * np.ones((n_steps_per_sub_count[subexp_count] + 1,))
                        const_plot_bf = (
                            min_val + best_fit_obs_const[const_idx]
                        ) * np.ones((n_steps_per_sub_count[subexp_count] + 1,))
                        axs.plot(
                            tSim_per_sub_count[subexp_count],
                            conversion * const_plot_gt,
                            color=obs_info["plot_colors"][II],
                            linestyle="--",
                            label=f'{obs_info["operations"][II]} gt',
                        )
                        axs.plot(
                            tSim_per_sub_count[subexp_count],
                            conversion * const_plot_bf,
                            color=obs_info["plot_colors"][II],
                            linestyle="-",
                            label=f'{obs_info["operations"][II]} output',
                        )
                    elif pt == "vertical":
                        axs.axvline(
                            x=obs_info["ground_truth_const"][const_idx]
                            - protocol_info["pre_times"][exp_idx],
                            color=obs_info["plot_colors"][II],
                            linestyle="--",
                            label=f'{obs_info["operations"][II]} desired',
                        )
                        axs.axvline(
                            x=best_fit_obs_const[const_idx]
                            - protocol_info["pre_times"][exp_idx],
                            color=obs_info["plot_colors"][II],
                            label=f'{obs_info["operations"][II]} output',
                        )
                    elif pt == "vertical_from_subexp_start":
                        t_gt = (
                            obs_info["ground_truth_const"][const_idx]
                            + tSim_per_sub_count[subexp_count][0]
                        )
                        t_bf = (
                            best_fit_obs_const[const_idx]
                            + tSim_per_sub_count[subexp_count][0]
                        )
                        axs.axvline(
                            x=t_gt,
                            color=obs_info["plot_colors"][II],
                            linestyle="--",
                            label=f'{obs_info["operations"][II]} desired',
                        )
                        axs.axvline(
                            x=t_bf,
                            color=obs_info["plot_colors"][II],
                            label=f'{obs_info["operations"][II]} output',
                        )
                    elif pt in (
                        None,
                        "None",
                        "none",
                        "NULL",
                        "null",
                        "Null",
                        np.nan,
                        "nan",
                    ):
                        pass
                    else:
                        print(
                            f'plot_type for {obs_info["obs_names"][II]} '
                            f"of {obs_info['plot_type'][II]} is not recognised",
                            "for constants it must be in [None, horizontal, veritical, horizontal_from_min], exiting",
                        )
                        exit()
                elif obs_info["data_types"][II] == "series":
                    start_time = np.sum(
                        protocol_info["sim_times"][exp_idx][:this_sub_idx]
                    )
                    t_obs = np.linspace(
                        start_time,
                        start_time + protocol_info["sim_times"][exp_idx][this_sub_idx],
                        len(obs_info["ground_truth_series"][series_idx]),
                    )
                    axs.plot(
                        t_obs,
                        conversion * obs_info["ground_truth_series"][series_idx],
                        "k--",
                        label="gt",
                    )
                elif obs_info["data_types"][II] == "frequency":
                    axs.plot(
                        obs_info["freqs"][II],
                        conversion * obs_info["ground_truth_amp"][freq_idx],
                        "kx",
                        label="gt",
                    )
                    if phase:
                        axs_phase.plot(
                            obs_info["freqs"][II],
                            obs_info["ground_truth_phase"][freq_idx],
                            "kx",
                            label="gt",
                        )
                elif obs_info["data_types"][II] == "prob_dist":
                    if obs_info["plot_type"][II] == "horizontal":
                        means = obs_info["ground_truth_prob_dist_params"][
                            prob_dist_idx
                        ]["means"]
                        for mean_idx, val_to_plot in enumerate(means):
                            mean_plot = val_to_plot * np.ones(
                                (n_steps_per_sub_count[subexp_count] + 1,)
                            )
                            axs.plot(
                                tSim_per_sub_count[subexp_count],
                                conversion * mean_plot,
                                color=obs_info["plot_colors"][II],
                                linestyle="--",
                                label=f'{obs_info["operations"][II]} gt mean {mean_idx}',
                            )
                        val_bf = (
                            best_fit_obs_prob_dist[prob_dist_idx]
                            * np.ones((n_steps_per_sub_count[subexp_count] + 1,))
                        )
                        axs.plot(
                            tSim_per_sub_count[subexp_count],
                            conversion * val_bf,
                            color=obs_info["plot_colors"][II],
                            linestyle="-",
                            label=f'{obs_info["operations"][II]} output',
                        )
                    elif obs_info["plot_type"] is None:
                        pass

                if (
                    exp_idx == obs_info["experiment_idxs"][II]
                    and this_sub_idx == obs_info["subexperiment_idxs"][II]
                ):
                    if obs_info["data_types"][II] == "constant":
                        percent_error_vec[II] = (
                            100
                            * (
                                best_fit_obs_const[const_idx]
                                - obs_info["ground_truth_const"][const_idx]
                            )
                            / (obs_info["ground_truth_const"][const_idx] + 1e-10)
                        )
                        std_error_vec[II] = (
                            best_fit_obs_const[const_idx]
                            - obs_info["ground_truth_const"][const_idx]
                        ) / obs_info["std_const_vec"][const_idx]
                    elif obs_info["data_types"][II] == "series":
                        if obs_info["obs_dt"][series_idx] != dt:
                            time_series = np.linspace(
                                0,
                                best_fit_obs_series[series_idx].shape[0] * dt,
                                best_fit_obs_series[series_idx].shape[0],
                            )
                            gs = obs_info["ground_truth_series"][series_idx]
                            obs_time_series = np.linspace(
                                0,
                                gs.shape[0] * obs_info["obs_dt"][series_idx],
                                gs.shape[0],
                            )
                            series_entry = np.interp(
                                obs_time_series,
                                time_series,
                                best_fit_obs_series[series_idx],
                            )
                            obs_entry = gs
                            std_entry = obs_info["std_series_vec"][series_idx]
                        else:
                            min_len_series = min(
                                obs_info["ground_truth_series"][series_idx].shape[0],
                                len(best_fit_obs_series[series_idx]),
                            )
                            series_entry = best_fit_obs_series[series_idx][:min_len_series]
                            obs_entry = obs_info["ground_truth_series"][series_idx][
                                :min_len_series
                            ]
                            std_entry = obs_info["std_series_vec"][series_idx][
                                :min_len_series
                            ]
                        percent_error_vec[II] = (
                            100
                            * np.sum(
                                np.abs(
                                    (obs_entry - series_entry) / (np.mean(obs_entry))
                                )
                            )
                            / len(obs_entry)
                        )
                        std_error_vec[II] = np.sum(
                            np.abs((obs_entry - series_entry / (std_entry)))
                            / len(obs_entry)
                        )
                    elif obs_info["data_types"][II] == "frequency":
                        std_error_vec[II] = np.sum(
                            np.abs(
                                (
                                    best_fit_obs_amp[freq_idx]
                                    - obs_info["ground_truth_amp"][freq_idx]
                                )
                                * obs_info["weight_amp_vec"][freq_idx]
                                / obs_info["std_amp_vec"][freq_idx]
                            )
                            / len(best_fit_obs_amp[freq_idx])
                        )
                        ga = obs_info["ground_truth_amp"][freq_idx]
                        percent_error_vec[II] = (
                            100
                            * np.sum(
                                np.abs(
                                    (best_fit_obs_amp[freq_idx] - ga)
                                    / (np.mean(ga))
                                )
                            )
                            / len(best_fit_obs_amp[freq_idx])
                        )
                        if phase:
                            phase_error_vec[II] = np.sum(
                                np.abs(
                                    (
                                        best_fit_obs_phase[freq_idx]
                                        - obs_info["ground_truth_phase"][freq_idx]
                                    )
                                    * obs_info["weight_phase_vec"][freq_idx]
                                )
                            ) / len(best_fit_obs_phase[freq_idx])
                    elif obs_info["data_types"][II] == "prob_dist":
                        print(
                            "prob dist error not implemented properly yet error from first mean presented"
                        )
                        gpp = obs_info["ground_truth_prob_dist_params"][prob_dist_idx]
                        percent_error_vec[II] = (
                            100
                            * (
                                best_fit_obs_prob_dist[prob_dist_idx] - gpp["means"][0]
                            )
                            / (gpp["means"][0] + 1e-10)
                        )
                        if "stds" in gpp:
                            std_error_vec[II] = (
                                best_fit_obs_prob_dist[prob_dist_idx] - gpp["means"][0]
                            ) / gpp["stds"][0]
                        else:
                            std_error_vec[II] = 0.0

            plot_saved = False

            axs.legend(fontsize=10)
            if phase:
                axs_phase.legend(loc="upper right", fontsize=10)
            fig.tight_layout()
            if phase:
                fig_phase.tight_layout()
            self._save_reconstruction_figure_bundle(
                plot_dir,
                prefix,
                obs_stub,
                plot_idx,
                fig,
                axs,
                fig_phase,
                axs_phase,
                phase,
            )
            plt.close(fig)
            if phase:
                plt.close(fig_phase)

            plot_saved = True
            plot_idx += 1
            if unique_obs_count != len(obs_tuples_unique) - 1:
                fig, axs = plt.subplots(squeeze=False)
                axs = axs[0, 0]
                if phase:
                    fig_phase, axs_phase = plt.subplots(squeeze=False)
                    axs_phase = axs_phase[0, 0]
                plot_saved = False

        self.finalize_reconstruction_if_unsaved(
            plot_saved, axs, plot_dir, prefix, obs_stub, plot_idx
        )

        return percent_error_vec, std_error_vec, phase_error_vec

    def finalize_reconstruction_if_unsaved(
        self,
        plot_saved: bool,
        axs,
        plot_dir: str,
        prefix: str,
        obs_stub: str,
        plot_idx: int,
    ) -> None:
        """Save the last open reconstruction figure if the unique-obs loop did not."""
        if plot_saved:
            return
        axs.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        for ext in ("eps", "pdf"):
            plt.savefig(
                os.path.join(
                    plot_dir,
                    f"reconstruct_{prefix}_{obs_stub}_{plot_idx}.{ext}",
                )
            )
        plt.close()

    @staticmethod
    def _save_reconstruction_figure_bundle(
        plot_dir,
        prefix,
        obs_stub,
        plot_idx,
        fig,
        axs,
        fig_phase,
        axs_phase,
        phase,
    ) -> None:
        base = os.path.join(
            plot_dir, f"reconstruct_{prefix}_{obs_stub}_{plot_idx}"
        )
        fig.savefig(base + ".eps")
        fig.savefig(base + ".pdf")
        axs.legend().get_frame().set_alpha(0.5)
        fig.savefig(base + ".png")
        if phase:
            pbase = os.path.join(
                plot_dir, f"phase_reconstruct_{prefix}_{obs_stub}_{plot_idx}"
            )
            fig_phase.savefig(pbase + ".eps")
            fig_phase.savefig(pbase + ".pdf")
            axs_phase.legend().get_frame().set_alpha(0.5)
            fig_phase.savefig(pbase + ".png")

    def save_error_vectors(
        self, percent_error_vec: np.ndarray, std_error_vec: np.ndarray
    ) -> None:
        out = self.client.output_dir
        np.save(os.path.join(out, "percent_error_vec.npy"), percent_error_vec)
        np.save(os.path.join(out, "std_error_vec.npy"), std_error_vec)
        # The error vectors are positional: entry i belongs to data_items[i]. Every
        # consumer therefore re-derives the labels by reading obs_data.json in the
        # same order, so reordering that file silently relabels every bar -- a wrong
        # plot that looks entirely right. These artefacts are an interface now, not an
        # implementation detail (#341), so the names CA itself used are saved beside
        # them and the data identifies itself.
        #
        # Raw names, not the '$...$' mathtext the bar plots wrap them in: this is data
        # for whoever reads it. A consumer wanting mathtext can add the delimiters; one
        # wanting the plain name could not reliably strip them, since a name may
        # legitimately contain '$'.
        np.save(os.path.join(out, "error_vec_names.npy"), self.observable_names())

    def observable_names(self) -> np.ndarray:
        """The per-observable labels, in the order the error vectors use.

        Public because it defines the meaning of ``error_vec_names.npy``: this is
        the ordering an external consumer binds to.
        """
        obs_info = self.client.obs_info
        return np.array(
            [str(obs_info["names_for_plotting"][II]) for II in range(obs_info["num_obs"])]
        )

    def _observable_names_for_error_plots(self) -> np.ndarray:
        """The same labels wrapped for matplotlib mathtext, for the bar plots."""
        return np.array([f"${name}$" for name in self.observable_names()])

    def plot_percent_error_bar_pages(
        self, obs_names_for_plot: np.ndarray, percent_error_vec: np.ndarray
    ) -> None:
        plot_dir = self.client.plot_dir
        prefix = self.client.file_name_prefix
        obs_stub = self.client.param_id_obs_file_prefix
        obs_info = self.client.obs_info
        protocol_info = self.client.protocol_info

        do_plots_per_exp = True
        num_plots = (
            len(protocol_info["pre_times"])
            if do_plots_per_exp
            else len(obs_names_for_plot) // 10 + 1
        )

        if len(percent_error_vec) == 0:
            return
        y_min_percent = 1.05 * np.min(percent_error_vec)
        y_max_percent = 1.05 * np.max(percent_error_vec)

        for plot_idx_inner in range(num_plots):
            fig, axs = plt.subplots()
            if do_plots_per_exp:
                obs_idx_for_plot = [
                    II
                    for II in range(obs_info["num_obs"])
                    if obs_info["experiment_idxs"][II] == plot_idx_inner
                ]
                if len(obs_idx_for_plot) == 0:
                    plt.close(fig)
                    continue
            else:
                start_idx = plot_idx_inner * 10
                end_idx = min(start_idx + 10, len(obs_names_for_plot))
                obs_idx_for_plot = list(range(start_idx, end_idx))

            axs.bar(
                obs_names_for_plot[obs_idx_for_plot],
                percent_error_vec[obs_idx_for_plot],
                label="% error",
                width=1.0,
                color="b",
                edgecolor="black",
            )
            axs.set_ylim(y_min_percent, y_max_percent)
            axs.axhline(y=0.0, linewidth=3, color="k", linestyle="dotted")
            axs.set_ylabel(r"E$_{\%}$")
            plt.xticks(rotation=90)
            plt.tight_layout()
            for ext in ("eps", "pdf", "png"):
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        f"error_bars_{prefix}_{obs_stub}_{plot_idx_inner}.{ext}",
                    )
                )
            plt.close(fig)

    def plot_std_error_bar_pages(
        self, obs_names_for_plot: np.ndarray, std_error_vec: np.ndarray
    ) -> None:
        plot_dir = self.client.plot_dir
        prefix = self.client.file_name_prefix
        obs_stub = self.client.param_id_obs_file_prefix
        obs_info = self.client.obs_info
        protocol_info = self.client.protocol_info

        do_plots_per_exp = True
        num_plots = (
            len(protocol_info["pre_times"])
            if do_plots_per_exp
            else len(obs_names_for_plot) // 10 + 1
        )

        for plot_idx_inner in range(num_plots):
            fig, axs = plt.subplots()
            if do_plots_per_exp:
                obs_idx_for_plot = [
                    II
                    for II in range(obs_info["num_obs"])
                    if obs_info["experiment_idxs"][II] == plot_idx_inner
                ]
                if len(obs_idx_for_plot) == 0:
                    plt.close(fig)
                    continue
                axs.bar(
                    obs_names_for_plot[obs_idx_for_plot],
                    std_error_vec[obs_idx_for_plot],
                    label="% error",
                    width=1.0,
                    color="b",
                    edgecolor="black",
                )
            else:
                if plot_idx_inner == num_plots - 1:
                    axs.bar(
                        obs_names_for_plot[plot_idx_inner * 10 :],
                        std_error_vec[plot_idx_inner * 10 :],
                        label="% error",
                        width=1.0,
                        color="b",
                        edgecolor="black",
                    )
                else:
                    axs.bar(
                        obs_names_for_plot[
                            plot_idx_inner * 10 : plot_idx_inner * 10 + 10
                        ],
                        std_error_vec[
                            plot_idx_inner * 10 : plot_idx_inner * 10 + 10
                        ],
                        label="% error",
                        width=1.0,
                        color="b",
                        edgecolor="black",
                    )
            axs.axhline(y=0.0, linewidth=3, color="k", linestyle="dotted")
            axs.set_ylabel("E$_{std}$")
            plt.xticks(rotation=90)
            plt.tight_layout()
            for ext in ("eps", "pdf", "png"):
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        f"std_error_bars_{prefix}_{obs_stub}_{plot_idx_inner}.{ext}",
                    )
                )
            plt.close(fig)

    @staticmethod
    def _normalize_sub_vals(vals: Union[Sequence, Any]) -> List[Any]:
        if isinstance(vals, (list, tuple, np.ndarray)):
            return list(vals)
        return [vals]

    @staticmethod
    def _schedule_segments_for_param_experiment(
        protocol_info: dict, exp_idx: int, vals: Sequence[Any]
    ) -> Optional[List[Tuple[float, float, Any]]]:
        """
        Build (t_start, t_end, scheduled_value) for each subexperiment.

        Matches the CVODE/Myokit timestepping order: segment 0 covers
        pre_time + sim_times[exp][0]; later segments cover only their sim_times.
        ``scheduled_value`` is numeric or a protocol_traces key (str).

        Returns None when the observation row shape does not match the protocol.
        """
        try:
            n_sub_e = protocol_info["num_sub_per_exp"][exp_idx]
        except (IndexError, KeyError, TypeError):
            return None
        vals_list = ParamIDPlotOutputs._normalize_sub_vals(vals)
        if len(vals_list) != n_sub_e:
            return None

        try:
            pre_t = float(protocol_info["pre_times"][exp_idx])
            sim_blk = protocol_info["sim_times"][exp_idx]
        except (IndexError, KeyError, TypeError):
            return None

        segments: List[Tuple[float, float, Any]] = []
        t_cursor = 0.0

        dur0 = pre_t + float(sim_blk[0])
        segments.append((t_cursor, t_cursor + dur0, vals_list[0]))
        t_cursor += dur0

        for k in range(1, n_sub_e):
            dur = float(sim_blk[k])
            segments.append((t_cursor, t_cursor + dur, vals_list[k]))
            t_cursor += dur

        return segments

    def _plot_param_segment(
        self,
        ax,
        t0: float,
        t1: float,
        val: Any,
        traces: Dict[str, Any],
        color: Any,
        label: Optional[str],
    ) -> None:
        """Draw one schedule segment on *ax* at absolute experiment time [t0, t1)."""
        str_key = isinstance(val, str)

        def _numeric_ok(v):
            if isinstance(v, bool):
                return False
            return isinstance(v, (int, float, np.integer, np.floating))

        if not str_key and _numeric_ok(val):
            y = float(val)
            ax.plot([t0, t1], [y, y], color=color, linewidth=2.0, label=label, zorder=2)
            return

        if str_key:
            trace_key = val
            if trace_key not in traces:
                ax.annotate(
                    f"Missing trace '{trace_key}'",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    clip_on=False,
                )
                return
            trace = traces[trace_key]
            if isinstance(trace, dict) and "t" in trace:
                tt = np.asarray(trace["t"], dtype=float)
            elif hasattr(trace, "to_dict"):
                d = trace.to_dict()
                tt = np.asarray(d.get("t", []), dtype=float)
            else:
                ax.annotate(
                    f"Trace '{trace_key}' has no time vector",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    clip_on=False,
                )
                return
            if isinstance(trace, dict) and "values" in trace:
                yy = np.asarray(trace["values"], dtype=float)
            elif hasattr(trace, "to_dict"):
                d = trace.to_dict()
                yy = np.asarray(d.get("values", []), dtype=float)
            else:
                return
            if tt.size != yy.size or tt.size == 0:
                ax.annotate(
                    f"Bad trace '{trace_key}' lengths",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    clip_on=False,
                )
                return
            ax.plot(t0 + tt, yy, color=color, linewidth=1.8, label=label, zorder=2)
            return

        # Non-string, non-supported (e.g. bad type): annotate
        ax.annotate(
            f"(unplotable)\n{type(val).__name__}: {val!s}",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            clip_on=False,
        )

    @staticmethod
    def _safe_param_fname_token(pname: str) -> str:
        tok = pname.replace("/", "__").strip()
        tok = re.sub(r"[^\w.\-]+", "_", tok, flags=re.ASCII)
        tok = tok.strip("_") or "param"
        if len(tok) > 128:
            tok = tok[:128]
        return tok

    # Experiments wrapped left-to-right, then downward (reading order).
    _PROTOCOL_PARAM_SUBPLOTS_PER_ROW = 3

    def plot_protocol_params_to_change(self) -> None:
        """
        One **saved figure file** per ``params_to_change`` key: subplots sit on a grid
        with **three panels per row** (:math:`3` columns); rows wrap until every
        experiment has one axes (time vs applied parameter).

        Numeric schedules render as horizontal segments; uniform numeric values yield
        a single horizontal line. String entries resolve ``protocol_traces`` as before.
        """
        client = self.client
        if client.rank != 0:
            return
        if client.protocol_info is None:
            return
        ptc = client.protocol_info.get("params_to_change") or {}
        if not ptc:
            return
        num_exp = int(client.protocol_info.get("num_experiments", 0))
        if num_exp < 1:
            return

        param_keys = list(ptc.keys())
        traces_raw = client.protocol_info.get("protocol_traces") or {}

        traces: Dict[str, Any] = traces_raw if isinstance(traces_raw, dict) else {}
        exp_labels = client.protocol_info.get("experiment_labels") or [None] * num_exp
        exp_colors = client.protocol_info.get("experiment_colors") or []

        ncol = min(self._PROTOCOL_PARAM_SUBPLOTS_PER_ROW, num_exp)
        nrow = max(1, int(np.ceil(num_exp / ncol)))

        for pname in param_keys:
            row_vals = ptc[pname]
            fname_tok = self._safe_param_fname_token(pname)
            fig, axes = plt.subplots(
                nrow,
                ncol,
                squeeze=False,
                figsize=(max(3.7 * ncol, 6.0), max(3.2 * nrow, 2.9)),
                sharey=False,
            )

            fig.suptitle(
                pname.replace("/", " / "),
                fontsize=11,
                y=1.06,
            )

            for k in range(num_exp, nrow * ncol):
                r, cc = divmod(k, ncol)
                axes[r, cc].set_visible(False)

            if not isinstance(row_vals, (list, tuple)) or len(row_vals) < num_exp:
                for exp_idx in range(num_exp):
                    r, cc = divmod(exp_idx, ncol)
                    ax = axes[r, cc]
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "params_to_change row length mismatch vs num_experiments",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
                base = os.path.join(
                    client.plot_dir,
                    f"protocol_params_to_change_{client.file_name_prefix}_"
                    f"{client.param_id_obs_file_prefix}_{fname_tok}",
                )
                fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
                for ext in ("eps", "pdf", "png"):
                    dpi = 150 if ext == "png" else None
                    if dpi:
                        fig.savefig(base + "." + ext, dpi=dpi)
                    else:
                        fig.savefig(base + "." + ext)
                plt.close(fig)
                continue

            for exp_idx in range(num_exp):
                r, cc = divmod(exp_idx, ncol)
                ax = axes[r, cc]
                elab = exp_labels[exp_idx] if exp_idx < len(exp_labels) else None
                sub_title = f"Experiment {exp_idx}"
                if (
                    elab is not None
                    and str(elab).strip() != ""
                    and str(elab).lower() != "none"
                ):
                    sub_title = f"{sub_title}\n{elab}"
                ax.set_title(sub_title, fontsize=9)

                c = (
                    exp_colors[exp_idx]
                    if exp_idx < len(exp_colors)
                    else f"C{exp_idx % 10}"
                )

                vals = ParamIDPlotOutputs._normalize_sub_vals(row_vals[exp_idx])
                segments = self._schedule_segments_for_param_experiment(
                    client.protocol_info, exp_idx, vals
                )

                if segments is None:
                    ax.axis("off")
                    ax.annotate(
                        "Sub-count mismatch vs protocol",
                        xy=(0.5, 0.5),
                        xycoords="axes fraction",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=c,
                        clip_on=False,
                    )
                else:
                    for seg_i, (t0, t1, val) in enumerate(segments):
                        # Label only the first segment (trace vs piecewise cue)
                        if seg_i != 0:
                            lbl_tr = None
                        elif isinstance(val, str):
                            lbl_tr = f"trace `{val}`"
                        else:
                            lbl_tr = None
                        self._plot_param_segment(
                            ax, t0, t1, val, traces, c, lbl_tr
                        )
                    ax.axhline(
                        0.0, color="0.82", linewidth=0.8, linestyle="-", zorder=0
                    )
                    ax.grid(True, axis="both", linestyle=":", alpha=0.35)
                    handles, legends = ax.get_legend_handles_labels()
                    if any(legends):
                        ax.legend(handles, legends, fontsize=7, loc="best")

                ax.set_xlabel(r"Time [$s$]", fontsize=9)

            for r in range(nrow):
                if r * ncol < num_exp:
                    axes[r, 0].set_ylabel(r"Applied value", fontsize=9)

            fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.90))
            base = os.path.join(
                client.plot_dir,
                f"protocol_params_to_change_{client.file_name_prefix}_"
                f"{client.param_id_obs_file_prefix}_{fname_tok}",
            )
            fig.savefig(base + ".eps")
            fig.savefig(base + ".pdf")
            fig.savefig(base + ".png", dpi=150)
            plt.close(fig)

    def print_observable_errors(
        self,
        phase: bool,
        percent_error_vec: np.ndarray,
        phase_error_vec: np.ndarray,
    ) -> None:
        obs_info = self.client.obs_info
        gt_df = self.client.gt_df

        print("______observable errors______")
        for obs_idx in range(obs_info["num_obs"]):
            dt_row = gt_df.iloc[obs_idx]["data_type"]
            if dt_row == "constant":
                if obs_info["operations"][obs_idx] is not None:
                    print(
                        f'{obs_info["names_for_plotting"][obs_idx]} {obs_info["operations"][obs_idx]} error:'
                    )
                else:
                    print(
                        f'{obs_info["names_for_plotting"][obs_idx]} {obs_info["data_types"][obs_idx]} error:'
                    )
                print(f"{percent_error_vec[obs_idx]:.2f} %")
            if dt_row == "series":
                if obs_info["operations"][obs_idx] is not None:
                    print(
                        f'{obs_info["names_for_plotting"][obs_idx]} {obs_info["operations"][obs_idx]} series error:'
                    )
                else:
                    print(
                        f'{obs_info["obs_names"][obs_idx]} {obs_info["data_types"][obs_idx]} error:'
                    )
                print(f"{percent_error_vec[obs_idx]:.2f} %")
            if dt_row == "frequency":
                print(
                    f'{obs_info["names_for_plotting"][obs_idx]} {obs_info["data_types"][obs_idx]} error:'
                )
                print(f"{percent_error_vec[obs_idx]:.2f} %")
                if phase:
                    print(
                        f'{obs_info["names_for_plotting"][obs_idx]} {obs_info["data_types"][obs_idx]} phase error:'
                    )
                    print(f"{phase_error_vec[obs_idx]:.2f}")
