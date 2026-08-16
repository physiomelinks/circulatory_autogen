'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
import numpy as np
# Not `from mpi4py import MPI`: that import initialises MPI and registers an
# atexit MPI_Finalize, and with no launcher present that finalise is what aborts
# on macOS when a NIC goes away (#396). Under mpiexec get_MPI hands back the real
# mpi4py.MPI, so a multi-rank run is unchanged.
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI

MPI = _get_MPI()
import matplotlib
import matplotlib.pyplot as plt
from libcuflynx.utilities import paperPlotSetup
matplotlib.use('Agg')
paperPlotSetup.Setup_Plot(3)

root_dir_path = os.path.join(os.path.dirname(__file__), '../../..')

resources_dir_path = os.path.join(root_dir_path, 'resources')
param_id_dir_path = os.path.join(root_dir_path, 'src/libcuflynx/param_id')
user_plots_path = os.path.join(root_dir_path, 'user_plots')
generated_models_dir_path = os.path.join(root_dir_path, 'generated_models')

from libcuflynx.param_id.paramID import CVS0DParamID
from libcuflynx.utilities.utility_funcs import obj_to_string
import traceback
from libcuflynx.solver_wrappers.opencor_helper import *

if __name__ == '__main__':

    try:

        if not os.path.exists(user_plots_path):
            os.mkdir(user_plots_path)

        param_id_method = 'genetic_algorithm'
        file_name_prefix = '3compartment'
        model_path = os.path.join(generated_models_dir_path, f'{file_name_prefix}.cellml')
        file_name_prefix_phys = 'physiological'
        model_path_phys = os.path.join(generated_models_dir_path, f'{file_name_prefix_phys}.cellml')

        model_type = 'CVS0D' # TODO make this an input variable eventually

        input_params_to_id = True
        if input_params_to_id:
            input_params_path = os.path.join(resources_dir_path, f'{file_name_prefix}_params_for_id.csv')
            input_params_path_phys = os.path.join(resources_dir_path, f'{file_name_prefix_phys}_params_for_id.csv')
        else:
            input_params_path = False
        param_id_obs_path = '/home/finbar/Documents/data/cardiohance_data/cardiohance_observables.json'
        # TODO do the param id for physiological model
        # param_id_obs_path_phys = 'home/finbar/Documents/git_projects/circulatory_autogen/physiological_observables.json'

        # set the simulation time where the cost is calculated (sim_time) and the amount of
        # simulation time it takes to get to an oscilating steady state before that (pre_time)
        if file_name_prefix == '3compartment':
            pre_time = 6.0
        else:
            pre_time = 20.0
        sim_time = 2.0
        dt = 0.01
        max_step = 0.001

        param_id = CVS0DParamID(model_path, model_type, param_id_method, file_name_prefix,
                                input_params_path=input_params_path, param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time, maximumStep=max_step, dt=dt)


        # print(obj_to_string(param_id))
        param_id.simulate_with_best_param_vals()
        sim_helper = param_id.param_id.sim_helper

        tSim = sim_helper.tSim - pre_time

        # define observables to plot
        obs_state_list = ['aortic_root/v', 'heart/q_lv']
        obs_alg_list = ['aortic_root/u', 'heart/u_lv']
        obs_list = obs_state_list + obs_alg_list

        v_ar_idx = 0
        q_lv_idx = 1
        u_ar_idx = 2
        u_lv_idx = 3

        # get ground truth values
        for II in range(param_id.obs_info["num_obs"]):
            if param_id.gt_df.iloc[II]['data_type'] == 'series':
                if param_id.gt_df.iloc[II]['variable'] == 'heart/q_lv':
                    gt_q_lv = np.array(param_id.gt_df.iloc[II]['series'])
                if param_id.gt_df.iloc[II]['variable'] == 'aortic_root/v':
                    gt_v_ar = np.array(param_id.gt_df.iloc[II]['series'])
                if param_id.gt_df.iloc[II]['variable'] == 'aortic_root/u':
                    gt_u_ar = np.array(param_id.gt_df.iloc[II]['series'])

        # _______  Now get results of unmodified simulation ______ #

        pred_obs_nom = sim_helper.get_results(obs_state_list, obs_alg_list)

        # get results with modified venous compliance x5
        mod_factors = [5.0]
        pred_obs_C_venous_increase = sim_helper.modify_params_and_run_and_get_results([], ['venous_svc/C'],
                                                 mod_factors, obs_state_list, obs_alg_list)
        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()
        # get results with modified venous compliance x10
        mod_factors = [10.0]
        pred_obs_C_venous_increase_10 = sim_helper.modify_params_and_run_and_get_results([], ['venous_svc/C'],
                                                 mod_factors, obs_state_list, obs_alg_list)


        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        # get results with modified stenosis to 0.5
        mod_variables = ['heart/M_st_aov', 'heart/K_vo_aov', 'heart/K_vc_aov'],
        abs_mod_factors = [0.5, 0.01, 0.01]
        pred_obs_aov_stenosis_50 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                      abs_mod_factors, obs_state_list,
                                                                                      obs_alg_list, absolute=True)
        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        abs_mod_factors = [0.15, 0.002, 0.002]
        pred_obs_aov_stenosis_85 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                abs_mod_factors, obs_state_list,
                                                                                obs_alg_list, absolute=True)

        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        # get results with modified mitral valve stenosis to 0.3
        mod_variables = ['heart/M_st_miv'],
        abs_mod_factors = [0.5]
        pred_obs_miv_stenosis_50 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                abs_mod_factors, obs_state_list,
                                                                                obs_alg_list, absolute=True)
        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        abs_mod_factors = [0.15]
        pred_obs_miv_stenosis_85 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                    abs_mod_factors, obs_state_list,
                                                                                    obs_alg_list, absolute=True)

        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        # get results with modified aortic regurgitation to 0.05
        mod_variables = ['heart/M_rg_aov']
        abs_mod_factors = [0.02]
        pred_obs_aov_regurge_02 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                             abs_mod_factors, obs_state_list,
                                                                             obs_alg_list, absolute=True)
        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        abs_mod_factors = [0.05]
        pred_obs_aov_regurge_05 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                               abs_mod_factors, obs_state_list,
                                                                               obs_alg_list, absolute=True)

        # get results with modified mitral valve regurgitation to 0.05
        mod_variables = ['heart/M_rg_miv']
        abs_mod_factors = [0.02]
        pred_obs_miv_regurge_02 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                   abs_mod_factors, obs_state_list,
                                                                                   obs_alg_list, absolute=True)
        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        abs_mod_factors = [0.05]
        pred_obs_miv_regurge_05 = sim_helper.modify_params_and_run_and_get_results([], mod_variables,
                                                                                   abs_mod_factors, obs_state_list,
                                                                                   obs_alg_list, absolute=True)

        # simulate with best param vals again to make sure best params are set after resetting.
        param_id.simulate_with_best_param_vals()

        # steps per beat
        n_steps = 101
        # conversions
        m3_to_cm3 = 1e6
        Pa_to_kPa = 1e-3
        # now plot

        # ______ Plot venous compliance increase comparison ______ #
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_xlabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 0].set_ylabel('$P_{lv}$ [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, 200.0)
        axs[0, 0].set_ylim(0.0, 20.0)
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, -n_steps:], 'b', label='nominal')
        axs[0, 0].plot(m3_to_cm3*pred_obs_C_venous_increase[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_C_venous_increase[u_lv_idx, -n_steps:], 'r--', label='$C_{ven}$ x5')
        axs[0, 0].plot(m3_to_cm3*pred_obs_C_venous_increase_10[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_C_venous_increase_10[u_lv_idx, -n_steps:], 'r', label='$C_{ven}$ x10')

        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='nominal')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_C_venous_increase[q_lv_idx, :], 'r--', label='$C_{ven}$ x5')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_C_venous_increase_10[q_lv_idx, :], 'r', label='$C_{ven}$ x10')

        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('$P_{ar}$ [$kPa$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='nominal')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_C_venous_increase[u_ar_idx, :], 'r--', label='$C_{ven}$ x5')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_C_venous_increase_10[u_ar_idx, :], 'r', label='$C_{ven}$ x10')

        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('$v_{ar}$ [$ml/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        axs[1, 1].plot(tSim, m3_to_cm3*gt_v_ar, 'k--', label='experimental')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom[v_ar_idx, :], 'b', label='nominal')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_C_venous_increase[v_ar_idx, :], 'r--', label='$C_{ven}$ x5')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_C_venous_increase_10[v_ar_idx, :], 'r', label='$C_{ven}$ x10')

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_C_venous_increase.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_C_venous_increase.pdf'))

        # ______ Plot aortic valve stenosis comparison ______ #
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_xlabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 0].set_ylabel('$P_{lv}$ [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, 200.0)
        axs[0, 0].set_ylim(0.0, 30.0)
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, -n_steps:], 'b', label='nominal')
        axs[0, 0].plot(m3_to_cm3*pred_obs_aov_stenosis_50[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_aov_stenosis_50[u_lv_idx, -n_steps:], 'r--', label='50% aov stenosis')
        axs[0, 0].plot(m3_to_cm3*pred_obs_aov_stenosis_85[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_aov_stenosis_85[u_lv_idx, -n_steps:], 'r', label='85% aov stenosis')

        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='nominal')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_aov_stenosis_50[q_lv_idx, :], 'r--', label='50% aov stenosis')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_aov_stenosis_85[q_lv_idx, :], 'r', label='85% aov stenosis')

        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('$P_{ar}$ [$kPa$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='nominal')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_aov_stenosis_50[u_ar_idx, :], 'r--', label='50% aov stenosis')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_aov_stenosis_85[u_ar_idx, :], 'r', label='85% aov stenosis')

        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('$v_{ar}$ [$ml/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        axs[1, 1].plot(tSim, m3_to_cm3*gt_v_ar, 'k--', label='experimental')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom[v_ar_idx, :], 'b', label='nominal')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_aov_stenosis_50[v_ar_idx, :], 'r--', label='50% aov stenosis')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_aov_stenosis_85[v_ar_idx, :], 'r', label='85% aov stenosis')

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_aov_stenosis.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_aov_stenosis.pdf'))

        # ______ Plot mitral valve stenosis comparison ______ #
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_xlabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 0].set_ylabel('$P_{lv}$ [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, 200.0)
        axs[0, 0].set_ylim(0.0, 20.0)
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom[q_lv_idx, :],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, :], 'b', label='nominal')
        axs[0, 0].plot(m3_to_cm3*pred_obs_miv_stenosis_50[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_miv_stenosis_50[u_lv_idx, -n_steps:], 'r--', label='50% miv stenosis')
        axs[0, 0].plot(m3_to_cm3*pred_obs_miv_stenosis_85[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_miv_stenosis_85[u_lv_idx, -n_steps:], 'r', label='85% miv stenosis')

        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='nominal')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_miv_stenosis_50[q_lv_idx, :], 'r--', label='50% miv stenosis')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_miv_stenosis_85[q_lv_idx, :], 'r', label='85% miv stenosis')

        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('$P_{ar}$ [$kPa$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='nominal')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_miv_stenosis_50[u_ar_idx, :], 'r--', label='50% miv stenosis')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_miv_stenosis_85[u_ar_idx, :], 'r', label='85% miv stenosis')

        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('$v_{ar}$ [$ml/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        axs[1, 1].plot(tSim, m3_to_cm3*gt_v_ar, 'k--', label='experimental')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom[v_ar_idx, :], 'b', label='nominal')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_miv_stenosis_50[v_ar_idx, :], 'r--', label='50% miv stenosis')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_miv_stenosis_85[v_ar_idx, :], 'r', label='85% miv stenosis')

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_miv_stenosis.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_miv_stenosis.pdf'))

        # ______ Plot regurgitation comparison ______ #
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_xlabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 0].set_ylabel('$P_{lv}$ [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, 200.0)
        axs[0, 0].set_ylim(0.0, 20.0)
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, -n_steps:], 'b', label='nominal')
        axs[0, 0].plot(m3_to_cm3*pred_obs_miv_regurge_02[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_miv_regurge_02[u_lv_idx, -n_steps:], 'r--', label='2% miv regurge')
        axs[0, 0].plot(m3_to_cm3*pred_obs_miv_regurge_05[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_miv_regurge_05[u_lv_idx, -n_steps:], 'r', label='5% miv regurge')

        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='nominal')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_miv_regurge_02[q_lv_idx, :], 'r--', label='2% miv regurge')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_miv_regurge_05[q_lv_idx, :], 'r', label='5% miv regurge')

        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('$P_{ar}$ [$kPa$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='nominal')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_miv_regurge_02[u_ar_idx, :], 'r--', label='2% miv regurge')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_miv_regurge_05[u_ar_idx, :], 'r', label='5% miv regurge')

        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('$v_{ar}$ [$ml/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        axs[1, 1].plot(tSim, m3_to_cm3*gt_v_ar, 'k--', label='experimental')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom[v_ar_idx, :], 'b', label='nominal')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_miv_regurge_02[v_ar_idx, :], 'r--', label='2% miv regurge')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_miv_regurge_05[v_ar_idx, :], 'r', label='5% miv regurge')

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_miv_regurgitation.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_miv_regurgitation.pdf'))

        param_id.close_simulation()

        pre_time = 25.0
        # now do simulation for physiological
        phys_sim_helper = SimulationHelper(model_path_phys, dt, sim_time, maximumNumberofSteps=100000000,
                             maximumStep=max_step, pre_time=pre_time)

        # define observables to plot
        obs_state_list = ['aortic_arch_C46/v', 'heart/q_lv']
        obs_alg_list = ['aortic_arch_C46/u', 'heart/u_lv']
        obs_list = obs_state_list + obs_alg_list

        # v_ar_idx_phys = 0
        # q_lv_idx_phys = 1
        # u_ar_idx_phys = 2
        # u_lv_idx_phys = 3
        success = phys_sim_helper.run()
        if success:
            pred_obs_nom_phys = phys_sim_helper.get_results(obs_state_list, obs_alg_list)
        else:
            print('phys sim failed')
            exit()


        # plot nominal and physiological results
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_xlabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 0].set_ylabel('$P_{lv}$ [$kPa$]', fontsize=14)
        axs[0, 0].set_xlim(0.0, 200.0)
        axs[0, 0].set_ylim(0.0, 20.0)
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, -n_steps:], 'b', label='9 sections')
        axs[0, 0].plot(m3_to_cm3*pred_obs_nom_phys[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom_phys[u_lv_idx, -n_steps:], 'r', label='80 sections')

        axs[0, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[0, 1].set_ylabel('$q_{lv}$ [$ml$]', fontsize=14)
        axs[0, 1].set_xlim(0.0, sim_time)
        axs[0, 1].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='9 sections')
        axs[0, 1].plot(tSim, m3_to_cm3*pred_obs_nom_phys[q_lv_idx, :], 'r', label='80 sections')

        axs[1, 0].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 0].set_ylabel('$P_{ar}$ [$kPa$]', fontsize=14)
        axs[1, 0].set_xlim(0.0, sim_time)
        axs[1, 0].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='9 sections')
        axs[1, 0].plot(tSim, Pa_to_kPa*pred_obs_nom_phys[u_ar_idx, :], 'r', label='80 sections')

        axs[1, 1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1, 1].set_ylabel('$v_{ar}$ [$ml/s$]', fontsize=14)
        axs[1, 1].set_xlim(0.0, sim_time)
        axs[1, 1].plot(tSim, m3_to_cm3*gt_v_ar, 'k--', label='experimental')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom[v_ar_idx, :], 'b', label='9 sections')
        axs[1, 1].plot(tSim, m3_to_cm3*pred_obs_nom_phys[v_ar_idx, :], 'r', label='80 sections')

        fig.align_ylabels(axs[:, 0])
        fig.align_ylabels(axs[:, 1])
        axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_vs_physiological.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_vs_physiological.pdf'))

        # plot 3compartment for Sinclair poster results
        fig, ax = plt.subplots(1, 1)

        # fig.suptitle('Computation of Left Ventricle PV Loop', fontsize=20)
        ax.set_xlabel('LV Volume [$ml$]', fontsize=18)
        ax.set_ylabel('LV Pressure [$kPa$]', fontsize=18)
        ax.set_xlim(0.0, 200.0)
        ax.set_ylim(0.0, 20.0)
        ax.plot(m3_to_cm3*pred_obs_nom[q_lv_idx, -n_steps:],
                       Pa_to_kPa*pred_obs_nom[u_lv_idx, -n_steps:], 'b')

        # fig.align_ylabels(axs[:, 0])
        # fig.align_ylabels(axs[:, 1])
        # axs[1, 1].legend(loc='lower right', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_PV_loop.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_PV_loop.pdf'))


        fig, axs = plt.subplots(2, 1)

        axs[0].set_xlabel('Time [$s$]', fontsize=14)
        axs[0].set_ylabel('LV Volume [$ml$]', fontsize=14)
        axs[0].set_xlim(0.0, 1.0)
        axs[0].set_ylim(0.0, 200.0)
        axs[0].plot(tSim, m3_to_cm3*gt_q_lv, 'k--', label='experimental')
        axs[0].plot(tSim, m3_to_cm3*pred_obs_nom[q_lv_idx, :], 'b', label='simulation')

        axs[1].set_xlabel('Time [$s$]', fontsize=14)
        axs[1].set_ylabel('AR Pressure [$kPa$]', fontsize=14)
        axs[1].set_xlim(0.0, 1.0)
        axs[1].set_ylim(0.0, 18.0)
        axs[1].plot(tSim, Pa_to_kPa*gt_u_ar, 'k--', label='experimental')
        axs[1].plot(tSim, Pa_to_kPa*pred_obs_nom[u_ar_idx, :], 'b', label='simulation')

        fig.align_ylabels(axs[:])
        fig.align_ylabels(axs[:])
        fig.set_figwidth(3)
        axs[1].legend(loc='lower center', fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_exp_vs_sim.eps'))
        plt.savefig(os.path.join(user_plots_path, 'heart_model_3compartment_exp_vs_sim.pdf'))

    except:
        print(traceback.format_exc())
        print("Usage: param_id_method file_name_prefix input_params_to_id, param_id_obs_file")
        print("e.g. bayesian simple_physiological True simple_physiological_obs_data.json")
        exit()
