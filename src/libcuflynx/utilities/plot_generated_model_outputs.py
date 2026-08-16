# import opencor as oc
import numpy as np
import os
from libcuflynx.solver_wrappers.opencor_helper import SimulationHelper
from libcuflynx.utilities import paperPlotSetup
paperPlotSetup.Setup_Plot(3)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root_dir = os.path.join(os.path.dirname(__file__), '../../..')

# TODO get all of the info from the user_inputs.yaml file

file_path = os.path.join(root_dir, 'generated_models/3compartment/3compartment.cellml')
output_file_path = "generated_models/3compartment/generated_outputs"
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)
pre_time = 20
sim_time = 2

param_names = ['heart/q_lv_init', 'venous_svc/C']
param_vals_list_of_lists = [[1e-3, 1e-6], [1.5e-3, 1e-6], [1e-3, 2e-6], [1.5e-3, 2e-6]] # create a list of lists of parameter values to test. 
                                                                        # This could be done automatically from individual parameter ranges

sim_object = SimulationHelper(file_path, 0.01, sim_time, solver_info={'MaximumStep':0.001, 'MaximumNumberOfSteps':500000}, pre_time=pre_time)

for II in range(len(param_vals_list_of_lists)):
    param_vals = param_vals_list_of_lists[II]
    sim_object.set_param_vals(param_names, param_vals)
    sim_object.reset_states()
    sim_object.run()

    y = sim_object.get_results(['venous_svc/u', 'aortic_root/v', 'aortic_root/u', 'heart/u_rv'])
    t = sim_object.tSim - pre_time

    Pa_to_mmHg = 1/133.322
    m3_to_mL = 1e6

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0,0].plot(t, y[0][0]*Pa_to_mmHg)
    axs[0,1].plot(t, y[1][0]*m3_to_mL)
    axs[1,0].plot(t, y[2][0]*Pa_to_mmHg)
    axs[1,1].plot(t, y[3][0]*Pa_to_mmHg)

    axs[0,0].set_xlim([0, sim_time])
    # axs[0,0].set_ylim(ymin=)
    axs[0,0].set_xlabel('Time [s]')
    axs[0,0].set_ylabel('venous P [mmHg]')
    axs[0,1].set_xlim([0, sim_time])
    # axs[0,1].set_ylim([0, 50])
    axs[0,1].set_xlabel('Time [s]')
    axs[0,1].set_ylabel('aortic root v [mL/s]')
    axs[1,0].set_xlim([0, sim_time])
    # axs[1,0].set_ylim([0, 50])
    axs[1,0].set_xlabel('Time [s]')
    axs[1,0].set_ylabel('aortic root P [mmHg]')
    axs[1,1].set_xlim([0, sim_time])
    # axs[1,1].set_ylim([0, 50])
    axs[1,1].set_xlabel('Time [s]')
    axs[1,1].set_ylabel('RV P [mmHg]')

    plt.savefig(os.path.join(output_file_path, f'model_outputs_u_{II}.png'))
    plt.savefig(os.path.join(output_file_path, f'model_outputs_u_{II}.eps'))
