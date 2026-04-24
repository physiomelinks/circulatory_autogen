import numpy as np
import os, sys
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__))
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
from opencor_helper import SimulationHelper
import paperPlotSetup
paperPlotSetup.Setup_Plot(3)

class ProtocolRunner():
    def __init__(self, model_path, inp_data_dict=None):
        
        if inp_data_dict is None:
            import yaml
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
                if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                    with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                    print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                    print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                            "you want to use the default user_inputs.yaml location")
                    exit()

        self.inp_data_dict = inp_data_dict
        
        self.dt = inp_data_dict['dt']
        self.MaximumStep = inp_data_dict['solver_info']['MaximumStep']
        self.MaximumNumberOfSteps = inp_data_dict['solver_info']['MaximumNumberOfSteps']
        
        sim_time = 1.0 # this will be overridden by the protocol_info later
        self.sim_helper = SimulationHelper(model_path, self.dt, sim_time, solver_info={'MaximumNumberOfSteps':self.MaximumNumberOfSteps, 
                                                                                       'MaximumStep':self.MaximumStep})
        self.variable_names = self.sim_helper.get_all_variable_names()
    
    def get_variable_names(self):
        """ returns the variable names of the model, this is used to plot the results later.
        """

        return self.variable_names
    
    def get_var2idx_dict(self):
        
        var2idx = {}
        for var_idx in range(len(self.variable_names)):
            var2idx[self.variable_names[var_idx]] = var_idx
        return var2idx
        
    def run_protocols(self, model_path, protocol_info=None):

        if protocol_info is None:
            param_id_obs_path = os.path.join(self.inp_data_dict['param_id_obs_path'])
            with open(param_id_obs_path, encoding='utf-8-sig') as rf:
                json_obj = json.load(rf)

            protocol_info = json_obj['protocol_info']
            data_items = json_obj['data_items']

        sim_times = protocol_info['sim_times']
        pre_times = protocol_info['pre_times']
        #experiment_colors = protocol_info['experiment_colors']
        params_to_change_dict = protocol_info['params_to_change']

        num_experiments = len(sim_times)
        if num_experiments != len(pre_times):
            raise ValueError('pre_times and sim_times must be the same length')
            
        max_times = [np.sum(sim_times[II]) for II in range(num_experiments)]

        ######################################
        # AND HERE (as well as the plots at the end)

        t_list = []
        res_list = []
                    
        print('Running experiments for protocol_info: ', protocol_info)
        print('dt: ', self.dt)
        print('maximum time step: ', self.MaximumStep)

        # TODO I should parallelize this so each experiment runs on a different core.
        for exp_idx in range(num_experiments):
            current_time = 0
            for idx, sim_time  in enumerate(sim_times[exp_idx]):
                if idx == 0:
                    self.sim_helper.update_times(self.dt, current_time, sim_time, pre_times[exp_idx])
                    current_time += pre_times[exp_idx]
                else:
                    self.sim_helper.update_times(self.dt, current_time, sim_time, pre_time=0)
                # change parameters
                self.sim_helper.set_param_vals(list(params_to_change_dict.keys()), 
                                        [list(params_to_change_dict.values())[II][exp_idx][idx] for \
                                            II in range(len(params_to_change_dict.keys()))])
                self.sim_helper.run()
                current_time += sim_time
                if idx == 0:
                    t_vec = self.sim_helper.tSim
                    res_vec = self.sim_helper.get_all_results(flatten=True)
                    for var_idx in range(len(res_vec)):
                        # if result is a constant, multiply it by ones for the subexp
                        if not hasattr(res_vec[var_idx], '__len__'):
                            res_vec[var_idx] = np.ones_like(self.sim_helper.tSim) * res_vec[var_idx]
                else:
                    t_vec = np.concatenate((t_vec, self.sim_helper.tSim[1:]))
                    all_results = self.sim_helper.get_all_results(flatten=True)
                    for var_idx in range(len(res_vec)):
                        # if result is a constant, multiply it by ones for the subexp
                        if not hasattr(all_results[var_idx], '__len__'):
                            res_vec[var_idx] = np.concatenate((res_vec[var_idx], 
                                                               np.ones_like(self.sim_helper.tSim[1:]) * all_results[var_idx]))
                        else:
                            res_vec[var_idx] = np.concatenate((res_vec[var_idx], 
                                                            all_results[var_idx][1:]))

                print(f"Experiment {exp_idx}, sub-experiment {idx} completed.")

            # get results
            t_vec = t_vec - pre_times[exp_idx]
            t_list.append(t_vec)
            res_list.append(res_vec)

            self.sim_helper.reset_and_clear()
            print(f"Experiment {exp_idx} completed. experiments remaining : {num_experiments - exp_idx - 1} ")
            
        return t_list, res_list, sim_times

        
# DEPRECATED: This function is kept for backward compatibility, but it is recommended to use the ProtocolRunner class instead.
def run_protocols(model_path, variables_to_plot, protocol_info=None, inp_data_dict=None):
    """ runs the protocols defigned by protocol infor or if not defined, gets 
    protocol info from the obs_data.json file.

    Args:
        variables_to_plot (list[string]): list of variables to plot
        protocol_info (dict, optional): _description_. includes all information for simulating
            the protocols.
        inp_data_dict (dict, optional): _description_. If you don't want to use the circulatory_autogen user_inputs
            dict, you can define your own.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if inp_data_dict is None:
        import yaml
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
        if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
            if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                    inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            else:
                print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                        "you want to use the default user_inputs.yaml location")
                exit()
    
    dt = inp_data_dict['dt']

    if protocol_info is None:
        param_id_obs_path = inp_data_dict['param_id_obs_path']
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)

        protocol_info = json_obj['protocol_info']

    sim_times = protocol_info['sim_times']
    pre_times = protocol_info['pre_times']
    params_to_change_dict = protocol_info['params_to_change']


    num_experiments = len(sim_times)
    if num_experiments != len(pre_times):
        raise ValueError('pre_times and sim_times must be the same length')
        
    max_times = [np.sum(sim_times[II]) for II in range(num_experiments)]

    ######################################
    # AND HERE (as well as the plots at the end)

    t_list = []
    res_list = []

    for exp_idx in range(num_experiments):
        current_time = 0
        for idx, sim_time  in enumerate(sim_times[exp_idx]):
            if idx == 0:
                # TODO update this with the class in CA_user_FA_MG
                sim_helper = SimulationHelper(model_path, dt, sim_time, solver_info={'MaximumNumberOfSteps':1000, 'MaximumStep':0.0001}, 
                                              pre_time=pre_times[exp_idx])
                current_time += pre_times[exp_idx]
            else:
                sim_helper.update_times(dt, current_time, sim_time, pre_time=0)
            # change parameters
            sim_helper.set_param_vals(list(params_to_change_dict.keys()), 
                                    [list(params_to_change_dict.values())[II][exp_idx][idx] for \
                                        II in range(len(params_to_change_dict.keys()))])
            sim_helper.run()
            current_time += sim_time
            if idx == 0:
                t_vec = sim_helper.tSim
                res_vec = sim_helper.get_results(variables_to_plot, flatten=True)
            else:
                t_vec = np.concatenate((t_vec, sim_helper.tSim[1:]))
                for var_idx in range(len(variables_to_plot)):
                    res_vec[var_idx] = np.concatenate((res_vec[var_idx], 
                                                    sim_helper.get_results(variables_to_plot, 
                                                                            flatten=True)[var_idx][1:]))


        # get results
        t_vec = t_vec - pre_times[exp_idx]
        t_list.append(t_vec)
        res_list.append(res_vec)

        sim_helper.reset_and_clear()
    return t_list, res_list
