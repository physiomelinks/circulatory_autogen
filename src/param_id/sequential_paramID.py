import numpy as np
from param_id.paramID import CVS0DParamID
from mpi4py import MPI
import os
import csv
import re

class SequentialParamID:
    """
    This class contains a param_id object that can be run multiple times
    to reduce the parameter set to ensure identifiability.
    """

    def __init__(self, model_path, model_type, param_id_method, file_name_prefix,
                 params_for_id_path=None, num_calls_to_function=1000,
                 param_id_obs_path=None, sim_time=2.0, pre_time=20.0, 
                 solver_info=None, 
                 dt=0.01, mcmc_options=None, ga_options=None,
                 param_id_output_dir=None, resources_dir=None,
                 DEBUG=False):

        self.model_path = model_path
        self.model_type = model_type
        self.param_id_method = param_id_method
        self.file_name_prefix = file_name_prefix
        self.params_for_id_path = params_for_id_path
        self.num_calls_to_function = num_calls_to_function
        self.param_id_obs_path = param_id_obs_path
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.solver_info = solver_info
        self.dt = dt
        self.DEBUG =DEBUG
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        self.param_id_obs_file_prefix = re.sub('\.json', '', os.path.split(param_id_obs_path)[1])
        case_type = f'{param_id_method}_{file_name_prefix}_{self.param_id_obs_file_prefix}'
        if self.rank == 0:
            if param_id_output_dir is None:
                self.param_id_output_dir = os.path.join(os.path.dirname(__file__), '../../param_id_output')
            else:
                self.param_id_output_dir = param_id_output_dir
            
            if not os.path.exists(self.param_id_output_dir):
                os.mkdir(self.param_id_output_dir)
            self.output_dir = os.path.join(self.param_id_output_dir, f'{case_type}')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            self.plot_dir = os.path.join(self.output_dir, 'plots_param_id')
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
        
        if resources_dir is None:
            self.resources_dir = os.path.join(os.path.dirname(__file__), '../../resources')
        else:
            self.resources_dir = resources_dir


        self.param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_name_prefix,
                                params_for_id_path=params_for_id_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time,
                                solver_info=self.solver_info, dt=dt, ga_options=ga_options, DEBUG=DEBUG,
                                param_id_output_dir=self.param_id_output_dir, resources_dir=resources_dir, one_rank=True)


        self.param_id.set_genetic_algorithm_parameters(num_calls_to_function)
        self.best_param_vals = None
        self.best_param_names = None

        self.mcmc_options = mcmc_options

        # thresholds for identifiability TODO optimise these
        self.threshold_param_importance = 0.1
        self.keep_threshold_param_importance = 0.8
        self.threshold_collinearity = 20
        self.threshold_collinearity_pairs = 10
        self.second_deriv_threshold = -1000

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

    def run(self):
        # TODO delete the below

        #### TODO to here

        buf = np.array([False])
        identifiable = buf[0]
        num_params_to_remove_buf = np.array([0])
        param_names_to_remove_all_iterations = []
        # delete the params_to_remove file
        if self.rank == 0:
            if os.path.exists(os.path.join(self.param_id.output_dir, 'param_names_to_remove.csv')):
                os.remove(os.path.join(self.param_id.output_dir, 'param_names_to_remove.csv'))
        # create dictionary with original param idxs for each name
        # only use first name of each list of names that relates to one parameter.
        self.param_id_info["param_names"] = self.param_id.get_param_names()
        param_name_orig_idx_dict = {name[0]: II for II, name in enumerate(self.param_id_info["param_names"])}
        while not identifiable:

            # self.param_id.temp_test()
            # self.param_id.temp_test2()

            self.param_id.run()
            if self.rank == 0:
                self.param_id.run_single_sensitivity(None)

                self.best_param_vals = self.param_id.get_best_param_vals()
                self.param_id_info["param_names"] = self.param_id.get_param_names()

                param_importance = self.param_id.get_param_importance()
                # collinearity_idx = self.param_id.get_collinearity_idx()
                collinearity_idx_pairs = self.param_id.get_collinearity_idx_pairs()
                pred_param_importance = self.param_id.get_pred_param_importance()
                # collinearity_idx = self.param_id.get_collinearity_idx()
                pred_collinearity_idx_pairs = self.param_id.get_pred_collinearity_idx_pairs()

                if np.min(param_importance) > self.threshold_param_importance and \
                            np.max(collinearity_idx_pairs) < self.threshold_collinearity_pairs:
                    print(f'The model is structurally identifiable with {len(self.param_id_info["param_names"])} parameters:')
                    print(self.param_id_info["param_names"])
                    identifiable = True
                else:
                    # remove parameters that aren't identifiable
                    # and update param_id object
                    print(f'The model is NOT structurally identifiable with {len(self.param_id_info["param_names"])} parameters')
                    print(f'determining which parameters to remove from identifying set')
                    param_idxs_to_remove = []
                    for II in range(len(self.param_id_info["param_names"])):
                        param_name = self.param_id_info["param_names"][II]
                        if param_importance[II] < self.threshold_param_importance:
                            if pred_param_importance is not None:
                                print(f'parameter importance of {param_name} is below threshold, '
                                    f'checking param importance with respect to predictions.')
                                if pred_param_importance[II] < self.threshold_param_importance:
                                    print(f'{param_name} pred param importance is {pred_param_importance[II]}, '
                                          f'which is below threshold of {self.threshold_param_importance}'
                                          f', so param will be removed.')
                                    param_idxs_to_remove.append(II)
                                    param_names_to_remove_all_iterations.append(param_name)
                                    identifiable = False
                                else:
                                    print(f'{param_name} pred param importance is {pred_param_importance[II]}, '
                                          f'which is above threshold of {self.threshold_param_importance}'
                                          f', so param will not be removed.')
                                    # set param_importance for this parameter to a high value
                                    param_importance[II] = 999
                                    identifiable = False
                            else:
                                print(f'parameter importance of {param_name} is below threshold, removing parameter')
                                param_idxs_to_remove.append(II)
                                param_names_to_remove_all_iterations.append(param_name)
                                identifiable = False
                        else:
                            for JJ in range(len(self.param_id_info["param_names"])):
                                if collinearity_idx_pairs[II, JJ] > self.threshold_collinearity_pairs:
                                    if param_importance[II] < param_importance[JJ]:
                                        if pred_param_importance is not None:
                                            print(f'collinearity of {self.param_id_info["param_names"][II]}, {self.param_id_info["param_names"][JJ]} '
                                              f'is above threshold, '
                                              f'checking collinearity with respect to predictions.')
                                            if pred_collinearity_idx_pairs[II, JJ] > self.threshold_collinearity_pairs:
                                                print(f'{self.param_id_info["param_names"][II]}, {self.param_id_info["param_names"][JJ]} '
                                                      f'pred collinearity is '
                                                      f'{pred_collinearity_idx_pairs[II, JJ]}, '
                                                      f'which is above threshold of {self.threshold_collinearity_pairs}'
                                                      f', so param can be removed.')
                                                param_idxs_to_remove.append(II)
                                                param_names_to_remove_all_iterations.append(param_name)
                                                identifiable = False
                                                break
                                            else:
                                                print(f'{self.param_id_info["param_names"][II]}, {self.param_id_info["param_names"][JJ]} '
                                                      f'pred collinearity is '
                                                      f'{pred_collinearity_idx_pairs[II, JJ]}, '
                                                      f'which is below threshold of {self.threshold_collinearity_pairs}'
                                                      f', so param will not be removed.')
                                                # set collinearity pair to a small value so it isn't removed
                                                pred_collinearity_idx_pairs[II, JJ] = 0
                                                pred_collinearity_idx_pairs[JJ, II] = 0
                                                identifiable = False
                                        else:
                                            print(f'{self.param_id_info["param_names"][II]}, {self.param_id_info["param_names"][JJ]} '
                                                  f'collinearity is '
                                                  f'{collinearity_idx_pairs[II, JJ]}, '
                                                  f'which is above threshold of {self.threshold_collinearity_pairs}'
                                                  f', so param can be removed.')
                                            param_idxs_to_remove.append(II)
                                            param_names_to_remove_all_iterations.append(param_name)
                                            identifiable = False
                        if identifiable:
                            print('error, not identifiable, but no params to remove added.')
                            exit()

                    if len(param_idxs_to_remove) > 1:
                        # TODO make sure we aren't removing important parameters
                        # it is better to remove too few than too many
                        # for idx in param_idxs_to_remove:
                        #     # TODO this doesn't allow us to remove linearly related params if they are both important
                        #     #  Fix this!
                        #     if param_importance[idx] > self.keep_threshold_param_importance:
                        #         param_idxs_to_remove.remove(idx)
                        pass

                    # TODO future work: if we are reformulating the equations we will need to create and run a
                    #  CVS0DCellMLGenerator object.

            buf[0] = identifiable
            self.comm.Bcast(buf, root=0)
            identifiable = buf[0]

            if not identifiable:
                if self.rank == 0:
                    num_params_to_remove_buf[0] = len(param_idxs_to_remove)
                self.comm.Bcast(num_params_to_remove_buf, root=0)
                if self.rank == 0:
                    param_idxs_to_remove_array = np.array(param_idxs_to_remove)
                else:
                    param_idxs_to_remove_array = np.empty(num_params_to_remove_buf[0], dtype=int)

                self.comm.Bcast(param_idxs_to_remove_array, root=0)
                if self.rank == 0:
                    print('removing the following parameter idxs:')
                    print(param_idxs_to_remove_array)
                self.param_id.remove_params_by_idx(param_idxs_to_remove_array)

                # save param_names to remove
                if self.rank == 0:
                    with open(os.path.join(self.param_id.output_dir, 'param_names_to_remove.csv'), 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows(param_names_to_remove_all_iterations)
                if self.rank == 0:
                    print('These params have been removed from the original parameter set')
                    print(param_names_to_remove_all_iterations)

        # get the original idxs of removed params
        param_idxs_to_remove_all_iterations = [param_name_orig_idx_dict[name[0]] for name in
                                               param_names_to_remove_all_iterations]

        # communicate the original idxs of removed params so we can remove them on all procs
        if self.rank == 0:
            num_params_to_remove_buf[0] = len(param_idxs_to_remove_all_iterations)
        self.comm.Bcast(num_params_to_remove_buf, root=0)
        if self.rank == 0:
            param_idxs_to_remove_array = np.array(param_idxs_to_remove_all_iterations)
        else:
            param_idxs_to_remove_array = np.empty(num_params_to_remove_buf[0], dtype=int)
        self.comm.Bcast(param_idxs_to_remove_array, root=0)

        self.best_param_vals = self.param_id.get_best_param_vals()
        self.param_id.close_simulation()
        
        mcmc = CVS0DParamID(self.model_path, self.model_type, self.param_id_method, True,
                            self.file_name_prefix,
                            params_for_id_path=self.params_for_id_path,
                            param_id_obs_path=self.param_id_obs_path,
                            sim_time=self.sim_time, pre_time=self.pre_time,
                            solver_info=self.solver_info, mcmc_options=self.mcmc_options, dt=self.dt,
                            param_id_output_dir=self.param_id_output_dir, resources_dir=self.resources_dir,
                            DEBUG=self.DEBUG)

        mcmc.remove_params_by_idx(param_idxs_to_remove_array)
        mcmc.set_best_param_vals(self.best_param_vals)
        # save param_names to remove
        if self.rank == 0:
            with open(os.path.join(mcmc.output_dir, 'param_names_to_remove.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_names_to_remove_all_iterations)

        # Now run mcmc to check practical identifiability
        mcmc.run_mcmc()
        self.plot_mcmc_and_predictions(mcmc=mcmc)

        if self.rank == 0:
            print('parameter identification complete. Model parameters are identifiable')

    def plot_mcmc_and_predictions(self, mcmc=None):
        if mcmc == None:
            print('creating mcmc object')
            mcmc = CVS0DParamID(self.model_path, self.model_type, self.param_id_method, True,
                                self.file_name_prefix,
                                params_for_id_path=self.params_for_id_path,
                                param_id_obs_path=self.param_id_obs_path,
                                sim_time=self.sim_time, pre_time=self.pre_time, dt=self.dt,
                                param_id_output_dir=self.param_id_output_dir, resources_dir=self.resources_dir,
                                solver_info=self.solver_info, mcmc_options=self.mcmc_options,
                                DEBUG=self.DEBUG, one_rank=True)
            if self.rank == 0:
                if os.path.exists(os.path.join(mcmc.output_dir, 'param_names_to_remove.csv')):
                    with open(os.path.join(mcmc.output_dir, 'param_names_to_remove.csv'), 'r') as r:
                        param_names_to_remove = []
                        for row in r:
                            name_list = row.split(',')
                            name_list = [name.strip() for name in name_list]
                            param_names_to_remove.append(name_list)
                    mcmc.remove_params_by_name(param_names_to_remove)
        if self.rank != 0:
            return

        if self.best_param_vals is not None:
            self.best_param_vals = np.load(os.path.join(mcmc.output_dir, 'best_param_vals.npy'))

        mcmc.set_best_param_vals(self.best_param_vals)

        print('Plotting mcmc parameter distributions')
        mcmc.plot_mcmc()
        print('Plotting core predictions distribution to check uncertainty on predictions')
        mcmc.postprocess_predictions()
        print('Plotting complete')

    def get_best_param_names(self):
        return self.best_param_names

    def get_best_param_vals(self):
        return self.best_param_vals


