import opencor as oc
import numpy as np
import os
import sys
from importlib import import_module  # Only used for debugging
# from func_timeout import func_timeout, FunctionTimedOut


class SimulationHelper():
    def __init__(self, cellml_path, dt,
                 sim_time, solver_info=None,
                 pre_time=0.0):

        # TODO comment this out
        # self.resource_module = import_module('psutil')

        self.cellml_path = cellml_path  # path to cellml file
        self.dt = dt  # time step
        self.stop_time = pre_time + sim_time  # full time of simulation
        self.pre_steps = int(pre_time/dt)  # number of steps to do before storing data (used to reach steady state)
        self.n_steps = int(sim_time/dt)  # number of steps for storing data
        self.simulation = oc.open_simulation(cellml_path)
        if not self.simulation.valid():
            print(f'simulation object opened from {cellml_path} is not valid, exiting')
            exit()
        self.data = self.simulation.data()
        if solver_info is None:
            solver_info = {'MaximumNumberOfSteps': 5000, 'MaximumStep': 0.0001}
        for key, value in solver_info.items():
            # ignore high-level/legacy keys that aren't part of OpenCOR solver properties
            if key.lower() == "method":
                continue
            if key.lower() == "solver":
                continue
            if key not in self.data.odeSolverProperties():
                print(f'{key} is not a valid key for CVODE solver properties; valid keys are '
                      f'{list(self.data.odeSolverProperties())}. Skipping.')
                continue
            self.data.set_ode_solver_property(key, value)
        self.data.set_point_interval(self.dt)  # time interval for data storage
        self.data.set_starting_point(0)
        self.data.set_ending_point(self.stop_time)
        self.tSim = np.linspace(pre_time, self.stop_time, self.n_steps + 1)  # time values for stored part of simulation

    # inner psutil function # TODO only needed for memory checking
    def process_memory(self):
        process = self.resource_module.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    def run(self):
        try:
            self.simulation.run()
        except RuntimeError:
            print("Failed to converge")
            print('restarting simulation object')
            self.simulation.reset()
            self.simulation.release_all_values()
            self.simulation.clear_results()
            return False

        return True

    def reset_and_clear(self, only_one_exp=-1):
        self.simulation.reset(True)
        self.simulation.release_all_values()
        self.simulation.clear_results()

    def reset_states(self):
        self.simulation.reset(False)  # True resets everything, False resets only the states

    def get_all_variable_names(self):
        # get all states, algebraics and constants
        variable_names = list(self.simulation.results().states().keys()) + \
            list(self.simulation.results().algebraic().keys()) + \
            list(self.data.constants().keys())
        return variable_names

    def get_all_results(self, flatten=False):
        variable_names = self.get_all_variable_names()
        results = self.get_results(variable_names, flatten=flatten)
        return results

    def get_results(self, variables_list_of_lists, flatten=False):
        # if the input is a list of variables, turn it into a list of lists
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]

        results = []
        for JJ, variables_list in enumerate(variables_list_of_lists):
            results.append([])
            for variable_name in variables_list:
                if variable_name == 'time':
                    results[JJ].append(self.tSim)
                elif variable_name in self.simulation.results().states():
                    results[JJ].append(self.simulation.results().states()[variable_name].values()[-self.n_steps - 1:].copy())
                elif variable_name in self.simulation.results().algebraic():
                    results[JJ].append(self.simulation.results().algebraic()[variable_name].values()[-self.n_steps-1:].copy())
                elif variable_name in self.data.constants():
                    results[JJ].append(self.data.constants()[variable_name])
                else:
                    print(f'variable {variable_name} is not a model variable. model variables are')
                    print([name for name in self.simulation.results().states()])
                    print([name for name in self.simulation.results().algebraic()])
                    print([name for name in self.data.constants()])
                    print('exiting')
                    exit()

        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_init_param_vals(self, param_names):
        param_init = []
        for JJ, param_name_or_list in enumerate(param_names):
            if not isinstance(param_name_or_list, list):
                param_name_or_list = [param_name_or_list]

            param_init.append([])
            for param_name in param_name_or_list:
                if param_name in self.data.states():
                    param_init[JJ].append(self.data.states()[param_name])
                elif param_name in self.data.constants():
                    param_init[JJ].append(self.data.constants()[param_name])
                else:
                    print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                          f'The states are:')
                    print([name for name in self.data.states()])
                    print('the constants are:')
                    print([name for name in self.data.constants()])
                    exit()

        return param_init

    def set_param_vals(self, param_names, param_vals):
        # ensure param_vals stores state values first, then constant values
        for JJ, param_name_or_list in enumerate(param_names):
            if not isinstance(param_name_or_list, list):
                param_name_or_list = [param_name_or_list]

            for param_name in param_name_or_list:
                if param_name in self.data.states():
                    self.data.states()[param_name] = param_vals[JJ]
                elif param_name in self.data.constants():
                    self.data.constants()[param_name] = param_vals[JJ]
                else:
                    print(f'parameter name of {param_name} doesn\'t exist in either constants or states'
                          f'The states are:')
                    print([name for name in self.data.states()])
                    print('the constants are:')
                    print([name for name in self.data.constants()])
                    exit()

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names, absolute=False):

        if absolute:
            new_param_vals = mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a*b for a, b in zip(init_param_vals, mod_factors)]

        print(new_param_vals)
        self.set_param_vals(param_names, new_param_vals)

        success = self.run()
        print('here')
        if success:
            pred_obs_new = self.get_results(obs_names)
            # reset params
            self.reset_and_clear()

        else:
            print('simulation failed ')
            exit()

        return pred_obs_new

    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.stop_time = start_time + pre_time + sim_time  # full time of simulation
        self.pre_steps = int(pre_time/self.dt)  # number of steps to do before storing data (used to reach steady state)
        self.n_steps = int(sim_time/self.dt)  # number of steps for storing data
        self.data.set_starting_point(start_time)
        self.data.set_ending_point(self.stop_time)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)  # time values for stored part of simulation

    def close_simulation(self):
        oc.close_simulation(self.simulation)

