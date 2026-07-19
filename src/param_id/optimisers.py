'''
Optimiser classes for parameter identification.

This module provides a base class for optimisers and implementations for
genetic algorithm, bayesian optimisation, and scipy minimizers.
'''

import numpy as np
from mpi4py import MPI
import math
import os
import csv
import time
import warnings
import traceback
from datetime import date
from abc import ABC, abstractmethod
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
from scipy.stats import qmc
try:
    from skopt import Optimizer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    Optimizer = None

try:
    import nevergrad as ng
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False

# Model types for which OpencorParamID.get_gradient() has an AD backend: a symbolic jacobian
# for casadi models, a tape reverse pass for aadc ones. Everything else falls back to finite
# differences. Keep in sync with OpencorParamID.get_gradient.
AD_GRADIENT_MODEL_TYPES = ('casadi_python', 'aadc_python')


class Optimiser(ABC):
    """
    Base class for all optimisers used in parameter identification.
    
    All optimisers must implement the run() method which performs the optimization
    and sets self.best_param_vals and self.best_cost.
    """
    
    def __init__(self, param_id_obj, param_id_info, param_norm_obj, 
                 num_params, output_dir, optimiser_options=None, DEBUG=False):
        """
        Initialize the optimiser.
        
        Args:
            param_id_obj: The OpencorParamID object that provides get_cost_from_params
            param_id_info: Dictionary with param_names, param_mins, param_maxs
            param_norm_obj: Normalise_class object for parameter normalization
            num_params: Number of parameters to optimize
            output_dir: Directory to save optimization results
            optimiser_options: Dictionary with optimizer-specific options (preferred)
            DEBUG: Debug flag for reduced population sizes in GA
        """
        self.param_id_obj = param_id_obj
        self.param_id_info = param_id_info
        self.param_norm_obj = param_norm_obj
        self.num_params = num_params
        self.output_dir = output_dir
        self.DEBUG = DEBUG
        
        self.optimiser_options = optimiser_options or {}
        
        # These will be set by the run() method
        self.best_param_vals = None
        self.best_cost = np.inf
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        
        # Set default options if not provided
        if 'cost_convergence' not in self.optimiser_options:
            self.optimiser_options['cost_convergence'] = 0.0001
        if 'max_patience' not in self.optimiser_options:
            self.optimiser_options['max_patience'] = 10
    
    @abstractmethod
    def run(self):
        """
        Run the optimization algorithm.
        
        This method should:
        1. Perform the optimization
        2. Set self.best_param_vals to the best parameter values found
        3. Set self.best_cost to the best cost found
        4. Save results to output_dir
        """
        pass
    
    def _save_best_params(self):
        """Helper method to save best parameters and cost."""
        if self.rank == 0:
            np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
            np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)


class GeneticAlgorithmOptimiser(Optimiser):
    """
    Genetic algorithm optimiser for parameter identification.
    
    This is a refactored version of the original genetic algorithm implementation
    in OpencorParamID, maintaining the same functionality.
    """
    
    def run(self):
        """Run the genetic algorithm optimization."""
        comm = self.comm
        rank = self.rank
        num_procs = self.num_procs
        
        if self.DEBUG:
            num_elite = 4
            num_survivors = 6
            num_mutations_per_survivor = 2
            num_cross_breed = 10
        else:
            num_elite = 12
            num_survivors = 48
            num_mutations_per_survivor = 12
            num_cross_breed = 120
        
        num_pop = num_survivors + num_survivors*num_mutations_per_survivor + num_cross_breed
        
        if self.optimiser_options['num_calls_to_function'] < num_pop:
            print(f'Number of calls (n_calls={self.optimiser_options["num_calls_to_function"]}) must be greater than the '
                  f'gen alg population (num_pop={num_pop}), exiting')
            exit()
        if num_procs > num_pop:
            print(f'Number of processors must be less than number of population, exiting')
            exit()
        
        self.max_generations = math.floor(self.optimiser_options['num_calls_to_function']/num_pop)
        
        if rank == 0:
            print(f'Running genetic algorithm with a population size of {num_pop},\n'
                  f'and a maximum number of generations of {self.max_generations}')
        
        simulated_bools = [False]*num_pop
        gen_count = 0
        
        if rank == 0:
            param_vals_norm = np.random.rand(self.num_params, num_pop)
            param_vals = self.param_norm_obj.unnormalise(param_vals_norm)
        else:
            param_vals = None
        
        finished_ga = np.empty(1, dtype=bool)
        finished_ga[0] = False
        cost = np.zeros(num_pop)
        cost[0] = np.inf
        
        last_loss = None
        loss_repeat_counter = 0
        
        while cost[0] > self.optimiser_options["cost_convergence"] and gen_count < self.max_generations and loss_repeat_counter < self.optimiser_options["max_patience"]:
            mutation_weight = 0.1
            gen_count += 1
            
            if rank == 0:
                print('generation num: {}'.format(gen_count))
                # check param_vals are within bounds
                for II in range(self.num_params):
                    for JJ in range(num_pop):
                        if param_vals[II, JJ] < self.param_id_info["param_mins"][II]:
                            param_vals[II, JJ] = self.param_id_info["param_mins"][II]
                        elif param_vals[II, JJ] > self.param_id_info["param_maxs"][II]:
                            param_vals[II, JJ] = self.param_id_info["param_maxs"][II]
                
                send_buf = param_vals.T.copy()
                send_buf_cost = cost
                send_buf_bools = np.array(simulated_bools)
                
                ave, res = divmod(param_vals.shape[1], num_procs)
                pop_per_proc = np.zeros(num_procs, dtype=int)
                for II in range(num_procs):
                    if II < res:
                        pop_per_proc[II] = ave + 1
                    else:
                        pop_per_proc[II] = ave
            else:
                pop_per_proc = np.empty(num_procs, dtype=int)
                send_buf = None
                send_buf_bools = None
                send_buf_cost = None
            
            comm.Bcast(pop_per_proc, root=0)
            recv_buf = np.zeros((pop_per_proc[rank], self.num_params))
            recv_buf_bools = np.empty(pop_per_proc[rank], dtype=bool)
            recv_buf_cost = np.zeros(pop_per_proc[rank])
            
            comm.Scatterv([send_buf, pop_per_proc*self.num_params, None, MPI.DOUBLE],
                          recv_buf, root=0)
            param_vals_proc = recv_buf.T.copy()
            comm.Scatterv([send_buf_bools, pop_per_proc, None, MPI.BOOL],
                          recv_buf_bools, root=0)
            bools_proc = recv_buf_bools
            comm.Scatterv([send_buf_cost, pop_per_proc, None, MPI.DOUBLE],
                          recv_buf_cost, root=0)
            cost_proc = recv_buf_cost
            
            if rank == 0 and gen_count == 1:
                print('population per processor is')
                print(pop_per_proc)
            
            # Each processor runs until all param_val_proc sets have been simulated successfully
            for II in range(pop_per_proc[rank]):
                success = False
                while not success:
                    if bools_proc[II]:
                        success = True
                        break
                    
                    cost_proc[II] = self.param_id_obj.get_cost_from_params(param_vals_proc[:, II])
                    
                    if cost_proc[II] == np.inf:
                        print('... choosing a new random point')
                        param_vals_proc[:, II:II + 1] = self.param_norm_obj.unnormalise(np.random.rand(self.num_params, 1))
                        cost_proc[II] = np.inf
                        success = False
                        break
                    else:
                        bools_proc[II] = True
                    
                    simulated_bools[II] = True
                    success = True
                    if num_procs == 1:
                        if II%5 == 0 and II > num_survivors:
                            print(' this generation is {:.0f}% done'.format(100.0*(II + 1)/pop_per_proc[0]))
                    else:
                        if rank == num_procs - 1:
                            print(' this generation is {:.0f}% done'.format(100.0*(II + 1)/pop_per_proc[0]))
            
            recv_buf = np.zeros((num_pop, self.num_params))
            recv_buf_cost = np.zeros(num_pop)
            send_buf = param_vals_proc.T.copy()
            send_buf_cost = cost_proc
            
            comm.Gatherv(send_buf, [recv_buf, pop_per_proc*self.num_params,
                                     None, MPI.DOUBLE], root=0)
            comm.Gatherv(send_buf_cost, [recv_buf_cost, pop_per_proc,
                                         None, MPI.DOUBLE], root=0)
            
            if rank == 0:
                param_vals = recv_buf.T.copy()
                cost = recv_buf_cost
                
                # order the vertices in order of cost
                order_indices = np.argsort(cost)
                cost = cost[order_indices]
                param_vals = param_vals[:, order_indices]
                print('Cost of first 10 of population : {}'.format(cost[:10]))
                param_vals_norm = self.param_norm_obj.normalise(param_vals)
                print('worst survivor params normed : {}'.format(param_vals_norm[:, num_survivors - 1]))
                print('best params normed : {}'.format(param_vals_norm[:, 0]))
                
                np.save(os.path.join(self.output_dir, 'best_cost'), cost[0])
                np.save(os.path.join(self.output_dir, 'best_param_vals'), param_vals[:, 0])
                
                with open(os.path.join(self.output_dir, 'best_cost_history.csv'), 'a') as file:
                    np.savetxt(file, cost[:10].reshape(1,-1), fmt='%1.9f', delimiter=', ')
                
                with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'a') as file:
                    np.savetxt(file, param_vals_norm[:, 0].reshape(1,-1), fmt='%.5e', delimiter=', ')
                
                #count the repeat number
                if last_loss is not None:
                    if abs(cost[0]-last_loss) < self.optimiser_options["cost_convergence"]:
                        loss_repeat_counter += 1
                    else:
                        loss_repeat_counter = 0
                        last_loss = cost[0]
                else:
                    last_loss = cost[0]
                
                # if cost is small enough then exit
                if cost[0] < self.optimiser_options["cost_convergence"]:
                    print(f'Cost is less than cost_convergence={self.optimiser_options["cost_convergence"]}', 
                            'Exiting calibration with calibration converged to below cost tolerance')
                    finished_ga[0] = True
                elif loss_repeat_counter >= self.optimiser_options["max_patience"]:
                    print(f'loss has been unchanged for max_patience={self.optimiser_options["max_patience"]} generations.',
                            'Exiting calibration with converged optimisation.')
                    finished_ga[0] = True
                else:
                    # At this stage all of the population has been simulated
                    simulated_bools = [True]*num_pop
                    # keep the num_survivors best param_vals, replace these with mutations
                    param_idx = num_elite
                    
                    # set the cases with nan cost to have a very large but not nan cost
                    for idx in range(num_pop):
                        if np.isnan(cost[idx]):
                            cost[idx] = 1e25
                        if cost[idx] > 1e25:
                            cost[idx] = 1e25
                    
                    survive_prob = cost[num_elite:num_pop]**-1/sum(cost[num_elite:num_pop]**-1)
                    rand_survivor_idxs = np.random.choice(np.arange(num_elite, num_pop),
                                                        size=num_survivors-num_elite, p=survive_prob)
                    param_vals_norm[:, num_elite:num_survivors] = param_vals_norm[:, rand_survivor_idxs]
                    
                    param_idx = num_survivors
                    
                    for survivor_idx in range(num_survivors):
                        for JJ in range(num_mutations_per_survivor):
                            simulated_bools[param_idx] = False
                            fifty_fifty = np.random.rand()
                            if fifty_fifty < 0.5:
                                param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx]* \
                                                            (1.0 + mutation_weight*np.random.randn(self.num_params))
                            else:
                                param_vals_norm[:, param_idx] = param_vals_norm[:, survivor_idx] + \
                                                            mutation_weight*np.random.randn(self.num_params)
                            param_idx += 1
                    
                    # now do cross breeding
                    cross_breed_indices = np.random.randint(0, num_survivors, (num_cross_breed, 2))
                    for couple in cross_breed_indices:
                        if couple[0] == couple[1]:
                            couple[1] += 1
                        simulated_bools[param_idx] = False
                        
                        fifty_fifty = np.random.rand()
                        if fifty_fifty < 0.5:
                            param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                        param_vals_norm[:, couple[1]])/2* \
                                                        (1 + mutation_weight*np.random.randn(self.num_params))
                        else:
                            param_vals_norm[:, param_idx] = (param_vals_norm[:, couple[0]] +
                                                            param_vals_norm[:, couple[1]])/2 + \
                                                            mutation_weight*np.random.randn(self.num_params)
                        param_idx += 1
                    
                    param_vals = self.param_norm_obj.unnormalise(param_vals_norm)
            
            comm.Bcast(finished_ga, root=0)
            if finished_ga[0]:
                break
        
        if rank == 0:
            self.best_cost = cost[0]
            best_cost_in_array = np.array([self.best_cost])
            self.best_param_vals = param_vals[:, 0]
        else:
            best_cost_in_array = np.empty(1, dtype=float)
            self.best_param_vals = np.empty(self.num_params, dtype=float)
        
        comm.Bcast(best_cost_in_array, root=0)
        self.best_cost = best_cost_in_array[0]
        comm.Bcast(self.best_param_vals, root=0)
        
        self._save_best_params()


class BayesianOptimiser(Optimiser):
    """
    Bayesian optimisation using scikit-optimize.
    
    This is a refactored version of the original bayesian implementation
    in OpencorParamID, maintaining the same functionality.
    """
    
    def __init__(self, param_id_obj, param_id_info, param_norm_obj, 
                 num_params, output_dir, optimiser_options=None, DEBUG=False,
                 acq_func='EI', n_initial_points=5, random_state=1234, acq_func_kwargs=None):
        """
        Initialize the Bayesian optimiser.
        
        Args:
            acq_func: Acquisition function ('EI', 'PI', 'LCB', or 'gp_hedge')
            n_initial_points: Number of random initialization points
            random_state: Random seed
            acq_func_kwargs: Additional kwargs for acquisition function
            DEBUG: Debug flag
        """
        super().__init__(param_id_obj, param_id_info, param_norm_obj, 
                        num_params, output_dir, optimiser_options, DEBUG)
        self.acq_func = acq_func
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.acq_func_kwargs = acq_func_kwargs or {}
    
    def run(self):
        """Run the Bayesian optimization."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize (skopt) is required for Bayesian optimiser. Install it with: pip install scikit-optimize")
        
        comm = self.comm
        rank = self.rank
        num_procs = self.num_procs
        
        print('WARNING bayesian will be deprecated and is untested')
        if rank == 0:
            print('Running bayesian optimisation')
        
        param_ranges = [a for a in zip(self.param_id_info["param_mins"], self.param_id_info["param_maxs"])]
        
        if rank == 0:
            opt = Optimizer(param_ranges,
                            base_estimator='GP',
                            acq_func=self.acq_func,
                            n_initial_points=self.n_initial_points,
                            random_state=self.random_state,
                            acq_func_kwargs=self.acq_func_kwargs,
                            n_jobs=num_procs)
        
        call_num = 0
        iter_num = 0
        cost = np.zeros(num_procs)
        
        while call_num < self.optimiser_options['num_calls_to_function']:
            if rank == 0:
                if num_procs > 1:
                    points = opt.ask(n_points=num_procs)
                    points_np = np.array(points)
                else:
                    points = opt.ask()
            else:
                points_np = np.zeros((num_procs, self.num_params))
            
            if num_procs > 1:
                comm.Bcast(points_np, root=0)
                cost_proc = self.param_id_obj.get_cost_from_params(points_np[rank, :])
                
                recv_buf_cost = np.zeros(num_procs)
                send_buf_cost = cost_proc
                comm.Gatherv(send_buf_cost, [recv_buf_cost, 1,
                                              None, MPI.DOUBLE], root=0)
                cost_np = recv_buf_cost
                cost = cost_np.tolist()
            else:
                cost[0] = self.param_id_obj.get_cost_from_params(points)
            
            if rank == 0:
                if num_procs > 1:
                    opt.tell(points, cost)
                else:
                    opt.tell(points, cost[0])
                
                call_num += num_procs
                iter_num += 1
                
                if iter_num % 10 == 0:
                    print(f'iteration {iter_num}, call_num = {call_num}')
                
                # Save best results
                best_idx = np.argmin(cost) if num_procs > 1 else 0
                if num_procs > 1:
                    best_params = points_np[best_idx, :]
                    best_cost_val = cost[best_idx]
                else:
                    best_params = points
                    best_cost_val = cost[0]
                
                if best_cost_val < self.best_cost:
                    self.best_cost = best_cost_val
                    self.best_param_vals = best_params
                    np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                    np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
                    
                    with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'a') as file:
                        param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals.reshape(-1, 1))
                        np.savetxt(file, param_vals_norm.reshape(1,-1), fmt='%.5e', delimiter=', ')
        
        # Broadcast final results
        if rank == 0:
            best_cost_in_array = np.array([self.best_cost])
        else:
            best_cost_in_array = np.empty(1, dtype=float)
            self.best_param_vals = np.empty(self.num_params, dtype=float)
        
        comm.Bcast(best_cost_in_array, root=0)
        self.best_cost = best_cost_in_array[0]
        comm.Bcast(self.best_param_vals, root=0)
        
        self._save_best_params()


class CMAESOptimiser(Optimiser):
    """
    CMA-ES optimiser for parameter identification using Nevergrad.
    
    This uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm
    from Nevergrad, which supports parallel evaluations.
    """
    
    def __init__(self, param_id_obj, param_id_info, param_norm_obj, 
                 num_params, output_dir, optimiser_options=None, DEBUG=False):
        """
        Initialize the CMA-ES optimiser.
        
        Args:
            optimiser_options: Dictionary with keys:
                - num_calls_to_function: Number of function evaluations (default: 10000)
                - sigma0: Initial standard deviation (CMA-ES specific, optional, default: 0.2 of parameter range)
                - cost_convergence: Convergence tolerance (shared across optimisers)
                - max_patience: Maximum patience for convergence (shared across optimisers)
            DEBUG: Debug flag
            
        Note: 
            - num_workers is determined at runtime from the number of MPI processes
            - Initial parameter values (x0) are automatically loaded from the parameters CSV file
        """
        super().__init__(param_id_obj, param_id_info, param_norm_obj, 
                        num_params, output_dir, optimiser_options, DEBUG)
        
        if not NEVERGRAD_AVAILABLE:
            raise ImportError("Nevergrad is required for CMA-ES optimiser. Install it with: pip install nevergrad")
        
        # Use num_calls_to_function directly (no separate budget option)
        self.budget = self.optimiser_options.get('num_calls_to_function', 10000)
        
        # Number of parallel workers is determined at runtime from num_procs
        # (set in run() method)
        
        # Prepare bounds
        self.param_mins = np.array(self.param_id_info["param_mins"])
        self.param_maxs = np.array(self.param_id_info["param_maxs"])
        
        # Initial parameter values will be loaded from parameters CSV in run() method
        # via param_id_obj.param_init
        
        # Initial standard deviation (sigma0) - CMA-ES specific option
        if 'sigma0' in self.optimiser_options:
            self.sigma0 = self.optimiser_options['sigma0']
        else:
            # Default to 0.2 of the parameter range
            param_ranges = self.param_maxs - self.param_mins
            self.sigma0 = 0.2 * np.mean(param_ranges)
    
    def run(self):
        """Run the CMA-ES optimization."""
        comm = self.comm
        rank = self.rank
        num_procs = self.num_procs
        
        # Number of workers is determined at runtime from num_procs
        num_workers = num_procs
        
        # Get initial parameter values from the parameters CSV file
        # param_init is a list of lists (one list per parameter, which may have multiple names)
        # We need to extract the first value from each list to get a flat array
        if self.param_id_obj.param_init is not None and len(self.param_id_obj.param_init) > 0:
            x0_list = []
            for vals in self.param_id_obj.param_init:
                if isinstance(vals, list) and len(vals) > 0:
                    x0_list.append(vals[0])
                elif not isinstance(vals, list):
                    x0_list.append(vals)
                else:
                    # Empty list - this shouldn't happen, but handle it
                    if rank == 0:
                        print('Warning: Empty parameter value list found, using random initial guess for this parameter')
                    x0_norm = np.random.rand(1)
                    x0_list.append(self.param_norm_obj.unnormalise(x0_norm.reshape(-1, 1)).flatten()[0])
            x0 = np.array(x0_list)
        else:
            # Fallback to random if param_init is not available
            if rank == 0:
                print('Warning: param_init not available, using random initial guess')
            x0_norm = np.random.rand(self.num_params)
            x0 = self.param_norm_obj.unnormalise(x0_norm.reshape(-1, 1)).flatten()
        
        # Check if initial parameter values are within bounds
        # If not, print warning and set to mean of min and max
        if rank == 0:
            param_names = [name_list[0] if isinstance(name_list, list) else name_list 
                          for name_list in self.param_id_info["param_names"]]
            out_of_bounds = []
            for i in range(self.num_params):
                if x0[i] < self.param_mins[i] or x0[i] > self.param_maxs[i]:
                    out_of_bounds.append(i)
                    # Set to mean of min and max
                    x0[i] = 0.5 * (self.param_mins[i] + self.param_maxs[i])
            
            if out_of_bounds:
                print('\n' + '='*80)
                print('WARNING: Initial parameter values from CSV are outside bounds!')
                print('='*80)
                for i in out_of_bounds:
                    param_name = param_names[i] if i < len(param_names) else f'Parameter {i}'
                    print(f'  Parameter: {param_name}')
                    print(f'    Value from CSV: {self.param_id_obj.param_init[i] if self.param_id_obj.param_init else "N/A"}')
                    print(f'    Bounds: [{self.param_mins[i]:.6e}, {self.param_maxs[i]:.6e}]')
                    print(f'    Setting to mean: {x0[i]:.6e}')
                print('='*80 + '\n')
        
        if rank == 0:
            print(f'Running CMA-ES optimization with Nevergrad')
            print(f'  Budget: {self.budget} function evaluations')
            print(f'  Number of workers: {num_workers}')
            print(f'  Initial sigma: {self.sigma0}')
            print(f'  Initial parameters: {x0}')
        
        # Broadcast initial guess to all ranks
        comm.Bcast(x0, root=0)
        
        # Create parametrization with bounds
        # Nevergrad uses Array parametrization with bounds
        parametrization = ng.p.Array(
            init=x0,
            lower=self.param_mins,
            upper=self.param_maxs
        )
        
        # Create CMA-ES optimizer
        if rank == 0:
            # Nevergrad CMA optimizer doesn't accept sigma0 directly
            # Instead, we can set it via the parametrization or use the optimizer's default
            optimizer = ng.optimizers.CMA(
                parametrization=parametrization,
                budget=self.budget,
                num_workers=num_workers
            )
            # Set initial sigma if provided (some versions of nevergrad support this)
            # For now, we'll use the default sigma and let the optimizer adapt
        
        # Track best results
        best_cost = np.inf
        best_params = None
        iteration = 0
        last_improve_iter = 0
        max_patience = self.optimiser_options.get('max_patience', 10)
        
        # Main optimization loop
        while True:
            if rank == 0:
                # Ask for candidate solutions
                candidates = []
                for _ in range(min(num_workers, self.budget - iteration)):
                    try:
                        candidate = optimizer.ask()
                        candidates.append(candidate)
                    except StopIteration:
                        break
                
                # Convert to numpy array for broadcasting
                num_candidates = len(candidates)
                if num_candidates > 0:
                    candidate_array = np.array([c.value for c in candidates])
                else:
                    # Ensure all ranks receive zero candidates and exit cleanly
                    candidate_array = None
            else:
                candidate_array = None
                num_candidates = None
            
            # Broadcast number of candidates
            num_candidates_buf = np.array([0], dtype=int)
            if rank == 0:
                num_candidates_buf[0] = num_candidates
            comm.Bcast(num_candidates_buf, root=0)
            num_candidates = num_candidates_buf[0]
            
            stop_loop = False
            if num_candidates == 0:
                # Nothing to evaluate; terminate loop for all ranks
                stop_loop = True
                continue_flag = np.array([0], dtype='i')
                comm.Bcast(continue_flag, root=0)
                break
            
            # Broadcast candidates to all ranks
            if rank != 0:
                candidate_array = np.zeros((num_candidates, self.num_params))
            comm.Bcast(candidate_array, root=0)
            
            # Evaluate candidates in parallel
            # Each processor evaluates its assigned candidates
            costs_local = np.full(num_candidates, np.inf)  # Initialize with inf
            local_eval_count = len([i for i in range(num_candidates) if i % num_procs == rank])
            eval_counts = comm.gather(local_eval_count, root=0)
            if rank == 0:
                print(f'[CMA-ES] Evaluating {num_candidates} candidates across {num_procs} rank(s); per-rank load={eval_counts}')
            for i in range(num_candidates):
                if i % num_procs == rank:
                    cost = self.param_id_obj.get_cost_from_params(candidate_array[i, :])
                    costs_local[i] = cost
            
            # Gather all costs using Allreduce with MIN to combine results
            # (since unassigned candidates will have inf)
            all_costs = np.zeros(num_candidates)
            comm.Allreduce(costs_local, all_costs, op=MPI.MIN)
            
            # Tell optimizer the results (only rank 0)
            if rank == 0:
                for candidate, cost in zip(candidates, all_costs):
                    optimizer.tell(candidate, cost)
                    
                    # Track best result
                    if cost < best_cost:
                        best_cost = cost
                        best_params = candidate.value
                        last_improve_iter = iteration
                        
                        # Save intermediate results
                        self.best_cost = best_cost
                        self.best_param_vals = np.array(best_params)
                        np.save(os.path.join(self.output_dir, 'best_cost'), self.best_cost)
                        np.save(os.path.join(self.output_dir, 'best_param_vals'), self.best_param_vals)
                        
                        # Save to history
                        with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'a') as file:
                            param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals.reshape(-1, 1))
                            np.savetxt(file, param_vals_norm.reshape(1,-1), fmt='%.5e', delimiter=', ')
                
                iteration += num_candidates
                
                if iteration % 10 == 0:
                    print(f'Iteration {iteration}/{self.budget}, best cost: {best_cost:.6e}')
                
                # Check convergence / stopping on rank 0 only
                if best_cost < self.optimiser_options.get('cost_convergence', 1e-6):
                    print(f'Cost converged to {best_cost:.6e} (below tolerance {self.optimiser_options.get("cost_convergence", 1e-6)})')
                    stop_loop = True
                if (iteration - last_improve_iter) >= max_patience:
                    print(f'Stopping CMA-ES: no improvement for {max_patience} iterations (best_cost={best_cost:.6e})')
                    stop_loop = True
                if iteration >= self.budget:
                    stop_loop = True
            
            # Broadcast whether to continue
            continue_flag = np.array([1], dtype='i')
            if rank == 0:
                if stop_loop or iteration >= self.budget or best_cost < self.optimiser_options.get('cost_convergence', 1e-6) or (iteration - last_improve_iter) >= max_patience:
                    continue_flag[0] = 0
            comm.Bcast(continue_flag, root=0)
            
            if continue_flag[0] == 0:
                break
        
        # Get final recommendation (rank 0)
        if rank == 0:
            try:
                recommendation = optimizer.provide_recommendation()
                final_params = recommendation.value
                final_cost = self.param_id_obj.get_cost_from_params(final_params)
                
                # Use recommendation if it's better
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_params = final_params
                
                self.best_param_vals = np.array(best_params)
                self.best_cost = best_cost
                
                print(f'CMA-ES optimization completed:')
                print(f'  Final cost: {self.best_cost:.6e}')
                print(f'  Total iterations: {iteration}')
                
                self._save_best_params()
            except Exception as e:
                print(f'Error getting final recommendation: {e}')
                if best_params is not None:
                    self.best_param_vals = np.array(best_params)
                    self.best_cost = best_cost
                    self._save_best_params()
        else:
            self.best_param_vals = np.empty(self.num_params, dtype=float)
            self.best_cost = np.inf
        
        # Broadcast final results to all ranks
        comm.Bcast(self.best_param_vals, root=0)
        best_cost_array = np.array([self.best_cost])
        comm.Bcast(best_cost_array, root=0)
        self.best_cost = best_cost_array[0]

class SciPyMinimizeOptimiser(Optimiser):
    """
    SciPy minimize optimiser for parameter identification.
    
    This uses the gradients of the cost function for gradient-based optimisation.
    """

    def __init__(self, param_id_obj, param_id_info, param_norm_obj, 
                 num_params, output_dir, optimiser_options=None, 
                 do_ad=True, DEBUG=False):
        """
        Initialize the SciPy minimize optimiser.
        """
        super().__init__(param_id_obj, param_id_info, param_norm_obj, 
                        num_params, output_dir, optimiser_options, DEBUG)
        
        self.do_ad = do_ad

        # Prepare bounds
        self.param_mins = self.param_id_info["param_mins"]
        self.param_maxs = self.param_id_info["param_maxs"]
        self.param_ranges = self.param_maxs - self.param_mins
    
    def run(self):
        """Run the SciPy Minimize optimization."""

        comm = self.comm
        rank = self.rank

        best_param_vals = np.empty(self.num_params, dtype=float)
        best_gradient_vals = np.empty(self.num_params, dtype=float)
        best_cost_array = np.empty(1, dtype=float)
        init_gradient = np.empty(self.num_params, dtype=float)

        error_flag = False
        exc_info = None
        
        if rank == 0:
            try:
                init_param_vals = np.asarray(self.param_id_obj.param_init)

                param_mins_norm = self.param_norm_obj.normalise(self.param_mins)
                param_maxs_norm = self.param_norm_obj.normalise(self.param_maxs)
                param_ranges_norm = list(zip(param_mins_norm, param_maxs_norm))

                cost_fun = lambda p: float(self.param_id_obj.get_cost(self.param_norm_obj.unnormalise(p)))
                
                init_cost = self.param_id_obj.get_cost(init_param_vals)
                print(f'Cost before gradient-based optimisation: {init_cost}')
                init_gradient = self.param_id_obj.get_gradient(init_param_vals)

                if self.DEBUG:
                    print('[sp_minimize] initial parameters and gradient:')
                    for i, names in enumerate(self.param_id_info["param_names"]):
                        label = names[0] if isinstance(names, (list, tuple)) else str(names)
                        print(f'    {label:<30} p={init_param_vals[i]:.6g}  dJ/dp={init_gradient[i]:.6e}')

                # Cache the (cost, gradient) at the most recently visited point. L-BFGS-B
                # evaluates the objective and its gradient at the same parameter vector, and
                # get_cost_and_gradient returns both from ONE solve (a single augmented CVODES
                # solve on the Myokit FSA path). So the objective, the jacobian and the progress
                # callback all reuse that one solve instead of each triggering its own -- on the
                # FSA path this removes the separate cost solve per point and the callback's
                # extra cost/gradient solves per iteration.
                ad_eval_cache = {"key": None, "cost": None, "grad_real": None}

                def _ad_cost_and_grad(q):
                    q = np.asarray(q, dtype=float)
                    key = q.tobytes()
                    if ad_eval_cache["key"] != key:
                        p = self.param_norm_obj.unnormalise(q)
                        # Prefer the one-solve (cost, grad) path (e.g. Myokit FSA) when the
                        # param-id object provides it; otherwise fall back to separate calls so
                        # any object exposing get_cost/get_gradient still works.
                        combined = getattr(self.param_id_obj, "get_cost_and_gradient", None)
                        if combined is not None:
                            cost, grad_real = combined(p)
                        else:
                            cost = self.param_id_obj.get_cost(p)
                            grad_real = self.param_id_obj.get_gradient(p)
                        ad_eval_cache["key"] = key
                        ad_eval_cache["cost"] = float(cost)
                        ad_eval_cache["grad_real"] = np.asarray(grad_real, dtype=float).flatten()
                    return ad_eval_cache["cost"], ad_eval_cache["grad_real"]

                if (self.do_ad):
                    min_cost_fun = lambda q: _ad_cost_and_grad(q)[0]
                    gradient_func = lambda q: _ad_cost_and_grad(q)[1] * self.param_ranges
                else:
                    min_cost_fun = cost_fun
                    gradient_func = lambda q: approx_fprime(q, cost_fun, epsilon=1e-4)

                cost_history_path = os.path.join(self.output_dir, 'best_cost_history.csv')
                param_history_path = os.path.join(self.output_dir, 'best_param_vals_history.csv')

                def _append_history(cost_val, x_norm):
                    # Append one row per L-BFGS-B iteration so the cost / parameter
                    # progress plots update live during the run (same CSV format the
                    # population-based optimisers use). x_norm is already in
                    # normalised parameter space, matching best_param_vals_history.
                    with open(cost_history_path, 'a') as file:
                        np.savetxt(file, np.array([[float(cost_val)]]), fmt='%1.9f', delimiter=', ')
                    with open(param_history_path, 'a') as file:
                        np.savetxt(file, np.asarray(x_norm, dtype=float).reshape(1, -1),
                                   fmt='%.5e', delimiter=', ')

                # Record the starting point so the progress curve begins at the
                # pre-optimisation cost.
                _append_history(init_cost, self.param_norm_obj.normalise(init_param_vals))

                step_counter = [0]
                last_iterate = {"x_norm": None, "cost": None}

                def lbfgsb_callback(x_norm):
                    step_counter[0] += 1
                    x_norm = np.asarray(x_norm, dtype=float).copy()
                    last_iterate["x_norm"] = x_norm
                    param_vals = self.param_norm_obj.unnormalise(x_norm)
                    if self.do_ad:
                        # Reuse the solve L-BFGS-B already did at this iterate (cache hit) rather
                        # than recomputing the cost -- and, under DEBUG, the gradient too.
                        cost_val = float(_ad_cost_and_grad(x_norm)[0])
                    else:
                        cost_val = float(self.param_id_obj.get_cost(param_vals))
                    last_iterate["cost"] = cost_val
                    # Live progress: one history row per accepted iteration.
                    _append_history(cost_val, x_norm)
                    if self.DEBUG:
                        print(f'[sp_minimize] step {step_counter[0]}: cost = {cost_val:.6e}')
                        for i, names in enumerate(self.param_id_info["param_names"]):
                            label = names[0] if isinstance(names, (list, tuple)) else str(names)
                            print(f'    {label:<30} {param_vals[i]:.6g}')
                        if self.do_ad:
                            grad = _ad_cost_and_grad(x_norm)[1]  # cache hit, real-space gradient
                            print(f'    |grad|_inf = {np.max(np.abs(grad)):.6e}')
                    if cost_val <= self.optimiser_options['cost_convergence']:
                        raise StopIteration(f"Cost converged: {cost_val}")

                res = None
                try:
                    res = minimize(min_cost_fun, self.param_norm_obj.normalise(init_param_vals), method='L-BFGS-B',
                            bounds=param_ranges_norm, jac=gradient_func, callback=lbfgsb_callback)
                except StopIteration as e:
                    print(str(e))

                if res is not None:
                    best_param_vals = self.param_norm_obj.unnormalise(res.x)
                    best_gradient_vals = res.jac / self.param_ranges
                    best_cost_array = np.array([res.fun])
                elif last_iterate["x_norm"] is not None:
                    best_param_vals = self.param_norm_obj.unnormalise(last_iterate["x_norm"])
                    best_cost_array = np.array([last_iterate["cost"]])
                    if self.do_ad:
                        best_gradient_vals = np.asarray(
                            self.param_id_obj.get_gradient(best_param_vals), dtype=float
                        ).flatten()
                    else:
                        best_gradient_vals = approx_fprime(
                            last_iterate["x_norm"], cost_fun, epsilon=1e-4
                        ) / self.param_ranges
                else:
                    raise RuntimeError("L-BFGS-B finished without a result or callback iterate")

            except (Exception) as e:
                error_flag = True
                exc_info = {
                    "type": type(e).__name__,
                    "message": str(e)
                }

        # Broadcast error flag to all ranks
        error_flag_buf = np.array([error_flag], dtype=bool)
        comm.Bcast(error_flag_buf, root=0)
        exc_info = comm.bcast(exc_info, root=0)
        
        if error_flag_buf[0]:
            raise RuntimeError(
                f"Exception occurred!\n"
                f"{exc_info['type']}: {exc_info['message']}"
            )

        comm.Bcast(best_param_vals, root=0)
        comm.Bcast(best_gradient_vals, root=0)
        comm.Bcast(best_cost_array, root=0)
        comm.Bcast(init_gradient, root=0)

        self.param_id_obj.set_best_param_vals(best_param_vals)

        self.best_cost = best_cost_array[0]
        self.best_param_vals = best_param_vals
        self.init_gradient = init_gradient
        self.best_gradient = best_gradient_vals

        if rank == 0:
            # The per-iteration callback already streamed the cost / parameter
            # history during the run. Record the final best as the last point, but
            # only when L-BFGS-B refined past the last callback iterate (otherwise
            # it would duplicate the last streamed row).
            if last_iterate["cost"] is None or float(best_cost_array[0]) < float(last_iterate["cost"]):
                with open(os.path.join(self.output_dir, 'best_cost_history.csv'), 'a') as file:
                    np.savetxt(file, np.array([[float(best_cost_array[0])]]), fmt='%1.9f', delimiter=', ')
                with open(os.path.join(self.output_dir, 'best_param_vals_history.csv'), 'a') as file:
                    param_vals_norm = self.param_norm_obj.normalise(self.best_param_vals)
                    np.savetxt(file, np.asarray(param_vals_norm, dtype=float).reshape(1, -1),
                               fmt='%.5e', delimiter=', ')

            self._save_best_params()


class MultiStartSciPyMinimizeOptimiser(Optimiser):
    """
    Multi-start L-BFGS-B optimiser for parameter identification.

    L-BFGS-B on its own only ever finds the minimum of the basin it starts in, so on a
    multi-modal cost surface it is at the mercy of the initial parameter values. This
    optimiser scatters `num_starts` starting points over the bounded parameter space and
    runs a bounded L-BFGS-B descent from each one, keeping the best minimum found. It
    therefore explores globally (like the population-based optimisers) while still
    exploiting the gradient for a fast, accurate descent within each basin.

    The starts are independent, so they are distributed round-robin over the MPI ranks and
    each rank runs its own descents with no collective communication in between. The
    results are gathered once at the end.

    Gradients come from `param_id_obj.get_gradient()`, which has an AD backend for
    `casadi_python` (symbolic jacobian) and `aadc_python` (tape) models. For every other model
    type there is no AD gradient, so it falls back to finite differences on the ordinary
    simulation cost.

    Optimiser options:
        num_starts:         number of L-BFGS-B descents (default 10, or 4 when DEBUG).
        start_sampling:     'sobol' (default), 'latin_hypercube' or 'random'.
        include_init_point: if True (default) the first start is the initial parameter
                            values from the parameters csv, so this can never do worse
                            than a single-start sp_minimize run.
        seed:               seed for the start sampler (default 0), so runs are repeatable.
        cost_convergence:   once any start reaches this cost, every rank stops launching new
                            starts (a non-blocking message tells the others), so no rank keeps
                            grinding through starts after a good-enough solution has been found.
        fd_step:            finite-difference step used when AD is unavailable (default 1e-4).

    Parallelism: the starts are distributed statically round-robin over the MPI ranks with no
    per-start communication, so the speedup approaches the rank count only when there are many
    more starts than ranks (individual descents vary widely in length, and that variance only
    averages out across ranks when each rank runs many of them). With num_starts <= num_procs
    each rank runs one long-or-short descent and the wall-clock is bounded by the slowest single
    one, so there is little to gain. run() records the achieved speedup on rank 0 (self.speedup).
    """

    # Cost returned to L-BFGS-B when a simulation fails. Must be finite, otherwise the
    # finite-difference gradient becomes nan and the descent silently stalls.
    FAILED_SIM_COST = 1e10

    # MPI tag for the global early-stop signal.
    _STOP_TAG = 11711

    def __init__(self, param_id_obj, param_id_info, param_norm_obj,
                 num_params, output_dir, optimiser_options=None,
                 do_ad=True, model_type=None, DEBUG=False):
        super().__init__(param_id_obj, param_id_info, param_norm_obj,
                         num_params, output_dir, optimiser_options, DEBUG)

        self.do_ad = do_ad
        self.model_type = model_type
        # param_id_obj.get_gradient() has an AD backend for casadi (symbolic) and aadc
        # (tape) models, and for cellml_only models run through Myokit CVODES forward
        # sensitivity (advertised via fsa_gradient_available); for anything else it raises,
        # so those fall back to finite differences.
        fsa_available = getattr(param_id_obj, 'fsa_gradient_available', None)
        self.use_ad_gradient = do_ad and (
            model_type in AD_GRADIENT_MODEL_TYPES
            or (callable(fsa_available) and fsa_available())
        )

        self.param_mins = self.param_id_info["param_mins"]
        self.param_maxs = self.param_id_info["param_maxs"]
        self.param_ranges = self.param_maxs - self.param_mins

        if 'num_starts' not in self.optimiser_options:
            self.optimiser_options['num_starts'] = 4 if DEBUG else 10
        if 'start_sampling' not in self.optimiser_options:
            self.optimiser_options['start_sampling'] = 'sobol'
        if 'include_init_point' not in self.optimiser_options:
            self.optimiser_options['include_init_point'] = True
        if 'seed' not in self.optimiser_options:
            self.optimiser_options['seed'] = 0
        if 'fd_step' not in self.optimiser_options:
            self.optimiser_options['fd_step'] = 1e-4
        if 'no_new_starts_on_convergence' not in self.optimiser_options:
            # True (default): once any start reaches cost_convergence, no rank launches new
            # starts. False: run every start to completion regardless -- useful for mapping the
            # basins of a multi-modal problem (how many starts land in each minimum).
            self.optimiser_options['no_new_starts_on_convergence'] = True
        if 'convergence_cluster_tol_frac' not in self.optimiser_options:
            # Two converged starts are the "same" solution if every parameter agrees to within
            # this fraction of that parameter's range (max - min).
            self.optimiser_options['convergence_cluster_tol_frac'] = 0.02

        self.init_gradient = None
        self.best_gradient = None

        # Convergence report, filled in by run() on rank 0.
        self.num_converged = None
        self.convergence_clusters = None

        # Scaling bookkeeping, filled in by run() on rank 0.
        self.num_starts_run = None
        self.starts_run_per_rank = None
        self.serial_seconds = None
        self.wall_seconds = None
        self.speedup = None

    def _generate_starts(self):
        """Starting points in normalised [0, 1] parameter space, shape (num_starts, num_params).

        Deterministic given the seed, so every rank generates the identical set and no
        broadcast is needed to agree on who runs which start.
        """
        num_starts = int(self.optimiser_options['num_starts'])
        if num_starts < 1:
            raise ValueError(f'num_starts must be at least 1, got {num_starts}')
        sampling = self.optimiser_options['start_sampling']
        seed = self.optimiser_options['seed']
        include_init = self.optimiser_options['include_init_point']

        num_sampled = num_starts - 1 if include_init else num_starts

        sampled = np.zeros((0, self.num_params))
        if num_sampled > 0:
            if sampling == 'sobol':
                # Sobol warns for sample counts that aren't a power of two; the balance
                # properties we lose don't matter for scattering a handful of starts.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sampled = qmc.Sobol(d=self.num_params, scramble=True,
                                        seed=seed).random(num_sampled)
            elif sampling == 'latin_hypercube':
                sampled = qmc.LatinHypercube(d=self.num_params, seed=seed).random(num_sampled)
            elif sampling == 'random':
                sampled = np.random.default_rng(seed).random((num_sampled, self.num_params))
            else:
                raise ValueError(f'unknown start_sampling "{sampling}", expected one of '
                                 f'sobol, latin_hypercube or random')

        if include_init:
            init_norm = np.clip(
                np.asarray(self.param_norm_obj.normalise(np.asarray(self.param_id_obj.param_init)),
                           dtype=float).flatten(), 0.0, 1.0)
            starts = np.vstack([init_norm.reshape(1, -1), sampled])
        else:
            starts = sampled

        return starts

    def _make_cost_func(self):
        """Cost as a function of normalised parameters."""
        def cost_fun(q_norm):
            p = self.param_norm_obj.unnormalise(np.asarray(q_norm, dtype=float))
            # get_cost dispatches on model_type: the symbolic cost for casadi models, the
            # ordinary simulation cost otherwise.
            cost = float(self.param_id_obj.get_cost(p))
            # A failed simulation returns inf, which would poison the finite-difference
            # gradient and stall L-BFGS-B.
            return cost if np.isfinite(cost) else self.FAILED_SIM_COST
        return cost_fun

    def _make_gradient_func(self, cost_fun):
        """dJ/dq in normalised parameter space."""
        fd_step = float(self.optimiser_options['fd_step'])

        def fd_gradient(q_norm):
            return approx_fprime(np.asarray(q_norm, dtype=float), cost_fun, epsilon=fd_step)

        if self.use_ad_gradient:
            def gradient_func(q_norm):
                p = self.param_norm_obj.unnormalise(np.asarray(q_norm, dtype=float))
                # get_gradient dispatches on model_type: the CasADi symbolic jacobian, the
                # AADC tape gradient, or the Myokit CVODES FSA gradient. If the AD backend
                # cannot handle this configuration (e.g. a protocol shape it does not support),
                # fall back to finite differences rather than crash the descent -- warn once so
                # a silent switch to the slower/less-exact path is visible.
                try:
                    dJ_dp = np.asarray(self.param_id_obj.get_gradient(p), dtype=float).flatten()
                    if not np.all(np.isfinite(dJ_dp)):
                        raise ValueError("AD gradient returned non-finite values")
                except Exception as e:
                    if not getattr(self, '_ad_gradient_fallback_warned', False):
                        warnings.warn(
                            f"AD gradient unavailable for this configuration ({type(e).__name__}: "
                            f"{e}); falling back to finite differences for the gradient.")
                        self._ad_gradient_fallback_warned = True
                    return fd_gradient(q_norm)
                # chain rule: dJ/dq = dJ/dp * dp/dq = dJ/dp * (max - min)
                return dJ_dp * self.param_ranges
            return gradient_func

        return fd_gradient

    def _run_one_start(self, start_idx, x0_norm, cost_fun, gradient_func, bounds_norm):
        """Run a single bounded L-BFGS-B descent. Returns a result dict (never raises for a
        cost-converged early stop)."""
        iterates = []  # (cost, x_norm) per accepted L-BFGS-B iteration, for the history csv

        start_wall = time.perf_counter()
        init_cost = cost_fun(x0_norm)
        iterates.append((init_cost, np.asarray(x0_norm, dtype=float).copy()))

        last_iterate = {"x_norm": None, "cost": None}

        def callback(x_norm):
            x_norm = np.asarray(x_norm, dtype=float).copy()
            cost_val = cost_fun(x_norm)
            last_iterate["x_norm"] = x_norm
            last_iterate["cost"] = cost_val
            iterates.append((cost_val, x_norm))
            if self.DEBUG:
                print(f'[multi_start_sp_minimize] rank {self.rank} start {start_idx}: '
                      f'cost = {cost_val:.6e}')
            if cost_val <= self.optimiser_options['cost_convergence']:
                raise StopIteration(f'Cost converged: {cost_val}')

        res = None
        try:
            res = minimize(cost_fun, np.asarray(x0_norm, dtype=float), method='L-BFGS-B',
                           bounds=bounds_norm, jac=gradient_func, callback=callback)
        except StopIteration:
            pass

        if res is not None:
            final_x_norm = np.asarray(res.x, dtype=float)
            final_cost = float(res.fun)
        elif last_iterate["x_norm"] is not None:
            final_x_norm = last_iterate["x_norm"]
            final_cost = float(last_iterate["cost"])
        else:
            # L-BFGS-B stopped before the first accepted iteration; the start itself is the
            # best we have from this descent.
            final_x_norm = np.asarray(x0_norm, dtype=float)
            final_cost = float(init_cost)

        return {
            'start_idx': start_idx,
            'init_cost': float(init_cost),
            'final_cost': final_cost,
            'final_x_norm': final_x_norm,
            'num_iterations': max(len(iterates) - 1, 0),
            'iterates': iterates,
            'duration': time.perf_counter() - start_wall,
        }

    def run(self):
        """Run L-BFGS-B from every start, distributed over the MPI ranks."""
        comm = self.comm
        rank = self.rank
        num_procs = self.num_procs

        cost_fun = self._make_cost_func()
        gradient_func = self._make_gradient_func(cost_fun)

        # Non-blocking global early stop. When a start meets the convergence threshold, that rank
        # tells every OTHER rank to stop via a point-to-point message; between starts each rank
        # checks (without ever blocking) whether such a message has arrived. Point-to-point and
        # non-blocking is what keeps this deadlock-free with uneven start counts -- a blocking
        # collective inside the loop would hang the instant two ranks had different numbers of
        # starts left to run.
        stop_tag = self._STOP_TAG
        i_signalled = False
        should_stop = False
        stop_send_requests = []
        num_stop_received = 0

        # Every rank must take part in the same collectives, so any failure is caught
        # locally and only turned into an exception after all ranks have agreed on it.
        local_results = []
        local_error = None
        loop_wall = 0.0
        try:
            starts = self._generate_starts()
            num_starts = starts.shape[0]
            cost_convergence = self.optimiser_options['cost_convergence']
            stop_on_convergence = self.optimiser_options['no_new_starts_on_convergence']

            param_mins_norm = self.param_norm_obj.normalise(self.param_mins)
            param_maxs_norm = self.param_norm_obj.normalise(self.param_maxs)
            bounds_norm = list(zip(param_mins_norm, param_maxs_norm))

            if rank == 0:
                grad_source = (f'{self.model_type} AD' if self.use_ad_gradient
                               else 'finite differences')
                print(f'Running multi-start L-BFGS-B: {num_starts} starts '
                      f'({self.optimiser_options["start_sampling"]} sampling) over '
                      f'{num_procs} rank(s), gradients from {grad_source}')

            # Static round-robin: rank r runs starts r, r+P, r+2P, ... No communication is needed
            # to agree who runs what. With num_starts >> num_procs the per-rank workloads even out
            # even though individual starts vary a lot in length -- which is exactly why parallel
            # multi-start only pays off for many starts (see the docs).
            my_start_indices = [s for s in range(num_starts) if s % num_procs == rank]

            loop_t0 = time.perf_counter()
            for start_idx in my_start_indices:
                # Has any other rank reached the threshold? Drain every pending stop message
                # without blocking. (Only relevant when stop_on_convergence is on; when it is
                # off no rank ever signals, so the probe simply always finds nothing.)
                if stop_on_convergence:
                    while comm.iprobe(source=MPI.ANY_SOURCE, tag=stop_tag):
                        comm.recv(source=MPI.ANY_SOURCE, tag=stop_tag)
                        num_stop_received += 1
                        should_stop = True
                    if should_stop:
                        break

                result = self._run_one_start(start_idx, starts[start_idx, :], cost_fun,
                                             gradient_func, bounds_norm)
                local_results.append(result)

                # This start met the threshold. If early stopping is on, tell every other rank to
                # stop launching new starts (once); isend is non-blocking, drained after the loop.
                # If it is off, keep going -- every start runs, so the basins can be mapped.
                if (stop_on_convergence and not i_signalled
                        and result['final_cost'] <= cost_convergence):
                    for other in range(num_procs):
                        if other != rank:
                            stop_send_requests.append(comm.isend(1, dest=other, tag=stop_tag))
                    i_signalled = True
                    should_stop = True  # stop this rank's own remaining starts too
            loop_wall = time.perf_counter() - loop_t0
        except Exception as e:
            local_error = {"rank": rank, "type": type(e).__name__, "message": str(e),
                           "traceback": traceback.format_exc()}

        # Collective 1: agree on failures, and exchange the stop-signal bookkeeping so the
        # non-blocking messages can be drained without leaking.
        status = comm.allgather({"error": local_error, "signalled": i_signalled,
                                 "loop_wall": loop_wall})

        # Drain the stop machinery. Each signalling rank sent one message to every OTHER rank, so
        # this rank must receive one per other signalling rank; both counts are known here, so
        # neither the receives nor the waits can block. Done even on the error path so that no MPI
        # message or request is left dangling.
        num_signalling_others = sum(1 for i, st in enumerate(status)
                                    if st["signalled"] and i != rank)
        while num_stop_received < num_signalling_others:
            comm.recv(source=MPI.ANY_SOURCE, tag=stop_tag)
            num_stop_received += 1
        if stop_send_requests:
            MPI.Request.Waitall(stop_send_requests)

        errors = [st["error"] for st in status if st["error"] is not None]
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"multi-start L-BFGS-B failed on rank {first['rank']}\n"
                f"{first['type']}: {first['message']}\n{first['traceback']}"
            )

        # Collective 2: gather every start's result on rank 0.
        gathered = comm.gather(local_results, root=0)

        best = None
        # Everything rank 0 does below happens while the other ranks are already blocked in
        # the bcast at the end of this block. If rank 0 raises in here -- no results, a
        # read-only or full output directory in one of the _write_* calls, an empty `status`
        # in the max() -- it leaves the function and every other rank waits on that bcast
        # forever. The job hangs instead of failing, which under a scheduler burns the entire
        # wall-clock allocation with no diagnostic. Capture the failure and broadcast it
        # alongside the result so all ranks raise together, mirroring the all-ranks-agree
        # handling already used for per-start errors above.
        rank0_error = None
        if rank == 0:
            try:
                all_results = sorted([r for rank_results in gathered for r in rank_results],
                                     key=lambda r: r['start_idx'])
                if not all_results:
                    raise RuntimeError('multi-start L-BFGS-B produced no results')

                best_result = min(all_results, key=lambda r: r['final_cost'])
                best_param_vals = self.param_norm_obj.unnormalise(best_result['final_x_norm'])

                self._write_history(all_results)
                self._write_start_summary(all_results)
                self._cluster_converged_starts(all_results)

                # Scaling bookkeeping. The serial cost is the sum of every start's own wall time; the
                # parallel cost is the slowest rank's loop time. Their ratio is the achieved speedup,
                # which approaches num_procs only when there are many more starts than ranks.
                durations = [r['duration'] for r in all_results]
                self.num_starts_run = len(all_results)
                self.starts_run_per_rank = [len(rank_results) for rank_results in gathered]
                self.serial_seconds = float(sum(durations))
                self.wall_seconds = max(st['loop_wall'] for st in status)
                self.speedup = (self.serial_seconds / self.wall_seconds
                                if self.wall_seconds > 0 else float('nan'))

                skipped_note = ('the rest were skipped after a start converged'
                                if self.optimiser_options['no_new_starts_on_convergence']
                                else 'no_new_starts_on_convergence is off, so all ran')
                print(f'Multi-start L-BFGS-B ran {len(all_results)} of '
                      f'{int(self.optimiser_options["num_starts"])} starts '
                      f'({skipped_note}); '
                      f'best cost {best_result["final_cost"]:.6e} came from start '
                      f'{best_result["start_idx"]}; '
                      f'speedup {self.speedup:.1f}x over {num_procs} rank(s)')
                print(f'  {self.num_converged} of {len(all_results)} starts converged '
                      f'(cost <= {self.optimiser_options["cost_convergence"]:g}), landing in '
                      f'{len(self.convergence_clusters)} distinct solution(s):')
                for i, cl in enumerate(self.convergence_clusters):
                    params_str = ' '.join(f'{v:.4g}' for v in cl['params'])
                    print(f'    solution {i}: {cl["count"]} start(s), cost {cl["best_cost"]:.3e}, '
                          f'params [{params_str}]')

                best = {
                    'param_vals': np.asarray(best_param_vals, dtype=float).flatten(),
                    'cost': float(best_result['final_cost']),
                }
            except Exception as exc:
                best = None
                rank0_error = {
                    'type': type(exc).__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                }

        # Collective 3: give every rank the winner -- or rank 0's failure, so that an error
        # here raises on every rank instead of stranding them all in this bcast.
        best, rank0_error = comm.bcast((best, rank0_error), root=0)
        if rank0_error is not None:
            raise RuntimeError(
                'multi-start L-BFGS-B failed while collecting results on rank 0: '
                f"{rank0_error['type']}: {rank0_error['message']}\n{rank0_error['traceback']}"
            )

        self.best_param_vals = best['param_vals']
        self.best_cost = best['cost']
        self.param_id_obj.set_best_param_vals(self.best_param_vals)

        if self.use_ad_gradient:
            self.init_gradient = np.asarray(
                self.param_id_obj.get_gradient(np.asarray(self.param_id_obj.param_init)),
                dtype=float).flatten()
            self.best_gradient = np.asarray(
                self.param_id_obj.get_gradient(self.best_param_vals), dtype=float).flatten()

        self._save_best_params()

    def _write_history(self, all_results):
        """Write the running-best cost/parameter history over the concatenated starts.

        The history csvs are read as a monotonically improving progress curve by the
        plotting code, so we record a row only when a start improves on the best cost seen
        so far, walking the starts in order.
        """
        cost_history_path = os.path.join(self.output_dir, 'best_cost_history.csv')
        param_history_path = os.path.join(self.output_dir, 'best_param_vals_history.csv')

        best_so_far = np.inf
        rows_cost = []
        rows_params = []
        for result in all_results:
            for cost_val, x_norm in result['iterates']:
                if cost_val < best_so_far:
                    best_so_far = cost_val
                    rows_cost.append([float(cost_val)])
                    rows_params.append(np.asarray(x_norm, dtype=float).flatten())

        if not rows_cost:
            return

        with open(cost_history_path, 'a') as file:
            np.savetxt(file, np.array(rows_cost), fmt='%1.9f', delimiter=', ')
        with open(param_history_path, 'a') as file:
            np.savetxt(file, np.array(rows_params), fmt='%.5e', delimiter=', ')

    def _write_start_summary(self, all_results):
        """One row per start, so the basins the starts fell into can be inspected."""
        param_labels = [names[0] if isinstance(names, (list, tuple)) else str(names)
                        for names in self.param_id_info["param_names"]]

        summary_path = os.path.join(self.output_dir, 'multi_start_summary.csv')
        with open(summary_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['start_idx', 'init_cost', 'final_cost', 'num_iterations',
                             'duration_s'] +
                            [label.replace('/', ' ') for label in param_labels])
            for result in all_results:
                param_vals = self.param_norm_obj.unnormalise(result['final_x_norm'])
                writer.writerow(
                    [result['start_idx'], f'{result["init_cost"]:.9e}',
                     f'{result["final_cost"]:.9e}', result['num_iterations'],
                     f'{result.get("duration", float("nan")):.6f}'] +
                    [f'{val:.9e}' for val in np.asarray(param_vals, dtype=float).flatten()])

    def _cluster_converged_starts(self, all_results):
        """Group the converged starts by which solution they reached, and record how many landed
        in each.

        A start has "converged" if its final cost is at or below cost_convergence. Two converged
        starts are the same solution if every parameter agrees to within
        convergence_cluster_tol_frac of that parameter's range (max - min) -- so on a multi-modal
        problem the clusters are the distinct minima found, and their sizes say how the starts
        split between them. Sets self.num_converged and self.convergence_clusters (largest first)
        and writes multi_start_convergence_clusters.csv.
        """
        cost_convergence = self.optimiser_options['cost_convergence']
        tol = self.optimiser_options['convergence_cluster_tol_frac'] * self.param_ranges

        converged = [r for r in all_results if r['final_cost'] <= cost_convergence]
        self.num_converged = len(converged)

        clusters = []  # each: {'params', 'best_cost', 'count', 'start_idxs'}
        for result in converged:
            params = np.asarray(self.param_norm_obj.unnormalise(result['final_x_norm']),
                                dtype=float).flatten()
            for cluster in clusters:
                if np.all(np.abs(params - cluster['params']) <= tol):
                    cluster['count'] += 1
                    cluster['start_idxs'].append(result['start_idx'])
                    if result['final_cost'] < cluster['best_cost']:
                        # keep the lowest-cost member as the cluster's representative point
                        cluster['params'] = params
                        cluster['best_cost'] = result['final_cost']
                    break
            else:
                clusters.append({'params': params, 'best_cost': result['final_cost'],
                                 'count': 1, 'start_idxs': [result['start_idx']]})

        clusters.sort(key=lambda c: c['count'], reverse=True)
        self.convergence_clusters = clusters

        param_labels = [names[0] if isinstance(names, (list, tuple)) else str(names)
                        for names in self.param_id_info["param_names"]]
        path = os.path.join(self.output_dir, 'multi_start_convergence_clusters.csv')
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['solution_idx', 'num_starts', 'best_cost'] +
                            [label.replace('/', ' ') for label in param_labels])
            for i, cluster in enumerate(clusters):
                writer.writerow([i, cluster['count'], f'{cluster["best_cost"]:.9e}'] +
                                [f'{v:.9e}' for v in cluster['params']])
