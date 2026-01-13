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
from datetime import date
from abc import ABC, abstractmethod
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


class Optimiser(ABC):
    """
    Base class for all optimisers used in parameter identification.
    
    All optimisers must implement the run() method which performs the optimization
    and sets self.best_param_vals and self.best_cost.
    """
    
    def __init__(self, param_id_obj, param_id_info, param_norm_obj, 
                 num_params, output_dir, ga_options=None, optimiser_options=None, DEBUG=False):
        """
        Initialize the optimiser.
        
        Args:
            param_id_obj: The OpencorParamID object that provides get_cost_from_params
            param_id_info: Dictionary with param_names, param_mins, param_maxs
            param_norm_obj: Normalise_class object for parameter normalization
            num_params: Number of parameters to optimize
            output_dir: Directory to save optimization results
            ga_options: Dictionary with optimization options (legacy name, kept for backwards compatibility)
            optimiser_options: Dictionary with optimizer-specific options (preferred)
            DEBUG: Debug flag for reduced population sizes in GA
        """
        self.param_id_obj = param_id_obj
        self.param_id_info = param_id_info
        self.param_norm_obj = param_norm_obj
        self.num_params = num_params
        self.output_dir = output_dir
        self.DEBUG = DEBUG
        
        # Merge ga_options into optimiser_options for backwards compatibility
        # optimiser_options takes precedence if both are provided
        self.optimiser_options = optimiser_options or {}
        if ga_options is not None:
            for key, value in ga_options.items():
                if key not in self.optimiser_options:
                    self.optimiser_options[key] = value
        
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
                 num_params, output_dir, ga_options=None, optimiser_options=None, DEBUG=False,
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
                        num_params, output_dir, ga_options, optimiser_options, DEBUG)
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
                 num_params, output_dir, ga_options=None, optimiser_options=None, DEBUG=False):
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
                        num_params, output_dir, ga_options, optimiser_options, DEBUG)
        
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
                    print(f'    Value from CSV: {self.param_id_obj.param_init[i][0] if self.param_id_obj.param_init else "N/A"}')
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

