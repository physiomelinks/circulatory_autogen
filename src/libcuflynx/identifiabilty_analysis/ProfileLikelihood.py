import numpy as np  
import matplotlib.pyplot as plt  
import os  
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI  
import json
MPI = _get_MPI()

class ProfileLikelihood:  
    """  
    Profile likelihood analysis for parameter identifiability assessment.  
      
    For each parameter, this method:  
    1. Fixes the parameter at values around its best fit  
    2. Re-optimizes all other parameters  
    3. Records the minimum cost at each fixed value  
    4. Plots cost vs parameter value  
    """  

    def __init__(self, param_id, param_id_info, output_dir,   
                 num_points=50, range_factor=0.2, optimiser_options=None):  
        
        self.param_id = param_id
        self.param_id_info = param_id_info  
        self.output_dir = output_dir  
        self.num_points = num_points  
        self.range_factor = range_factor  
        self.optimiser_options = optimiser_options or {}  
          
        self.comm = MPI.COMM_WORLD  
        self.rank = self.comm.Get_rank()  
        self.num_procs = self.comm.Get_size()  
          
        # Storage for results  
        self.profile_results = {}  
        self.best_param_vals = None

    def set_best_param_vals(self, best_param_vals):  
        """Set the best fit parameter values."""  
        self.best_param_vals = best_param_vals.copy() 

    def _profile_single_parameter(self, param_idx):  
        """Profile a single parameter."""  
        # Create parameter sweep range  
        best_val = self.best_param_vals[param_idx]  
        param_min = self.param_id.param_id_info["param_mins"][param_idx]  
        param_max = self.param_id.param_id_info["param_maxs"][param_idx]  

        # Define sweep range around best fit  
        range_width = (param_max - param_min) * self.range_factor  
        sweep_min = max(param_min, best_val - range_width)  
        sweep_max = min(param_max, best_val + range_width)  
          
        param_values = np.linspace(sweep_min, sweep_max, self.num_points)  
        costs = np.full(self.num_points, np.inf)  
          
        # Sequential evaluation of each fixed parameter value  
        # Each optimization uses all MPI processes internally  
        for i, fixed_val in enumerate(param_values):  
            if self.rank == 0:  
                print(f"  Evaluating point {i+1}/{self.num_points}: {fixed_val:.6f}")  
            
            # All processes participate in this single optimization  
            cost = self._optimize_with_fixed_parameter(param_idx, fixed_val)  
            costs[i] = cost  
          
        return {  
            'param_values': param_values,  
            'costs': costs,  
            'param_name': self.param_id.param_id_info['param_names'][param_idx],  
            'best_val': best_val  
        }
    
    def _optimize_with_fixed_parameter(self, fixed_param_idx, fixed_value):  
        """Optimize all parameters except one that is fixed using existing optimizer."""  
        
        # Store original state  
        original_param_id_info = self.param_id.param_id_info.copy()  
        original_best_params = self.best_param_vals.copy()  
        original_output_dir = self.param_id.output_dir

        # Set the fixed parameter value  
        original_best_params[fixed_param_idx] = fixed_value  

        # 1. Setup Phase
        if self.rank == 0:  
            temp_output_dir = os.path.join(original_output_dir, 'temp_profile_opt')  
            os.makedirs(temp_output_dir, exist_ok=True)  
            self.param_id.output_dir = temp_output_dir

        param_name = self.param_id.param_id_info["param_names"][fixed_param_idx]  

        if isinstance(param_name, list):  
            param_name = param_name[0]  

        # 2. Reduction and Execution
        # Note: Ensure remove_params_by_idx handles ALL keys to avoid the index error
        self.param_id.remove_params_by_idx([fixed_param_idx])

        # Set the fixed parameter value in the simulation model  
        self.param_id.sim_helper.set_param_vals([param_name], [fixed_value])  

        # Create reduced best parameters (excluding fixed one)  
        reduced_best_params = np.delete(original_best_params, fixed_param_idx)  

        # Set initial parameters for the reduced optimization  
        self.param_id.param_init = reduced_best_params  

        # Temporarily reduce optimization options for speed  
        original_options = self.param_id.optimiser_options.copy()  

        # Run optimization
        self.param_id.run()  
        best_cost = self.param_id.best_cost

        # 3. Restoration Phase (Previously in 'finally')
        # We restore original options and state manually here
        self.param_id.optimiser_options = original_options  
        self.param_id.set_param_id_info(original_param_id_info)
        self.param_id.set_best_param_vals(original_best_params)
        self.param_id.num_params = len(self.best_param_vals)  
        self.param_id.param_init = self.best_param_vals
        self.param_id.output_dir = original_output_dir
        
        return best_cost

    def _save_results(self):  
        """Save profile likelihood results."""  
        results_dir = os.path.join(self.output_dir, 'profile_likelihood')  
        os.makedirs(results_dir, exist_ok=True)  
          
        # Save numerical results  
        for param_idx, profile in self.profile_results.items():  

            # Handle param_name which might be a list  
            param_name = profile['param_name']  
            if isinstance(param_name, list):  
                # Join list elements or take first element  
                param_name_str = '_'.join(param_name) if len(param_name) > 1 else param_name[0]  
            else:  
                param_name_str = param_name  

            filename = f"param_{param_idx}_{param_name_str.replace('/', '_')}.json"  
            filepath = os.path.join(results_dir, filename)  
              
            with open(filepath, 'w') as f:  
                json.dump({  
                    'param_values': profile['param_values'].tolist(),  
                    'costs': profile['costs'].tolist(),  
                    'param_name': profile['param_name'],  
                    'best_val': float(profile['best_val'])  
                }, f, indent=2)  
                  
        # Save summary  
        summary = {  
            'num_params': len(self.profile_results),  
            'num_points': self.num_points,  
            'range_factor': self.range_factor,  
            'parameters': [profile['param_name'] for profile in self.profile_results.values()]  
        }  
          
        with open(os.path.join(results_dir, 'summary.json'), 'w') as f:  
            json.dump(summary, f, indent=2)  

    def _plot_profiles(self):  
        """Plot profile likelihood for all parameters."""  
        results_dir = os.path.join(self.output_dir, 'profile_likelihood')  
        os.makedirs(results_dir, exist_ok=True)  

        num_params = len(self.profile_results)  
        cols = min(3, num_params)  
        rows = (num_params + cols - 1) // cols  
        
        # Calculate global y-limits across all parameters  
        all_costs = []  
        for param_idx, profile in self.profile_results.items():  
            all_costs.extend(profile['costs'])  
        
        global_ymin = min(all_costs)  
        global_ymax = max(all_costs)  
        
        # Add some padding to the limits  
        y_padding = (global_ymax - global_ymin) * 0.05  
        global_ymin -= y_padding  
        global_ymax += y_padding  

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))  
        if num_params == 1:  
            axes = np.array([[axes]])
        elif rows == 1:  
            axes = axes.reshape(1, -1)  
        
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, (param_idx, profile) in enumerate(self.profile_results.items()):  

            ax = axes_flat[idx]

            # Plot profile  
            ax.plot(profile['param_values'], profile['costs'], 'b-', linewidth=2)  
            ax.axvline(profile['best_val'], color='r', linestyle='--', alpha=0.7, label='Best fit')  
              
            # Formatting  
            param_name = profile['param_name']  
            if '/' in param_name:  
                param_name = param_name.split('/')[-1]  # Use only parameter name  
            ax.set_xlabel(param_name)  
            ax.set_ylabel('Cost') 

            # Set the same y-limits for all subplots  
            ax.set_ylim(global_ymin, global_ymax)  
            
            ax.set_title(f'Profile: {param_name}')  
            ax.legend()  
            ax.grid(True, alpha=0.3)  
              
        # Hide unused subplots  
        for idx in range(num_params, rows * cols):  
            row = idx // cols  
            col = idx % cols  
            if rows > 1:  
                axes[row, col].set_visible(False)  
            else:  
                axes[col].set_visible(False)  
                  
        plt.tight_layout()  
        plt.savefig(os.path.join(results_dir, 'profile_likelihood_plots.pdf'),   
                   dpi=300, bbox_inches='tight')  
        plt.close()  
          
        # Also create individual plots  
        for param_idx, profile in self.profile_results.items():  
            fig, ax = plt.subplots(figsize=(8, 6))  
              
            ax.plot(profile['param_values'], profile['costs'], 'b-', linewidth=2)  
            ax.axvline(profile['best_val'], color='r', linestyle='--', alpha=0.7, label='Best fit')  
              
            # Add confidence intervals if possible  
            min_cost = np.min(profile['costs'])  
            threshold = min_cost + 3.84  # Chi-square threshold for 95% CI (1 DOF)  
              
            # Find intersection points  
            intersections = []  
            for i in range(len(profile['costs']) - 1):  
                if (profile['costs'][i] <= threshold <= profile['costs'][i + 1]) or \
                   (profile['costs'][i + 1] <= threshold <= profile['costs'][i]):
                    # Linear interpolation  
                    x1, x2 = profile['param_values'][i], profile['param_values'][i + 1]  
                    y1, y2 = profile['costs'][i], profile['costs'][i + 1]  
                    x_int = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)  
                    intersections.append(x_int)  
                      
            if len(intersections) >= 2:  
                ax.axvspan(intersections[0], intersections[1], alpha=0.2, color='green',   
                          label='95% CI')  
                  
            # Handle param_name which might be a list  
            param_name = profile['param_name']  
            if isinstance(param_name, list):  
                # Join list elements or take first element  
                param_name_str = '_'.join(param_name) if len(param_name) > 1 else param_name[0]  
            else:  
                param_name_str = param_name  
            if '/' in param_name_str:  
                param_name_str = param_name_str.split('/')[-1]  
                
            ax.set_xlabel(param_name_str)  
            ax.set_ylabel('Cost')  
            ax.set_title(f'Profile Likelihood: {param_name_str}')  
            ax.legend()  
            ax.grid(True, alpha=0.3)  
            
            filename = f"param_{param_idx}_{param_name_str.replace('/', '_')}.pdf"  
            # print(">>>>>>>>>>>>>>>>>>>", filename)
            plt.savefig(os.path.join(results_dir, filename),   
                    dpi=300, bbox_inches='tight')  
            plt.close()

    def _run(self):  
        """Run profile likelihood analysis for all parameters."""  
        if self.best_param_vals is None:  
            raise ValueError("Best parameter values must be set first - Run param_id script first")  
              
        num_params = len(self.best_param_vals)  
          
        if self.rank == 0:  
            print(f"Starting profile likelihood analysis for {num_params} parameters")  
              
        # Run profile for each parameter  
        for param_idx in range(num_params):  
            if self.rank == 0:
                print(self.param_id.param_id_info['param_names'])
                print(f"Profiling parameter {param_idx + 1}/{num_params}: "  
                      f"{self.param_id.param_id_info['param_names'][param_idx]}")
                
              
            profile = self._profile_single_parameter(param_idx)  
            self.profile_results[param_idx] = profile  
              
        if self.rank == 0:  
            self._save_results()  
            self._plot_profiles()