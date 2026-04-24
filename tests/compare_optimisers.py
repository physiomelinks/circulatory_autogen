#!/usr/bin/env python3
"""
Script to run parameter identification with multiple optimization methods
and compare the results.

This script can be run standalone or as part of the test suite.
It allows easy extension for additional optimization methods.

Usage:
    # Run with default methods (GA and CMA-ES)
    python tests/compare_optimisers.py
    
    # Run with specific methods
    python tests/compare_optimisers.py --methods genetic_algorithm CMA-ES
    
    # Run with custom configuration
    python tests/compare_optimisers.py --config user_run_files/user_inputs.yaml
"""
import os
import sys
import argparse
import numpy as np
import yaml
import time
import warnings
import numpy as np
from pathlib import Path
from mpi4py import MPI

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / 'src'))

from scripts.param_id_run_script import run_param_id
from parsers.PrimitiveParsers import YamlFileParser


class OptimiserComparison:
    """Class to handle comparison of different optimization methods."""
    
    def __init__(self, base_config, methods=None, num_calls=10000):
        """
        Initialize the comparison.
        
        Args:
            base_config: Base configuration dictionary
            methods: List of method names to compare (default: ['genetic_algorithm', 'CMA-ES'])
            num_calls: Number of function evaluations for each method
        """
        self.base_config = base_config.copy()
        self.num_calls = num_calls

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        
        # Default methods if not specified
        if methods is None:
            methods = ['genetic_algorithm', 'CMA-ES']
        self.methods = methods
        
        # Method-specific configuration overrides
        self.method_configs = {
            'genetic_algorithm': {
                'param_id_method': 'genetic_algorithm',
                # Use optimiser_options for preferred path; GA still supports ga_options via merge
                'optimiser_options': {},
            },
            'CMA-ES': {
                'param_id_method': 'CMA-ES',
                'optimiser_options': {'num_calls_to_function': num_calls},
            },
            'bayesian': {
                'param_id_method': 'bayesian',
                'optimiser_options': None,
            },
        }
        
        # Results storage
        self.results = {}
        self.runtimes = {}
        self._summary_warned = False
        self.param_names = self._load_param_names()

    def _load_param_names(self):
        """
        Try to load parameter names from params_for_id_path if provided.
        Falls back to None (will generate param_0, param_1, ... later).
        """
        params_path = self.base_config.get('params_for_id_path')
        if params_path and os.path.exists(params_path):
            try:
                import pandas as pd
                df = pd.read_csv(params_path)
                # Expect a column with parameter names; pick first column
                col = df.columns[0]
                names = df[col].tolist()
                return names
            except Exception:
                return None
        return None

    def _emit_summary_warning_if_ready(self):
        """Emit a single consolidated warning with cost/runtime table (and params) when all methods finished."""
        if self._summary_warned:
            return
        if len(self.results) < len(self.methods):
            return

        # Summary table (method, cost, runtime)
        lines = ["\n=== Optimiser comparison summary ===", f"{'method':<20}{'cost':>14}{'runtime_s':>14}"]
        for method in self.methods:
            res = self.results.get(method)
            if not res:
                continue
            cost = res.get("cost", float("nan"))
            runtime = res.get("runtime", float("nan"))
            lines.append(f"{method:<20}{cost:>14.6e}{runtime:>14.2f}")

        # Parameter table
        # Determine parameter names
        # If not loaded, use param_0.. based on max length seen
        max_len = 0
        for res in self.results.values():
            params = res.get("params")
            if params is not None:
                max_len = max(max_len, len(params))
        names = self.param_names or [f"param_{i}" for i in range(max_len)]

        # Build header
        header = f"{'parameter':<20}" + "".join([f"{m:>18}" for m in self.methods])
        lines.append("\nParameters:")
        lines.append(header)
        # Rows
        for idx, pname in enumerate(names):
            row = f"{pname:<20}"
            for method in self.methods:
                params = self.results.get(method, {}).get("params")
                val = np.nan
                if params is not None and idx < len(params):
                    val = params[idx]
                row += f"{val:>18.6e}"
            lines.append(row)

        summary = "\n".join(lines)
        warnings.warn(summary)
        self._summary_warned = True
    
    def get_output_dir(self, method):
        """Get the output directory for a given method."""
        param_id_output_dir = self.base_config.get('param_id_output_dir', 
                                                   str(root_dir / 'param_id_output'))
        file_prefix = self.base_config['file_prefix']
        obs_file = os.path.basename(self.base_config['param_id_obs_path']).replace('.json', '')
        return os.path.join(param_id_output_dir, f'{method}_{file_prefix}_{obs_file}')
    
    def load_results(self, output_dir):
        """Load best cost and parameters from output directory."""
        cost_file = os.path.join(output_dir, 'best_cost.npy')
        params_file = os.path.join(output_dir, 'best_param_vals.npy')
        
        if not os.path.exists(cost_file) or not os.path.exists(params_file):
            return None, None
        
        cost = np.load(cost_file)
        params = np.load(params_file)
        return cost, params
    
    def run_method(self, method):
        """Run optimization for a specific method."""
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"RUNNING {method.upper()} OPTIMIZATION")
            print(f"{'='*80}")
            sys.__stdout__.write(f"[COMPARE] method={method} num_calls={self.num_calls} num_procs={self.num_procs}\n")
            sys.__stdout__.flush()
        
        # Get method-specific config
        if method not in self.method_configs:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(self.method_configs.keys())}")
        
        method_config = self.method_configs[method]
        config = self.base_config.copy()
        config.update(method_config)
        
        # Ensure optimiser_options exists and preserve any provided options (e.g., max_patience)
        base_opts = config.get('optimiser_options') or {}
        if not isinstance(base_opts, dict):
            raise ValueError("optimiser_options must be a dictionary")
        method_opts = method_config.get('optimiser_options') if isinstance(method_config, dict) else None
        if method_opts is None:
            method_opts = {}
        elif not isinstance(method_opts, dict):
            raise ValueError("method-specific optimiser_options must be a dictionary")
        merged_opts = base_opts.copy()
        merged_opts.update(method_opts)
        merged_opts['num_calls_to_function'] = self.num_calls
        config['optimiser_options'] = merged_opts
        
        # Apply debug overrides if DEBUG is True
        if config.get('DEBUG', False):
            if 'debug_optimiser_options' in config and isinstance(config['debug_optimiser_options'], dict):
                config['optimiser_options'].update(config['debug_optimiser_options'])

        # Ensure num_calls_to_function reflects the requested comparison budget after any debug overrides
        config['optimiser_options']['num_calls_to_function'] = self.num_calls
        
        # Disable time-consuming post-processing for comparison
        config['do_ia'] = False
        config['do_mcmc'] = False
        
        # Start timing
        start_time = time.time()
        
        try:
            run_param_id(config)
            
            # End timing
            elapsed_time = time.time() - start_time
            self.runtimes[method] = elapsed_time
            
            if self.rank == 0:
                print(f"\n✓ {method} optimization completed")
                print(f"  Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

                # Load and store results
                output_dir = self.get_output_dir(method)
                cost, params = self.load_results(output_dir)
                
                if cost is not None:
                    self.results[method] = {
                        'cost': cost,
                        'params': params,
                        'output_dir': output_dir,
                        'runtime': elapsed_time,
                    }
                    if self.rank == 0:
                        # Emit a single consolidated warning once all methods have finished
                        self._emit_summary_warning_if_ready()
                    return_message = True
                else:
                    if self.rank == 0:
                        print(f"✗ Could not load results from {output_dir}")
                    return_message = False
            else:
                return_message = None

            self.comm.Barrier()
            return return_message

        except Exception as e:
            # End timing even on failure
            elapsed_time = time.time() - start_time
            self.runtimes[method] = elapsed_time
            
            print(f"\n✗ {method} optimization failed: {e}")
            print(f"  Runtime before failure: {elapsed_time:.2f} seconds")
            import traceback
            traceback.print_exc()
            return False
    
    def compare_results(self, reference_method=None):
        """Compare results from all methods."""
        if not self.results:
            print("No results to compare. Run optimizations first.")
            return
        
        print("\n" + "="*80)
        print("COMPARING OPTIMIZATION RESULTS")
        print("="*80)
        
        # Use first method as reference if not specified
        if reference_method is None:
            reference_method = list(self.results.keys())[0]
        
        if reference_method not in self.results:
            print(f"Reference method {reference_method} not found in results.")
            return
        
        ref_cost = self.results[reference_method]['cost']
        ref_params = self.results[reference_method]['params']
        
        print(f"\nReference method: {reference_method}")
        print(f"  Best cost: {ref_cost:.6e}")
        print(f"  Best parameters: {ref_params}")
        if reference_method in self.runtimes:
            ref_runtime = self.runtimes[reference_method]
            print(f"  Runtime: {ref_runtime:.2f} seconds ({ref_runtime/60:.2f} minutes)")
        
        # Compare each method to reference
        for method, result in self.results.items():
            if method == reference_method:
                continue
            
            cost = result['cost']
            params = result['params']
            
            print(f"\n{method}:")
            print(f"  Best cost: {cost:.6e}")
            print(f"  Best parameters: {params}")
            
            # Runtime information
            if method in self.runtimes:
                runtime = self.runtimes[method]
                ref_runtime = self.runtimes.get(reference_method, 0)
                if ref_runtime > 0:
                    speedup = ref_runtime / runtime
                    print(f"  Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
                    print(f"  Speedup vs {reference_method}: {speedup:.2f}x")
                else:
                    print(f"  Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
            
            # Cost comparison
            cost_diff = abs(cost - ref_cost)
            cost_rel_diff = cost_diff / max(abs(ref_cost), 1e-10) * 100
            print(f"  Cost difference from {reference_method}: {cost_diff:.6e} ({cost_rel_diff:.2f}%)")
            
            # Parameter comparison
            if ref_params.shape == params.shape:
                param_diff = np.abs(params - ref_params)
                param_rel_diff = param_diff / (np.abs(ref_params) + 1e-10) * 100
                print(f"  Parameter differences:")
                print(f"    Max absolute difference: {np.max(param_diff):.6e}")
                print(f"    Mean absolute difference: {np.mean(param_diff):.6e}")
                print(f"    Max relative difference: {np.max(param_rel_diff):.2f}%")
                print(f"    Mean relative difference: {np.mean(param_rel_diff):.2f}%")
                
                # Check if results are approximately the same
                if cost_rel_diff < 10 and np.max(param_rel_diff) < 50:
                    print(f"  ✓ Results are approximately similar to {reference_method}")
                else:
                    print(f"  ⚠ Results differ significantly from {reference_method}")
            else:
                print(f"  Parameter shapes don't match: {reference_method}={ref_params.shape}, {method}={params.shape}")
    
    def print_summary(self):
        """Print a summary table of all results."""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        
        # Find best cost
        best_cost = min(r['cost'] for r in self.results.values())
        
        print(f"\n{'Method':<20} {'Cost':<15} {'Cost Ratio':<12} {'Runtime (s)':<12} {'Runtime (min)':<12} {'Status':<20}")
        print("-" * 100)
        
        for method, result in sorted(self.results.items()):
            cost = result['cost']
            ratio = cost / best_cost if best_cost > 0 else np.inf
            runtime = result.get('runtime', self.runtimes.get(method, 0))
            status = "✓ Best" if cost == best_cost else f"{((cost - best_cost) / best_cost * 100):.1f}% worse"
            print(f"{method:<20} {cost:<15.6e} {ratio:<12.4f} {runtime:<12.2f} {runtime/60:<12.2f} {status:<20}")
        
        print(f"\nResults saved in:")
        for method, result in self.results.items():
            print(f"  {method}: {result['output_dir']}")


def load_config(config_path=None):
    """Load configuration from YAML file."""
    yaml_parser = YamlFileParser()
    
    if config_path is None:
        config_path = root_dir / 'user_run_files' / 'user_inputs.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable user_inputs_path_override for comparison
    if 'user_inputs_path_override' in config:
        config['user_inputs_path_override'] = False
    
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compare different optimization methods for parameter identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare GA and CMA-ES with default settings
  python tests/compare_optimisers.py
  
  # Compare specific methods
  python tests/compare_optimisers.py --methods genetic_algorithm CMA-ES
  
  # Use custom configuration file
  python tests/compare_optimisers.py --config user_run_files/user_inputs.yaml
  
  # Set number of function evaluations
  python tests/compare_optimisers.py --num-calls 5000
        """
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['genetic_algorithm', 'CMA-ES'],
        choices=['genetic_algorithm', 'CMA-ES', 'bayesian'],
        help='Optimization methods to compare (default: genetic_algorithm CMA-ES)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to user_inputs.yaml file (default: user_run_files/user_inputs.yaml)'
    )
    
    parser.add_argument(
        '--num-calls',
        type=int,
        default=10000,
        help='Number of function evaluations for each method (default: 10000)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Reference method for comparison (default: first method)'
    )
    
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip running optimizations, only compare existing results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create comparison object
    comparison = OptimiserComparison(config, methods=args.methods, num_calls=args.num_calls)
    
    # Run optimizations
    if not args.skip_run:
        for method in args.methods:
            comparison.run_method(method)
    else:
        # Load existing results
        print("Loading existing results...")
        print("Note: Runtime information not available when loading existing results.")
        for method in args.methods:
            output_dir = comparison.get_output_dir(method)
            cost, params = comparison.load_results(output_dir)
            if cost is not None:
                comparison.results[method] = {
                    'cost': cost,
                    'params': params,
                    'output_dir': output_dir,
                    'runtime': 0,  # Unknown runtime for existing results
                }
                print(f"✓ Loaded results for {method}")
            else:
                print(f"✗ No results found for {method} at {output_dir}")
    
    # Compare results
    if comparison.results:
        comparison.compare_results(reference_method=args.reference)
        comparison.print_summary()
    else:
        print("\nNo results available for comparison.")


if __name__ == '__main__':
    main()

