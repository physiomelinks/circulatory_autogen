'''
@author: Mohammad H. Shafieizadegan
@ reference: https://salib.readthedocs.io/en/latest/index.html
'''

import json
import os
import re
import sys
from sys import exit


def sanitize_for_filename(name):
    """Turn a display label (e.g. a name_for_plotting like ``u_{A_{R}}`` or an output name with
    spaces/commas) into a safe filename stem: any run of characters outside ``[A-Za-z0-9.-]``
    collapses to a single ``_``, then leading/trailing ``_`` are stripped. Without this, LaTeX-ish
    names produced paths with ``{}``, spaces and backslashes that fail to write on Windows
    (issue #167)."""
    safe = re.sub(r'[^A-Za-z0-9.-]+', '_', str(name))
    return safe.strip('_') or 'output'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
try:
    import opencor as oc
    opencor_available = True
except:
    opencor_available = False
    pass
from solver_wrappers import get_simulation_helper
from protocol_runners.protocol_executor import ProtocolExecutor
from SALib.sample import saltelli
# The Sobol *sampler* and the Sobol *analyzer* are different SALib modules that share the name
# `sobol`. Import the sampler under a distinct name so it does not shadow (or get shadowed by)
# the analyzer below -- `sample_type: sobol` previously called SALib.analyze.sobol.sample(),
# which does not exist, so it raised AttributeError.
from SALib.sample import sobol as sobol_sampler
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
from parsers.PrimitiveParsers import CSVFileParser, ObsAndParamDataParser
import csv
from tqdm import tqdm  # make sure tqdm is installed

class sobol_SA():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """

    def _is_rank0(self):
        try:
            return MPI.COMM_WORLD.Get_rank() == 0
        except Exception:
            return True

    def _rank0_print(self, *args, **kwargs):
        if self._is_rank0():
            print(*args, **kwargs)

    def __init__(self, model_path, model_out_names, solver_info, SA_info, dt, sa_output_dir, 
                 param_id_path = None, params_for_id_path=None, use_MPI = False, verbose=False, 
                 sim_time=2.0, pre_time=20.0, model_type=None):

        """
        Initializes the Sensitivity_analysis class.
        Parameters:
            model_path (str): Path to the model file.
            model_out_names (list): Names of the model outputs to be analyzed.
            solver_info (dict): Solver configuration parameters.
            SA_info (dict): Configuration for sensitivity analysis, including sample type, number of samples,
                           parameter names, and their bounds.
            protocol_info (dict): Information about the simulation protocol, including simulation times and pre-times.
            dt (float): Time step for the simulation.
            save_path (str): Directory where results will be saved.
            verbose (bool): If True, prints additional information during execution.
            model_type (str): Type of the model (e.g., "casadi_python", "numpy").
        """

        self.model_path = model_path
        self.output_dir = None
        self.verbose = verbose
        self.set_output_dir(sa_output_dir)

        self.solver_info = solver_info
        self.SA_info = SA_info
        self.sample_type = self.SA_info["sample_type"]
        self.num_params = None
        self.protocol_info = None
        self.dt = dt

        self.model_type = model_type
        mode = "casadi" if self.model_type == "casadi_python" else "numpy"
        # set up observables functions
        self.sfp = scriptFunctionParser()
        self.operation_funcs_dict = self.sfp.get_operation_funcs_dict(mode)

        self.obs_and_param_parser = None
        self.gt_df = None
        self.obs_info = None
            

        if param_id_path is not None:
            self.obs_and_param_parser = ObsAndParamDataParser()
            parsed_data = self.obs_and_param_parser.parse_obs_data_json(
                param_id_obs_path=param_id_path,
                pre_time=pre_time,
                sim_time=sim_time
            )
            self.gt_df = parsed_data["gt_df"]
            self.protocol_info = parsed_data["protocol_info"]
            # TODO should we include prediction info in SA?
            self.prediction_info = parsed_data["prediction_info"]

            self.obs_info = self.obs_and_param_parser.process_obs_info(gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt)
            self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(
                gt_df=self.gt_df,
                protocol_info=self.protocol_info,
                dt=self.dt
            )

        if self.protocol_info is None:
            self.protocol_info = {
                "pre_times": [pre_time],
                "sim_times": [[sim_time]],
                "params_to_change": [[None]]
            }

        # set up opencor simulation
        if self.protocol_info['sim_times'][0][0] is not None:
            self.sim_time = self.protocol_info['sim_times'][0][0]
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if self.protocol_info['pre_times'][0] is not None:
            self.pre_time = self.protocol_info['pre_times'][0]
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001

        self.sim_helper = self.initialise_sim_helper()
        self._protocol_executor = ProtocolExecutor(self.sim_helper)
        if self.sim_time is not None and self.pre_time is not None:
            self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
            self.n_steps = int(self.sim_time/self.dt)


        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.use_mpi = use_MPI

        self.params_for_id_path = params_for_id_path
        self.param_id_info = None
        if self.params_for_id_path:
            self.param_id_info = self.obs_and_param_parser.get_param_id_info(self.params_for_id_path)
            self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)
            # self.__set_and_save_param_names()

        if self.param_id_info is not None:
            self.SA_info = self.create_SA_info(self.sample_type, self.SA_info["num_samples"])
    

    def add_user_operation_func(self, func):
        self.operation_funcs_dict = self.sfp.add_user_operation_func(self.operation_funcs_dict, func)
        
    def set_ground_truth_data(self, obs_data_dict):
        if self.rank == 0:
            print(f'Setting ground truth data: {obs_data_dict}')
        if self.obs_and_param_parser is None:
            self.obs_and_param_parser = ObsAndParamDataParser()
        parsed_data = self.obs_and_param_parser.parse_obs_data_json(
            obs_data_dict=obs_data_dict,
            pre_time=self.pre_time,
            sim_time=self.sim_time
        )
        self.gt_df = parsed_data["gt_df"]
        self.protocol_info = parsed_data["protocol_info"]
        self.prediction_info = parsed_data["prediction_info"]

        self.obs_info = self.obs_and_param_parser.process_obs_info(gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt)
        self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(
            gt_df=self.gt_df,
            protocol_info=self.protocol_info,
            dt=self.dt
        )
        if self.rank == 0:
            print(f'Ground truth data set: {self.obs_info}')
    
    def set_params_for_id(self, params_for_id_dict):
        if self.rank == 0:
            print(f'Setting params for id: {params_for_id_dict}')
        if self.obs_and_param_parser is None:
            self.obs_and_param_parser = ObsAndParamDataParser()
        self.param_id_info = self.obs_and_param_parser.get_param_id_info_from_entries(params_for_id_dict)
        self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)
        self.create_SA_info(self.sample_type, self.SA_info["num_samples"])
        if self.rank == 0:
            print(f'Params for id set: {self.param_id_info["param_names"]}')

    def set_sa_options(self, sa_options):
        self.SA_info = self._create_SA_info(sa_options['sample_type'], sa_options['num_samples'])
        self.set_output_dir(sa_options['output_dir'])

    def _create_SA_info(self, sample_type, num_samples):
        
        # Use param_id_info to build SA_info dynamically
        if not hasattr(self, "param_id_info") or not self.param_id_info:
            raise ValueError("param_id_info is not set. Please run __set_and_save_param_names() first.")

        SA_info = {
            "sample_type": sample_type,
            "param_names": [name[0] if isinstance(name, list) else name for name in self.param_id_info["param_names"]],
            "num_samples": num_samples,
            "param_mins": list(self.param_id_info["param_mins"]),
            "param_maxs": list(self.param_id_info["param_maxs"])
        }

        # if self.verbose:
        #     print("Sensitivity Analysis Configuration:")
        #     print(json.dumps(SA_info, indent=4))

        self.num_params = len(SA_info["param_names"])

        return SA_info

    def create_SA_info(self, sample_type, num_samples):
        # Backwards compatibility alias
        return self._create_SA_info(sample_type, num_samples)


    def initialise_sim_helper(self):
        solver = None
        if isinstance(self.solver_info, dict):
            solver = self.solver_info.get("solver")
        # Honour the configured model_type (e.g. casadi_python); only fall back to
        # inferring it from the file extension when it wasn't supplied.
        model_type = self.model_type
        if model_type is None:
            model_type = "python" if str(self.model_path).endswith(".py") else "cellml_only"
        return get_simulation_helper(
            model_path=self.model_path,
            solver=solver,
            model_type=model_type,
            dt=self.dt,
            sim_time=self.sim_time,
            solver_info=self.solver_info,
            pre_time=self.pre_time,
        )

    def set_output_dir(self, path):
        
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_samples(self):

        problem = {
            'num_vars': self.num_params,
            'names': self.SA_info["param_names"],
            'bounds': list(zip(self.SA_info["param_mins"], self.SA_info["param_maxs"]))
        }
        self.problem = problem

        self.num_samples = self.SA_info["num_samples"]

        if self.SA_info["sample_type"] == "saltelli":
            samples = saltelli.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        elif self.SA_info["sample_type"] == "sobol":
            samples = sobol_sampler.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        else:
            raise ValueError(f"Unsupported sample type: {self.SA_info['sample_type']}")
        
        return samples
    
    def run_model_and_get_results(self, param_vals):
        self.sim_helper.set_param_vals(self.SA_info["param_names"], param_vals)
        self.sim_helper.reset_states()
        success = self.sim_helper.run()
        if not success:
            print(f"[MPI Rank {self.rank}] Failed to converge for params: {param_vals}")
            return None

        operands = self.sim_helper.get_results(self.obs_info["operands"])

        self.sim_helper.reset_and_clear()
        # t = self.sim_helper.tSim - self.pre_time
        # return y, t
        return operands
    
    def generate_outputs_mpi(self, samples):
        #need to added an array to save tmp data, each calibration need to updated/re-initial
        self.temp_results = {}
        
        # Split samples across ranks
        n_samples = len(samples)
        samples_per_rank = n_samples // self.num_procs
        remainder = n_samples % self.num_procs

        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        local_samples = samples[start:end]

        self._rank0_print(f"[MPI Rank {self.rank}] Starting samples {start}:{end} (total {len(local_samples)})")

        local_outputs = []

        # Create a single progress bar for rank 0 only to avoid noisy output from all ranks
        with tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True, disable=self.rank != 0) as pbar:
            for param_vals in local_samples:

                # Delegate the multi-experiment / multi-subexperiment loop to
                # ProtocolExecutor.  continue_on_failure=True preserves existing
                # behaviour: failed sub-experiments produce None entries rather
                # than aborting the whole sample.
                _success, operands_outputs_dict, _, _ = self._protocol_executor.run_protocol(
                    self.protocol_info,
                    id_param_names=self.param_id_info["param_names"],
                    id_param_vals=param_vals,
                    result_variables=self.obs_info["operands"],
                    continue_on_failure=True,
                )
                if not _success:
                    self._rank0_print(
                        f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}"
                    )

                features = []
                for j in range(len(self.obs_info["operations"])):
                    func = self.operation_funcs_dict[self.obs_info["operations"][j]]
                    exp_idx = self.obs_info["experiment_idxs"][j]
                    subexp_idx = self.obs_info["subexperiment_idxs"][j]
                    operands_outputs = operands_outputs_dict.get((exp_idx, subexp_idx), None)
                    if operands_outputs is not None:
                        key_idxt = self.obs_info["names_for_plotting"][j]
                        raw_kwargs = self.obs_info["operation_kwargs"][j]
                        #every time check it and update to {} when not exist
                        if isinstance(raw_kwargs, dict):
                            kwargs = raw_kwargs.copy()
                        else:
                            kwargs = {}
                        for k, v in list(kwargs.items()):
                            if isinstance(v, str) and v in self.temp_results:
                                kwargs[k] = self.temp_results[v]
                        #also need to replace below sentence
                        feature = func(*operands_outputs[j], **kwargs)
                        self.temp_results[key_idxt] = feature
                        if feature is None or (isinstance(feature, (float, int)) and np.isnan(feature)):
                            feature = np.nanmean(features) if not np.all(np.isnan(features)) else 0.0

                        features.append(feature)
                    else:
                        # WARNING: using mean biases variance estimates (shrinks variance), underestimates sensitivity
                        # TODO: come up with a better way to impute missing features
                        # Append the mean of the current features (ignoring None) -> reduces variance and bias induces toward zero
                        features.append(np.mean(local_outputs))

                local_outputs.append(features)
                pbar.update(1)

        self._rank0_print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)
            self._rank0_print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")
            return outputs
        else:
            return None

    def sobol_index(self, outputs):

        if self.rank !=0:
            return None, None, None
        
        outputs = np.array(outputs)
        
        # Ensure outputs are numeric scalars; SALib expects 1D numeric Y per output.
        if outputs.dtype == object:
            def _coerce_scalar(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.size != 1:
                        raise TypeError(
                            "Sobol outputs must be scalar per sample/output; "
                            f"got array-like with size={arr.size} and dtype={arr.dtype}."
                        )
                    val = arr.item()
                return float(val)

            try:
                outputs = np.array([[_coerce_scalar(v) for v in row] for row in outputs], dtype=float)
            except Exception as e:
                raise TypeError(
                    "Sobol outputs are not numeric scalars. "
                    "Check your operation functions or operands; they may return lists/arrays "
                    "instead of a single number."
                ) from e
    
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]  # convert to (n_samples, 1)

        n_outputs = outputs.shape[1]
        S1_all = np.zeros((n_outputs, self.num_params))
        ST_all = np.zeros((n_outputs, self.num_params))
        S2_all = np.zeros((n_outputs, self.num_params, self.num_params))

        # change names to series here as SALib has an issue with lists https://github.com/SALib/SALib/issues/671
        self.problem['names'] = pd.Series(self.problem['names'])

        for i in range(n_outputs):
            Si = sobol.analyze(self.problem, outputs[:,i], print_to_console=self.verbose)
            S1_all[i, :] = Si['S1']
            ST_all[i, :] = Si['ST']
            S2_all[i, :] = np.array(Si['S2'])

        return S1_all, ST_all, S2_all

    def plot_sobol_first_order_idx(self, S1_all, ST_all):

        if self.rank !=0:
            return
        
        """
        Plot first-order and total-order Sobol indices for multiple outputs.

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        n_outputs = S1_all.shape[0]
        x = np.arange(self.num_params)

        for i in range(n_outputs):
            S1 = S1_all[i]
            ST = ST_all[i]
            output_name = rf"{self.obs_info['names_for_plotting'][i]} - experiment{self.obs_info['experiment_idxs'][i]}, subexperiment{self.obs_info['subexperiment_idxs'][i]}"
            # output_name = self.obs_info["names_for_plotting"][i] if hasattr(self, "obs_info") else f"Output_{i}"

            # Set figure width adaptively based on number of parameters (xticks)
            fig_width = max(12, 1.0 * len(self.SA_info["param_names"]))
            plt.figure(figsize=(fig_width, 5))
            plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
            plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

            plt.xticks(x, self.SA_info["param_names"], rotation=45, fontsize=8)
            plt.ylabel('Sensitivity Index')
            plt.title(rf'Sobol Sensitivity - {output_name}')
            plt.legend()
            plt.tight_layout()

            file_name = f"{sanitize_for_filename(output_name)}_n{self.num_samples}_First_order_idx.png"
            plt.savefig(os.path.join(self.output_dir, file_name))
            plt.clf()
            plt.close()

    def plot_sobol_S2_idx(self, S2_all):
        """
        Plot second-order Sobol interaction indices for multiple outputs.

        Parameters:
            S2_all (np.ndarray): Second-order indices, shape (n_outputs, n_params, n_params)
        """

        if self.rank !=0:
            return
        
        n_outputs = S2_all.shape[0]
        for i in range(n_outputs):
            S2 = S2_all[i]
            output_name = rf"{self.obs_info['names_for_plotting'][i]} - experiment{self.obs_info['experiment_idxs'][i]}, subexperiment{self.obs_info['subexperiment_idxs'][i]}"

            # plt.figure(figsize=(6, 5))
            fig_width = max(6, 1.0 * len(self.SA_info["param_names"]))
            plt.figure(figsize=(fig_width, fig_width))
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=self.SA_info["param_names"], yticklabels=self.SA_info["param_names"], cmap="coolwarm")
            plt.title(rf"2nd order Sobol Indices - {output_name}")
            plt.tight_layout()

            filename = f"{sanitize_for_filename(output_name)}_n{self.num_samples}_2nd_order_idx.png"
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.clf()
            plt.close()

    def get_sobol_output_labels(self, num_labels):
        """
        Generates a list of output labels for Sobol sensitivity analysis plots.

        Labels are generated based on whether plotting information exists in self.obs_info
        
        Args:
            self (object): The instance containing the obs_info dictionary.
            sobol_indices (np.ndarray): Array used for determining the number of labels.
            S1_all (np.ndarray): Array used for determining the number of labels (often has same shape as sobol_indices).

        Returns:
            list: A list of formatted label strings.
        """
        
        end_range = num_labels

        has_plotting_info = (
            hasattr(self, "obs_info") and 
            self.obs_info and 
            "names_for_plotting" in self.obs_info
        )
        
        if has_plotting_info:
            # Use a rich label format with experimental details
            def generate_label(i):
                name = self.obs_info['names_for_plotting'][i]
                # Use .get() with a default for slightly more robustness
                exp_idx = self.obs_info.get('experiment_idxs', ['?'])[i]
                sub_idx = self.obs_info.get('subexperiment_idxs', ['?'])[i]
                # The rf"..." is used to render text as LaTeX/Math Text
                return rf"{name} (Exp{exp_idx}, Sub{sub_idx})"
        else:
            # Use a generic label format
            def generate_label(i):
                return f"feature_{i}"

        output_labels = [generate_label(i) for i in range(end_range)]
            
        return output_labels
    
    def plot_sobol_heatmap(self, S1_all, ST_all):
        
        if self.rank != 0:
            return
        
        """
        Generates 2D heatmaps for first-order (S1) and total-order (ST) Sobol indices.
        
        The heatmaps show:
        Y-axis: Input Parameters (self.SA_info["param_names"])
        X-axis: Model Outputs (concatenated names from self.obs_info)
        Color: Sobol Index Value
        
        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        
        print("\nGenerating Sobol Index Heatmaps...")
        
        # 1. Define Axis Labels
        output_labels = self.get_sobol_output_labels(S1_all.shape[0])

        param_labels = [rf"{name}" for name in self.param_id_info["param_names_for_plotting"]]

        # Current shape: (n_outputs, n_params) -> Desired shape: (n_params, n_outputs)
        S1_heatmap_data = S1_all.T
        ST_heatmap_data = ST_all.T
        
        # Define the title prefix using the total sample count (N * (D+2))
        total_samples = S1_all.shape[1] * (S1_all.shape[0] + 2) if hasattr(self, 'num_params') else 'N/A'
        title_prefix = f"Sobol Indices (N={self.num_samples*(self.num_params+2)})"
        
        def create_heatmap(data, index_type):
            
            df_data = pd.DataFrame(data, index=param_labels, columns=output_labels)
            
            fig_width = max(10, len(output_labels) * 0.5) 
            fig_height = max(6, len(param_labels) * 0.5)
            
            plt.figure(figsize=(fig_width, fig_height))
            
            sns.heatmap(
                df_data,
                annot=True,               # Annotate with the index values
                fmt=".2f",                # Format annotations to 2 decimal places
                cmap="viridis",           # Good colormap for continuous data
                linewidths=0.5,           # Lines between cells
                linecolor='lightgray',
                cbar_kws={'label': f'{index_type} Index Value'}
            )

            plt.title(f'{title_prefix} - {index_type}', fontsize=14)
            plt.xlabel('Model Output', fontsize=12)
            plt.ylabel('Input Parameter', fontsize=12)
            
            plt.xticks(rotation=45, ha='right', fontsize=8) 
            plt.yticks(rotation=0, fontsize=8) 
            
            plt.tight_layout()
            
            file_name = f"{index_type.replace('-', '_')}_Sobol_Heatmap.png"
            save_path = os.path.join(self.output_dir, file_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {index_type} heatmap to {save_path}")

        create_heatmap(S1_heatmap_data, 'First-Order ($S_1$)')
        create_heatmap(ST_heatmap_data, 'Total-Order ($S_T$)')

    @staticmethod
    def _uniquify_output_names(labels, ops):
        """Return labels guaranteed unique so each Sobol output keeps its own
        column. pandas assigns DataFrame columns by label, so a repeated label
        would let a later data_item silently overwrite an earlier one and the
        outputs would collapse to one-per-name (issue #240). Repeated labels are
        first distinguished by their operation, then by a ' #k' suffix."""
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        stage1 = [
            f"{lab} [{ops[i]}]" if counts[lab] > 1 and ops[i] else lab
            for i, lab in enumerate(labels)
        ]
        seen, out = {}, []
        for lab in stage1:
            if lab in seen:
                seen[lab] += 1
                out.append(f"{lab} #{seen[lab]}")
            else:
                seen[lab] = 1
                out.append(lab)
        return out

    def save_sobol_indices(self, S1_all, ST_all, S2_all):
        if self.rank != 0:
            return

        """
        Save all Sobol indices to single CSV files (one for S1/ST, one for S2).

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
            S2_all (np.ndarray): Second-order Sobol indices, shape (n_outputs, n_params, n_params)
        """
        n_outputs = S1_all.shape[0]
        param_names = self.SA_info["param_names"]

        # Prepare output/feature names. Two data_items that resolve to the same
        # (name_for_plotting, experiment, subexperiment) produce identical column
        # labels; since pandas assigns DataFrame columns by label, the later one
        # would silently overwrite the earlier and the Sobol output would collapse
        # to one-per-name (issue #240). Build the labels, then disambiguate any
        # collisions so every data_item keeps its own column.
        names = self.obs_info['names_for_plotting']
        exps = self.obs_info['experiment_idxs']
        subs = self.obs_info['subexperiment_idxs']
        ops = self.obs_info.get('operations', [])

        n_named = n_outputs if n_outputs <= len(names) else n_outputs - 1
        base_labels, base_ops = [], []
        for i in range(n_named):
            base_labels.append(f"{names[i]} (Exp{exps[i]}, Sub{subs[i]})")
            base_ops.append(ops[i] if i < len(ops) else None)
        if n_outputs > len(names):
            base_labels.append("Cost")
            base_ops.append(None)

        output_names = self._uniquify_output_names(base_labels, base_ops)

        # --- Save S1/ST indices ---
        df_Sobol = pd.DataFrame({'Parameter': param_names})
        for i, out_name in enumerate(output_names):
            df_Sobol[f"S1_{out_name}"] = S1_all[i]
            df_Sobol[f"ST_{out_name}"] = ST_all[i]
        file_name = f"all_outputs_n{self.num_samples}_Sobol_indices.csv"
        df_Sobol.to_csv(os.path.join(self.output_dir, file_name), index=False)

        # --- Save S2 indices ---
        # For each output, flatten S2 into a DataFrame with MultiIndex columns
        s2_dict = {}
        for i, out_name in enumerate(output_names):
            # S2_all[i]: (n_params, n_params)
            s2_flat = pd.DataFrame(
                S2_all[i],
                index=param_names,
                columns=param_names
            )
            # Rename columns to include output name
            s2_flat.columns = [f"{out_name}__{col}" for col in s2_flat.columns]
            s2_dict[out_name] = s2_flat

        # Concatenate all S2 DataFrames horizontally
        df_S2 = pd.concat([s2_dict[out_name] for out_name in output_names], axis=1)
        df_S2.index.name = "Parameter"
        file_name_S2 = f"all_outputs_n{self.num_samples}_Sobol_2nd_order_indices.csv"
        df_S2.to_csv(os.path.join(self.output_dir, file_name_S2))

    def load_sobol_indices(self):
        """
        Loads S1 and ST indices from the saved CSV file and returns them as a dictionary.
        Returns:
            dict: { 'S1': {out_name: {param: val}}, 'ST': {out_name: {param: val}} }
        """
        file_name = f"all_outputs_n{self.num_samples}_Sobol_indices.csv"
        file_path = os.path.join(self.output_dir, file_name)
        
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Use the 'Parameter' column as the index for easy dict conversion
        df.set_index('Parameter', inplace=True)
        
        results = {'S1': {}, 'ST': {}}
        
        # Iterate through columns to separate S1 and ST by output name
        for col in df.columns:
            if col.startswith('S1_'):
                out_name = col.replace('S1_', '', 1)
                results['S1'][out_name] = df[col].to_dict()
            elif col.startswith('ST_'):
                out_name = col.replace('ST_', '', 1)
                results['ST'][out_name] = df[col].to_dict()
                
        return results
 
    def run(self):
        samples = self.generate_samples()
        if self.use_mpi:
            outputs = self.generate_outputs_mpi(samples)
            if self.rank == 0:
                S1_all, ST_all, S2_all = self.sobol_index(outputs)
                # print(f">>>>>>>>>>  {S1_all}, {ST_all}, {S2_all}")
                return S1_all, ST_all, S2_all
            else:
                return None, None, None
        else:
            outputs = self.generate_outputs(samples)
            S1_all, ST_all, S2_all = self.sobol_index(outputs)
            return S1_all, ST_all, S2_all

