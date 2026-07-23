'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../solver_wrappers'))
import math as math
try:
    import opencor as oc
    opencor_available = True
except:
    opencor_available = False
    pass
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import paperPlotSetup
paperPlotSetup.Setup_Plot(3)
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import math
from parsers.PrimitiveParsers import YamlFileParser
from parsers.PrimitiveParsers import analysis_options


def sa_method_choices():
    """The sensitivity-analysis ``method`` values, read from the discoverable schema
    (``ANALYSIS_OPTIONS['sensitivity_analysis']`` in parsers.PrimitiveParsers) rather than
    hardcoded, so the dispatch, its error message and the docs stay in step with CUFLynx."""
    for opt in analysis_options('sensitivity_analysis'):
        if opt['name'] == 'method':
            return list(opt.get('choices', []))
    return []
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )
from sensitivity_analysis.sobolSA import sobol_SA

GREEN = '\033[92m'
CYAN = '\033[36m'
RED = '\033[31m'
# ANSI escape code to reset the color back to the terminal's default
RESET = '\033[0m'

class SensitivityAnalysis():
    """Variance-based (Sobol) global sensitivity analysis for a 0D model.

    Wraps the Sobol SA manager and coordinates loading observation data,
    selecting parameters, running the analysis, and ranking the most impactful
    parameters. Construct from a config dict with
    [`init_from_dict`][sensitivity_analysis.sensitivityAnalysis.SensitivityAnalysis.init_from_dict].

    Typical flow::

        sa = SensitivityAnalysis.init_from_dict(inp)
        sa.set_ground_truth_data(obs_data_dict)
        sa.set_params_for_id(params_for_id_dict)
        sa.run_sensitivity_analysis(sa_options)
        top = sa.choose_most_impactful_params_sobol(top_n=5, index_type='ST')

    Args:
        model_path: Path to the generated model file.
        model_type: ``'cellml_only'``, ``'python'`` or ``'casadi_python'``.
        file_name_prefix: Model name prefix.
        sa_options: SA options dict (``method``, ``sample_type``,
            ``num_samples``, ``output_dir``).
        DEBUG: Enable debug behaviour.
        param_id_output_dir: Root output directory.
        resources_dir: Directory holding input resources.
        model_out_names: Optional explicit list of model output variable names.
        solver_info: Solver config dict.
        dt: Output sampling step (s).
        optimiser_options: Options dict (used if a nominal calibration is run).
        param_id_obs_path: Optional path to an ``obs_data.json``.
        params_for_id_path: Optional path to a ``{prefix}_params_for_id.csv``.
    """
    def __init__(self, model_path, model_type, file_name_prefix, sa_options, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, model_out_names=[],
                 solver_info={}, dt=0.01, optimiser_options={}, param_id_obs_path=None, params_for_id_path=None,
                 operation_funcs_external_path=None, cost_funcs_external_path=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.model_out_names = model_out_names
        self.solver_info = solver_info
        self.dt = dt
        self.optimiser_options = optimiser_options
        self.param_id_obs_path = param_id_obs_path
        self.params_for_id_path = params_for_id_path
        self.sa_options = sa_options
        # Optional external user-func files (issue #303), threaded into the Sobol manager and the
        # local-sensitivity engine so their operation/cost dicts merge them alongside the built-ins.
        self.operation_funcs_external_path = operation_funcs_external_path
        self.cost_funcs_external_path = cost_funcs_external_path
        sa_output_dir = sa_options['output_dir']

        self.SA_manager = sobol_SA(self.model_path, self.model_out_names, self.solver_info, sa_options, self.dt,
                            sa_output_dir, param_id_path=self.param_id_obs_path, params_for_id_path=self.params_for_id_path,
                            verbose=False, use_MPI=True, model_type=self.model_type,
                            operation_funcs_external_path=operation_funcs_external_path,
                            cost_funcs_external_path=cost_funcs_external_path)

        # For the local (derivative-based) method, which -- unlike Sobol -- runs through a
        # backend-agnostic param-id engine (mirroring IdentifiabilityAnalysis), not the Sobol
        # sampling manager. Populated lazily by run_local_sensitivity from the stashes below.
        self._inp_data_dict = None
        self._obs_data_dict = None
        self._params_for_id = None
        self._local_engine = None
        self.local_sensitivities = None

    @classmethod
    def init_from_dict(cls, inp_data_dict):
        """Build a `SensitivityAnalysis` from a configuration dict.

        ``file_prefix`` is accepted as an alias for ``file_name_prefix``.

        Args:
            inp_data_dict: Configuration dict (see
                [`get_default_inp_data_dict`][utilities.utility_funcs.get_default_inp_data_dict]).

        Returns:
            SensitivityAnalysis: A configured instance.
        """
        # parse the user inputs dictionary
        yaml_parser = YamlFileParser()
        inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict)
        # Only pass kwargs that exist in inp_data_dict
        arg_options = [
            'model_path', 'model_type', 'file_name_prefix', 'sa_options', 'DEBUG', 'param_id_output_dir',
            'resources_dir', 'model_out_names', 'solver_info',
            'dt', 'optimiser_options', 'param_id_obs_path', 'params_for_id_path',
            'operation_funcs_external_path', 'cost_funcs_external_path',
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}

        # Support common naming used elsewhere
        if 'file_name_prefix' not in kwargs and 'file_prefix' in inp_data_dict:
            kwargs['file_name_prefix'] = inp_data_dict['file_prefix']

        sa = cls(**kwargs)
        # Keep the parsed config so the local method can build a param-id engine from it.
        sa._inp_data_dict = inp_data_dict
        return sa

    def init_from_all_dicts(cls, inp_data_dict, obs_data_dict, params_for_id_dict, sa_options):
        sa = cls.init_from_dict(inp_data_dict)
        sa.set_ground_truth_data(obs_data_dict)
        sa.set_params_for_id(params_for_id_dict)
        sa.set_sa_options(sa_options)
        return sa

    def add_user_operation_func(self, func):
        """Register a custom feature-extraction function (see
        [`CVS0DParamID.add_user_operation_func`][param_id.paramID.CVS0DParamID.add_user_operation_func])."""
        self.SA_manager.add_user_operation_func(func)

    def set_sa_options(self, sa_options):
        """Set/update the sensitivity-analysis options dict.

        Args:
            sa_options: e.g. ``method`` (see ``sa_method_choices()`` / the sensitivity_analysis
                schema), ``sample_type``, ``num_samples``, ``output_dir``.
        """
        self.SA_manager.set_sa_options(sa_options)

    def set_ground_truth_data(self, obs_data_dict):
        """Set the observation data defining the outputs of interest.

        Args:
            obs_data_dict: Observation data dict (see
                [`ObsDataCreator`][utilities.obs_data_helpers.ObsDataCreator]).
        """
        self._obs_data_dict = obs_data_dict
        self.SA_manager.set_ground_truth_data(obs_data_dict)

    def set_params_for_id(self, params_for_id_dict):
        """Set which parameters to vary and their bounds.

        Args:
            params_for_id_dict: List of parameter entries (see
                [`CVS0DParamID.set_params_for_id`][param_id.paramID.CVS0DParamID.set_params_for_id]).
        """
        self._params_for_id = params_for_id_dict
        self.SA_manager.set_params_for_id(params_for_id_dict)

    def set_model_out_names(self, obs_data_dict):
        """Derive and store the model output variable names from the obs data."""
        # TODO fix for arbitrary number of operands
        # mohammad must have done this already.
        self.model_out_names = []
        for item in obs_data_dict["data_items"]:
            if len(item["operands"]) > 1:
                print(f'{RED}ERROR: more than one operand for {item["name_for_plotting"]}, not supported{RESET}')
                exit()
            self.model_out_names.append(item["operands"][0])

    def run_sensitivity_analysis(self, sa_options=None):
        """Run the sensitivity analysis, dispatching by ``method``.

        Args:
            sa_options: Optional options dict; if omitted, the options set at
                construction (or via ``set_sa_options``) are used. ``method`` is one of the
                values declared for ``sa_options.method`` in the sensitivity_analysis schema
                (``ANALYSIS_OPTIONS`` in parsers.PrimitiveParsers) -- see ``sa_method_choices()``.
        """
        if sa_options is None:
            sa_options = self.sa_options
        else:
            self.set_sa_options(sa_options)

        # Dispatch by naming convention: method 'x' -> self.run_x_sensitivity. The valid set is
        # the schema's method choices, so adding a method is: declare it in ANALYSIS_OPTIONS and
        # define run_<method>_sensitivity -- nothing here is hardcoded.
        method = sa_options['method']
        if method not in sa_method_choices():
            print(f'{RED}ERROR: sensitivity analysis method {method!r} not recognised; '
                  f'valid methods are {sa_method_choices()}{RESET}')
            exit()
        handler = getattr(self, f'run_{method}_sensitivity', None)
        if handler is None:
            print(f'{RED}ERROR: sensitivity method {method!r} is in the schema but no '
                  f'run_{method}_sensitivity handler is defined{RESET}')
            exit()
        handler(sa_options)

    def run_sobol_sensitivity(self, sa_options=None):
        """Run Sobol SA and (on rank 0) save indices and plots.

        Computes first-order (S1), total (ST) and second-order (S2) Sobol
        indices. Ground-truth data and parameters for id must be set first.

        Args:
            sa_options: Optional options dict (see ``set_sa_options``).
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        output_dir = self.SA_manager.output_dir

        self.SA_manager.set_sa_options(sa_options)

        if self.SA_manager.gt_df is None or self.SA_manager.param_id_info is None:
            print(f'{RED}ERROR: need to set ground truth data and params for id before running sobol sensitivity analysis{RESET}')
            exit()

        S1_all, ST_all, S2_all = self.SA_manager.run()

        if rank == 0:
            print(f"{GREEN}Sensitivity analysis completed successfully :){RESET}")
            print(f'{CYAN}saving results in {output_dir}{RESET}')
            self.SA_manager.save_sobol_indices(S1_all, ST_all, S2_all)
            self.SA_manager.plot_sobol_first_order_idx(S1_all, ST_all)
            self.SA_manager.plot_sobol_S2_idx(S2_all)
            self.SA_manager.plot_sobol_heatmap(S1_all, ST_all)

    def _build_local_engine(self):
        """Build (once) a param-id engine for the local method.

        Local SA is derivative-based and backend-agnostic, so -- unlike Sobol, which runs
        through the ``sobol_SA`` sampling manager -- it goes through a ``CVS0DParamID`` engine
        and its ``get_observable_sensitivities`` accessor, exactly as ``IdentifiabilityAnalysis``
        wraps the engine for the Hessian. ``do_ad`` is forced on so the analytic sensitivity
        path (CasADi jacobian / Myokit CVODES) is used.
        """
        if self._local_engine is not None:
            return self._local_engine
        from param_id.paramID import CVS0DParamID
        if self._inp_data_dict is None or self._obs_data_dict is None or self._params_for_id is None:
            raise RuntimeError(
                "Local SA needs the config, ground-truth data and params for id: build via "
                "init_from_dict and call set_ground_truth_data / set_params_for_id first.")
        inp = dict(self._inp_data_dict)
        inp['do_ad'] = True  # local SA is derivative-based
        engine = CVS0DParamID.init_from_dict(inp)
        engine.set_ground_truth_data(self._obs_data_dict)
        engine.set_params_for_id(self._params_for_id)
        self._local_engine = engine
        return engine

    def run_local_sensitivity(self, sa_options=None):
        """Run local (derivative-based) sensitivity analysis.

        Computes d(observable feature)/d(param) at the nominal parameter values -- the analytic,
        single-solve counterpart to the sampling-based Sobol SA -- via the backend-agnostic
        ``OpencorParamID.get_observable_sensitivities`` (CasADi jacobian for casadi_python;
        Myokit CVODES sensitivities for cellml_only + CVODE_myokit). On rank 0 it saves the
        absolute and relative sensitivity matrices to CSV. The result is also available via
        ``get_local_sensitivities()``.

        Args:
            sa_options: Optional options dict (see ``set_sa_options``).
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if sa_options is not None:
            self.set_sa_options(sa_options)

        engine = self._build_local_engine().param_id
        param_names = [n[0] if isinstance(n, list) else n
                       for n in engine.param_id_info["param_names"]]
        nominal = engine.sim_helper.get_init_param_vals(param_names)
        nominal = np.asarray(
            [float(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
             for v in nominal], dtype=float)

        sens = engine.get_observable_sensitivities(nominal)  # {obs_label: {param: d(feat)/dp}}

        output_names = list(sens.keys())
        n_out, n_par = len(output_names), len(param_names)
        absolute = np.zeros((n_out, n_par))
        relative = np.zeros((n_out, n_par))
        # Nominal feature magnitudes for the dimensionless (relative) normalisation.
        feat_mag = self._nominal_feature_magnitudes(engine, nominal, output_names)
        for i, oname in enumerate(output_names):
            fmag = feat_mag.get(oname, 0.0)
            for jj, pname in enumerate(param_names):
                d = float(sens[oname].get(pname, 0.0))
                absolute[i, jj] = d
                relative[i, jj] = d * abs(nominal[jj]) / fmag if fmag > 1e-30 else 0.0

        self.local_sensitivities = {
            'param_names': list(param_names),
            'output_names': output_names,
            'nominal_param_vals': list(nominal),
            'absolute': absolute,     # d(feature)/d(param)
            'relative': relative,     # dimensionless |p| * d(feature)/d(param) / |feature|
            'raw': sens,
        }
        if rank == 0:
            out_dir = self.SA_manager.output_dir
            print(f"{GREEN}Local sensitivity analysis completed successfully :){RESET}")
            print(f'{CYAN}saving results in {out_dir}{RESET}')
            for key in ('relative', 'absolute'):
                df = pd.DataFrame(self.local_sensitivities[key],
                                  index=output_names, columns=param_names)
                df.index.name = 'output'
                df.to_csv(os.path.join(out_dir, f'local_sensitivity_{key}.csv'))
        return self.local_sensitivities

    def _nominal_feature_magnitudes(self, engine, nominal, output_names):
        """|feature| at the nominal params, keyed by observable label, for relative scaling."""
        try:
            _, operands_list, _ = engine.get_cost_obs_and_pred_from_params(
                nominal, reset=True, only_one_exp=0)
            const = np.asarray(engine.get_obs_output_dict(operands_list[0])['const'], dtype=float)
            c2o = engine.obs_info["const_idx_to_obs_idx"]
            return {engine._observable_label(obs_i): abs(float(const[k]))
                    for k, obs_i in enumerate(c2o)}
        except Exception:
            return {name: 0.0 for name in output_names}

    def get_local_sensitivities(self):
        """Return the last local-sensitivity result (or None if not run yet).

        A dict with ``param_names``, ``output_names``, ``absolute`` and ``relative`` matrices
        ([n_output x n_param]) and the ``raw`` {observable: {param: d(feature)/d(param)}}. This
        is the accessor external tools (e.g. CUFLynx) read to obtain the local sensitivities.
        """
        return self.local_sensitivities


    def choose_most_impactful_params_sobol(self, top_n=5, index_type='ST', criterion='max', threshold=0.01, use_threshold=False):
        """
        Ranks and returns parameters based on Sobol indices.
        
        Args:
            top_n (int): Max number of parameters to return.
            index_type (str): 'ST' or 'S1'.
            criterion (str or func): 'max', 'mean', or custom lambda.
            threshold (float): Minimum score required. Only applied if use_threshold=True.
            use_threshold (bool): Whether to reject parameters below the threshold.
        """
        comm = MPI.COMM_WORLD
        if comm.Get_rank() != 0:
            return None

        indices_dict = self.SA_manager.load_sobol_indices()
        if not indices_dict or index_type.upper() not in indices_dict:
            print(f"{RED}ERROR: Index type '{index_type}' not found.{RESET}")
            return []

        data = indices_dict[index_type.upper()]
        
        # Flatten structure: {param_name: [val_out1, val_out2, ...]}
        param_scores_list = {}
        for out_name, params in data.items():
            for p_name, val in params.items():
                if p_name not in param_scores_list:
                    param_scores_list[p_name] = []
                param_scores_list[p_name].append(val)

        # Mapping criterion to calculation
        if criterion == 'max':
            calc_func = max
        elif criterion == 'mean':
            calc_func = lambda x: sum(x) / len(x)
        elif callable(criterion):
            calc_func = criterion
        else:
            print(f"{RED}ERROR: Invalid criterion '{criterion}'{RESET}")
            return []

        # 1. Calculate scores
        processed_scores = {p: calc_func(vals) for p, vals in param_scores_list.items()}

        # 2. Filter only if requested
        if use_threshold:
            filtered_data = {p: s for p, s in processed_scores.items() if s >= threshold}
            status_msg = f"filtered by threshold >= {threshold}"
        else:
            filtered_data = processed_scores
            status_msg = "unfiltered"

        # 3. Sort and select
        sorted_items = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        top_params = [item[0] for item in top_items]

        # 4. Final output
        if not top_params:
            print(f"{RED}No parameters found for criterion '{criterion}' ({status_msg}).{RESET}")
            return []

        print(f"{GREEN}Selected {len(top_params)} parameters (Criteria: {criterion}, Mode: {status_msg}):{RESET}")
        for i, (p, score) in enumerate(top_items):
            print(f"  {i+1}. {p:<35} | Score: {score:.4f}")
            
        return top_params

