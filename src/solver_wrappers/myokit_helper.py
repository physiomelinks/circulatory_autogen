import os
import copy
import numpy as np
import myokit
from myokit.formats import cellml as cellml_format
# src_dir = os.path.join(os.path.dirname(__file__), '..')
# sys.path.append(src_dir)
import utilities.libcellml_helper_funcs as cellml
import xml.etree.ElementTree as ET
import tempfile
import re
import os


class SimulationHelper:
    """
    Myokit-based solver wrapper matching the OpenCOR SimulationHelper interface.

    Key supported solver_info keys:
    - MaximumStep: mapped to Simulation.set_max_step_size
    - rtol / atol: mapped to Simulation.set_tolerance
    - method is ignored (kept for compatibility with other helpers)
    """

    def __init__(self, cellml_path, dt, sim_time, solver_info=None, pre_time=0.0):
        self.original_cellml_path = cellml_path
        self.cellml_path = cellml_path
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.protocol_info = None
        self.paced_parameter_qname = None
        self.solver_info = solver_info or {}

        self.model = self._load_model(self.cellml_path)
        self.processed_cellml_path = getattr(self, '_last_processed_path', None)
        self._recreate_simulation()

        if sim_time is not None and pre_time is not None:
            self._setup_time(dt, sim_time, pre_time)

        self._build_variable_maps()
        self._init_defaults()

        self.last_log = None
        self._last_results_dict = None

    def get_time(self, include_pre_time=False):
        print("TODO CHECK MYOKIT FUNCTIONALITY FOR TIME. I THINK IT IS DIFFERENT TO OPENCOR")
        if include_pre_time:
            return self.tSim
        else:
            return self.tSim - self.pre_time

    def set_protocol_info(self, protocol_info):
        """
        Store protocol metadata and (if needed) recreate Simulation with pace binding.
        """
        self.protocol_info = protocol_info
        paced_param_name = self._find_paced_parameter_name(protocol_info)
        if paced_param_name is None:
            return

        kind, qname = self._resolve_name(paced_param_name)
        if kind != "var":
            raise ValueError(
                f"Pacing parameter {paced_param_name} must resolve to a non-state variable"
            )

        pace_var = self.model.binding("pace")
        if pace_var is not None and pace_var.qname() != qname:
            pace_var.set_binding(None)

        target_var = self.qname_to_var[qname]
        target_var.set_binding("pace")
        self.paced_parameter_qname = qname

        # Myokit Simulation clones the model at construction time, so recreate
        # to ensure the new binding is present in the simulation model.
        self._recreate_simulation()

    def _load_model(self, path):
        """
        Load a CellML file via the Myokit CellML importer.
        """

        importer = cellml_format.CellMLImporter()
        # Prepare a temporary CellML that Myokit can import without touching the source file.
        prepared_path = self._prepare_cellml_for_myokit_libcellml(path)
        self._last_processed_path = prepared_path  # Store for later analysis
        try:
            model = importer.model(prepared_path)
            # Keep temp file for debugging
            print(f"Myokit import succeeded, temp file kept at: {prepared_path}")
        except Exception as e:
            # Keep temp file for debugging
            print(f"Myokit import failed, temp file kept at: {prepared_path}")
            raise e
        # Don't remove temp file for debugging
        try:
            self._apply_post_import_initial_expressions(model, prepared_path)
        except Exception as e:
            # Keep model usable even if this optional step fails; tests will surface issues.
            print(f"Warning: post-import initial expression handling failed: {e}")
        return model

    def _apply_post_import_initial_expressions(self, model, processed_cellml_path):
        """
        Convert CellML initial_value references into Myokit initial value expressions
        after import. This preserves dependency of state initial values on constants
        (Myokit #898 behavior), without pre-resolving values in XML.
        """
        if not processed_cellml_path or not os.path.exists(processed_cellml_path):
            return

        # Parse flattened CellML to find initial_value attributes
        try:
            tree = ET.parse(processed_cellml_path)
            root = tree.getroot()
        except Exception:
            return

        cellml_ns = "http://www.cellml.org/cellml/2.0#"
        # Build lookup: component -> set(varnames), and component.var -> initial_value string
        init_map = {}  # qname (comp.var) -> initial_value (string)
        for comp in root.findall(f".//{{{cellml_ns}}}component"):
            comp_name = comp.get("name")
            if not comp_name:
                continue
            for var in comp.findall(f".//{{{cellml_ns}}}variable"):
                vname = var.get("name")
                init_val = var.get("initial_value")
                if vname and init_val is not None:
                    init_map[f"{comp_name}.{vname}"] = init_val

        if not init_map:
            return

        # Map Myokit state qnames to their variables for fast lookup
        # Myokit states are named like "<component>_module.<var>" after CellML import.
        states = list(model.states())
        for s in states:
            qn = s.qname()
            # Translate Myokit component name back to CellML component name used in init_map
            # (strip the common "_module" suffix added by importer).
            if "." not in qn:
                continue
            comp_mod, var_name = qn.split(".", 1)
            comp = comp_mod[:-7] if comp_mod.endswith("_module") else comp_mod

            cellml_qn = f"{comp}.{var_name}"
            if cellml_qn not in init_map:
                continue

            init_val = init_map[cellml_qn].strip()
            # Only handle non-numeric initial values (variable references or expressions)
            try:
                float(init_val)
                continue
            except Exception:
                pass

            # If init_val is an unqualified identifier, qualify it within the same component
            # so Myokit parsing works (requires fully qualified names).
            expr = init_val
            if re.fullmatch(r"[A-Za-z_]\w*", init_val):
                expr = f"{comp_mod}.{init_val}"

            # Set initial value as expression string (Myokit will parse it)
            s.set_initial_value(expr)

    def _prepare_cellml_for_myokit_libcellml(self, path):
        """
        Prepare a CellML file for Myokit by flattening it using libcellml.
        This properly resolves imports and connections.
        """
        try:

            # Parse the model in non-strict mode to allow non CellML 2.0 models
            model = cellml.parse_model(path, False)

            # Resolve imports, in non-strict mode
            importer = cellml.resolve_imports(model, os.path.dirname(path), False)

            # Flatten the model to resolve all imports and connections
            flat_model = cellml.flatten_model(model, importer)

            # Print the flattened model to string
            model_string = cellml.print_model(flat_model)

            # Create temp file with the flattened model
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.cellml', delete=False) as f:
                prepared_path = f.name
            with open(prepared_path, 'w') as f:
                f.write(model_string)

            return prepared_path

        except ImportError:
            raise ImportError("libcellml not available, unable to prepare cellml for myokit")

    def _setup_time(self, dt, sim_time, pre_time, start_time=0.0):
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)

    def _recreate_simulation(self):
        self.simulation = myokit.Simulation(self.model)
        # Store baseline initial state used before any pre-simulation.
        self.original_state = self._get_simulation_model().initial_values(as_floats=True)

        # apply solver settings
        if "MaximumStep" in self.solver_info:
            try:
                self.simulation.set_max_step_size(self.solver_info["MaximumStep"])
            except Exception:
                pass
        rtol = self.solver_info.get("rtol", None)
        atol = self.solver_info.get("atol", None)
        if rtol is not None or atol is not None:
            self.simulation.set_tolerance(
                rtol if rtol is not None else 1e-8,
                atol if atol is not None else 1e-8,
            )
        self.last_log = None
        if hasattr(self, "all_vars"):
            self._init_defaults()

    def _get_simulation_model(self):
        # Myokit API differs across versions: prefer public model() when present.
        if hasattr(self.simulation, "model") and callable(getattr(self.simulation, "model")):
            return self.simulation.model()
        return self.simulation._model

    def _find_paced_parameter_name(self, protocol_info):
        if not isinstance(protocol_info, dict):
            return None
        params_to_change = protocol_info.get("params_to_change", {})
        if not isinstance(params_to_change, dict):
            return None

        for param_name, exp_values in params_to_change.items():
            if not isinstance(exp_values, list):
                continue
            for sub_values in exp_values:
                if not isinstance(sub_values, list):
                    continue
                for val in sub_values:
                    if isinstance(val, str):
                        return param_name
        return None

    def _build_variable_maps(self):
        # Qualified names for variables
        self.state_vars = self.model.states()
        self.state_qnames = [v.qname() for v in self.state_vars]
        self.state_index = {q: i for i, q in enumerate(self.state_qnames)}
        # include all variables (deep) for logging
        self.all_vars = list(self.model.variables(deep=True))
        self.all_qnames = [v.qname() for v in self.all_vars]
        # map for resolution
        self.qname_to_var = {v.qname(): v for v in self.all_vars}

    def _init_defaults(self):
        # default states
        self.default_states = list(self.simulation.state())
        # capture default values for all variables (best-effort)
        self.default_values = {}
        for var in self.all_vars:
            try:
                self.default_values[var.qname()] = var.eval()
            except Exception:
                # leave unset if evaluation fails
                pass

    # --------- core API ----------
    def run(self):
        try:
            if self.pre_time > 0:
                # Run a pre-simulation without logging
                self.simulation.pre(self.pre_time)
                # Note: pre-simulation may change state, but for pre_time=0, this doesn't run

            log = self._make_log()
            # Use explicit log times so the end-point is included.
            start_time = self.simulation.time()
            eps = 1e-12 # run for eps after the end time to make sure the final requested log point is emitted.
            self.last_log = self.simulation.run(
                self.sim_time+eps,
                log=log,
                log_times=self.tSim-self.pre_time,
            )
            # Restore exact endpoint (without epsilon overshoot) for continued runs.
            end_state = [float(np.asarray(self.last_log[qname])[-1]) for qname in self.state_qnames]
            self.simulation.set_state(end_state)
            self.simulation.set_time(start_time + self.sim_time)
        except Exception as e:
            print(f"Myokit simulation failed: {e}")
            return False
        return True

    def reset_and_clear(self, only_one_exp=-1):
        # Fully reset to baseline default state (before any pre-simulation).
        if self.last_log is not None:
            self._last_results_dict = self._collect_all_results_dict_from_log()
        self.simulation.set_default_state(self.original_state)
        self.simulation.reset()
        self.last_log = None

    def reset_states(self):
        # Reset to current default, then update state/default to reflect constants.
        if self.last_log is not None:
            self._last_results_dict = self._collect_all_results_dict_from_log()
        self.simulation.reset()
        updated_initial_state = self._get_simulation_model().initial_values(as_floats=True)
        self.simulation.set_state(updated_initial_state)
        self.simulation.set_default_state(updated_initial_state)
        self.default_states = list(updated_initial_state)
        self.last_log = None

    def get_all_variable_names(self):
        # Return variables that are actually logged (Myokit restrictions apply)
        if self.last_log is not None:
            return list(self.last_log.keys())
        else:
            # Return loggable variables only (non-constants)
            return [v.qname() for v in self.all_vars if not v.is_constant()]

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    def get_results(self, variables_list_of_lists, flatten=False):
        if self.last_log is None:
            raise RuntimeError("Simulation has not been run yet.")
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]

        results = []
        for variables_list in variables_list_of_lists:
            row = []
            for name in variables_list:
                row.append(self._extract(name))
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results
    
    def get_all_results_dict(self):
        if self.last_log is not None:
            self._last_results_dict = self._collect_all_results_dict_from_log()
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        if self._last_results_dict is not None:
            return {name: np.asarray(val).copy() for name, val in self._last_results_dict.items()}
        raise RuntimeError("Simulation has not been run yet.")

    def _collect_all_results_dict_from_log(self):
        results = {qname: np.asarray(self.last_log[qname]) for qname in self.last_log.keys()}
        # Keep a stable project-level time key regardless of importer-specific qnames.
        if "environment.time" not in results:
            results["environment.time"] = self._get_log_time_series()
        return results

    def _get_log_time_series(self):
        """
        Returns the logged time series from Myokit DataLog using its own time-key
        abstraction, with fallbacks for compatibility.
        """
        if self.last_log is None:
            raise RuntimeError("No log available")

        # Preferred: use DataLog's configured time key.
        time_key = None
        if hasattr(self.last_log, "time_key"):
            try:
                time_key = self.last_log.time_key()
            except Exception:
                time_key = None
        if time_key and time_key in self.last_log:
            return np.asarray(self.last_log[time_key])

        # Secondary: ask DataLog directly for time values.
        if hasattr(self.last_log, "time"):
            try:
                return np.asarray(self.last_log.time())
            except Exception:
                pass

        # Final fallback: model-declared bound time variable qname.
        model_time = self.model.time()
        if model_time is not None:
            model_time_qname = model_time.qname()
            if model_time_qname in self.last_log:
                return np.asarray(self.last_log[model_time_qname])

        raise RuntimeError("Unable to determine time series key from Myokit log.")

    def get_init_param_vals(self, param_names):
        param_init = []
        for name_or_list in param_names:
            names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
            vals = []
            for name in names:
                kind, qname = self._resolve_name(name)
                if kind == "state":
                    vals.append(self.default_states[self.state_index[qname]])
                elif kind == "var":
                    if qname in self.default_values:
                        vals.append(self.default_values[qname])
                    else:
                        # fallback to first log value if available later
                        vals.append(self._extract(qname)[0] if self.last_log else 0.0)
                else:
                    raise ValueError(f"parameter {name} not found")
            param_init.append(vals if len(vals) > 1 else vals[0])
        return param_init

    def set_param_vals(self, param_names, param_vals):
        for idx, name_or_list in enumerate(param_names):
            names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
            vals = param_vals[idx]
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            for name, val in zip(names, vals):
                kind, qname = self._resolve_name(name)

                
                if kind == "state":
                    self.simulation.set_state_value(self.state_index[qname], float(val))
                elif kind == "var":
                    # Set RHS to constant value
                    if isinstance(val, str):
                        trace_name = val
                        if self.paced_parameter_qname is None:
                            raise ValueError(
                                "Found string trace name in params_to_change, but no paced "
                                "parameter was configured in set_protocol_info."
                            )
                        if qname != self.paced_parameter_qname:
                            raise ValueError(
                                f"Trace name {trace_name} was provided for {qname}, but paced "
                                f"parameter is {self.paced_parameter_qname}."
                            )
                        # Validate protocol info exists  
                        if 'protocol_traces' not in self.protocol_info.keys():
                            raise ValueError("params_to_change entry is set to a string, Protocol traces not found in protocol info")
                        if trace_name not in self.protocol_info['protocol_traces'].keys():  
                            raise ValueError(f"params_to_change entry is set to a string, {trace_name}, Protocol trace '{trace_name}' not found")   
                        if 'values' not in self.protocol_info['protocol_traces'][trace_name].keys():
                            raise ValueError(f"params_to_change entry is set to a string, {trace_name}, Protocol trace '{trace_name}': values not found")

                        pace_time = self.protocol_info['protocol_traces'][trace_name]['t']
                        pace_values = self.protocol_info['protocol_traces'][trace_name]['values']

                        protocol = myokit.TimeSeriesProtocol(pace_time, pace_values)  
                        self.simulation.set_protocol(protocol, label='pace') 
                        
                    elif not isinstance(val, (float, np.float64, int)):
                        raise ValueError(f"Parameter value {val} is not a valid type. {type(val)}" + \
                                         "must be a float, np.float64, or int.")
                    else:
                        if self.paced_parameter_qname is not None and qname == self.paced_parameter_qname:
                            # If this variable is bound to "pace", set a constant protocol value
                            # for this segment instead of set_constant (which is invalid for bound vars).
                            pace_val = float(val)
                            duration = float(max(self.sim_time if self.sim_time is not None else 1.0, self.dt))
                            protocol = myokit.TimeSeriesProtocol(
                                [0.0, duration],
                                [pace_val, pace_val],
                            )
                            self.simulation.set_protocol(protocol, label='pace')
                        else:
                            self.simulation.set_constant(qname, float(val))
                else:
                    raise ValueError(f"parameter {name} not found")
        # Keep state defaults consistent with model-defined initial values.
        self.default_states = list(self._get_simulation_model().initial_values(as_floats=True))

    def modify_params_and_run_and_get_results(self, param_names, mod_factors, obs_names, absolute=False):
        if absolute:
            new_param_vals = mod_factors
        else:
            init_param_vals = self.get_init_param_vals(param_names)
            new_param_vals = [a * b for a, b in zip(init_param_vals, mod_factors)]

        self.set_param_vals(param_names, new_param_vals)
        success = self.run()
        if success:
            pred_obs_new = self.get_results(obs_names)
            self.reset_and_clear()
        else:
            raise RuntimeError("simulation failed")
        return pred_obs_new

    def update_times(self, dt, start_time, sim_time, pre_time):
        self._setup_time(dt, sim_time, pre_time, start_time=start_time)
        # Reset to ensure time/step changes are honored
        self.simulation.reset()

    def close_simulation(self):
        # Myokit doesn't require explicit close
        pass

    # --------- internals ----------
    def _make_log(self):
        # Create a list of loggable variable names (Myokit restrictions)
        # Myokit cannot log constants, only states and certain other variable types
        variables_to_log = []
        for v in self.all_vars:
            # Check if variable is loggable (not a constant)
            if not v.is_constant():
                variables_to_log.append(v.qname())
        return variables_to_log

    def _extract(self, name):
        if name == "time":
            if self.last_log is None:
                raise RuntimeError("No log available")
            time_key = self.model.time().qname()
            times = np.asarray(self.last_log[time_key])
            # Remove pre_time portion if present
            if self.pre_time > 0:
                times = times - self.pre_time
            return times
        kind, qname = self._resolve_name(name)
        if self.last_log and kind in ("state", "var"):
            data = np.asarray(self.last_log[qname])
            return data
        if kind == "state":
            return np.asarray([self.simulation.state()[self.state_index[qname]]])
        if kind == "var":
            if qname in self.default_values:
                return np.asarray([self.default_values[qname]])
        raise ValueError(f"variable {name} not found")

    def _resolve_name(self, name):
        name = str(name).strip()
        candidates = [name]
        if "/" in name:
            # this for cellml naming convention
            parts = name.split("/")
            last = parts[-1]
            first = parts[0]
            candidates.append(last)
            candidates.append(f"{first}.{last}")
            candidates.append(f"{first}_module.{last}")
        if "." in name:
            # this for myokit naming convention
            parts = name.split(".")
            last = parts[-1]
            first = parts[0]
            candidates.append(last)
        # try each candidate against qnames
        for cand in candidates:
            if cand in self.qname_to_var:
                var = self.qname_to_var[cand]
                if var.is_state():
                    return ("state", var.qname())
                return ("var", var.qname())
            # if candidate matches state name only
            if cand in self.state_qnames:
                return ("state", cand)

        # For parameters, also try with 'parameters.' prefix (common in flattened models)
        if not name.startswith('parameters.'):
            if name.startswith('global'):
                param_cand = f'parameters_global.{last}'
            else:
                param_cand = f'parameters.{last}_{first}'
            if param_cand in self.qname_to_var:
                var = self.qname_to_var[param_cand]
                if var.is_state():
                    return ("state", var.qname())
                return ("var", var.qname())

        return (None, None)

