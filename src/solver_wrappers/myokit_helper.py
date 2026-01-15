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
        self.solver_info = solver_info or {}

        self.model = self._load_model(self.cellml_path)
        self.processed_cellml_path = getattr(self, '_last_processed_path', None)
        self.simulation = myokit.Simulation(self.model)

        # apply solver settings
        if "MaximumStep" in self.solver_info:
            try:
                self.simulation.set_max_step_size(self.solver_info["MaximumStep"])
            except Exception:
                pass
        rtol = self.solver_info.get("rtol", None)
        atol = self.solver_info.get("atol", None)
        if rtol is not None or atol is not None:
            self.simulation.set_tolerance(rtol if rtol is not None else 1e-8,
                                          atol if atol is not None else 1e-8)

        self._setup_time(dt, sim_time, pre_time)
        self._build_variable_maps()
        self._init_defaults()

        # Set initial values here AND in run() to ensure they stick
        # self._set_cellml_initial_values()
        self.last_log = None

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

            # Apply minimal Myokit-specific fixes to the flattened model
            model_string = self._apply_myokit_fixes(model_string)

            # Create temp file with the flattened model
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.cellml', delete=False) as f:
                prepared_path = f.name
            with open(prepared_path, 'w') as f:
                f.write(model_string)

            return prepared_path

        except ImportError:
            raise ImportError("libcellml not available, unable to prepare cellml for myokit")

    def _apply_myokit_fixes(self, model_string):
        """
        Apply fixes needed for Myokit compatibility:
        1. Resolve variable references in initial_value to numeric values by tracing connections# no longer done!
        2. Replace max functions with simplified expressions
        """
        import xml.etree.ElementTree as ET

        # Parse model XML (needed for any structured rewrites, e.g. max->piecewise)
        # Note: We avoid touching initial_value resolution here; that is handled
        # post-import in Myokit (see issue #898 strategy).
        try:
            model_start = model_string.find('<model')
            if model_start >= 0:
                model_content = model_string[model_start:]
            else:
                # libcellml print_model should always include <model>, but guard anyway
                model_content = model_string
            root = ET.fromstring(model_content)
        except ET.ParseError as e:
            # If parsing fails, skip structured rewrites and return original string.
            print(f"XML parsing failed in _apply_myokit_fixes: {e}. Returning unmodified model string.")
            return model_string

        # # Build connection map: variable_2 -> [possible variable_1 sources]
        # # Since multiple components can have the same variable name, we collect all possibilities
        # connections = {}  # var_2 -> list of (comp1, comp2, var_1)

        # # Find all connection blocks - try different formats
        # conn_blocks = (root.findall('.//{http://www.cellml.org/cellml/2.0#}connection') or
        #               root.findall('.//connection') or
        #               root.findall('.//{http://www.cellml.org/cellml/1.1#}connection'))

        # print(f"DEBUG: Found {len(conn_blocks)} connection blocks")

        # for conn_block in conn_blocks:
        #     # Get the components being connected - try different formats
        #     comp1 = conn_block.get('component_1')
        #     comp2 = conn_block.get('component_2')

        #     if not comp1 or not comp2:
        #         # Try the old format with map_components
        #         comp1_elem = (conn_block.find('.//{http://www.cellml.org/cellml/1.1#}map_components') or
        #                      conn_block.find('.//map_components'))
        #         if comp1_elem is not None:
        #             comp1 = comp1_elem.get('component_1')
        #             comp2 = comp1_elem.get('component_2')

        #     if comp1 and comp2:
        #         # Find all map_variables in this connection - try different formats
        #         map_vars = (conn_block.findall('.//{http://www.cellml.org/cellml/2.0#}map_variables') or
        #                    conn_block.findall('.//{http://www.cellml.org/cellml/1.1#}map_variables') or
        #                    conn_block.findall('.//map_variables'))

        #         for map_var in map_vars:
        #             var_1 = map_var.get('variable_1')  # source
        #             var_2 = map_var.get('variable_2')  # target

        #             if var_1 and var_2:
        #                 if var_2 not in connections:
        #                     connections[var_2] = []
        #                 connections[var_2].append((comp1, comp2, var_1))

        # print(f"DEBUG: Found connections for {len(connections)} variables")
        # for var_2, sources in list(connections.items())[:3]:
        #     print(f"DEBUG: {var_2} <- {sources}")

        # # Keep the working connection map (either namespace or non-namespace)

        # # Build component-to-variable mapping and parameter values
        # component_vars = {}  # component -> list of (var_name, initial_value)
        # param_values = {}    # var_name -> initial_value (for parameters with numeric values)
        # computed_values = {} # var_name -> computed initial value

        # # Try with namespace-aware search first
        # for comp in root.findall('.//{http://www.cellml.org/cellml/2.0#}component'):
        #     comp_name = comp.get('name')
        #     if comp_name:
        #         component_vars[comp_name] = []
        #         for var in comp.findall('.//{http://www.cellml.org/cellml/2.0#}variable'):
        #             var_name = var.get('name')
        #             init_val = var.get('initial_value')
        #             if var_name:
        #                 component_vars[comp_name].append((var_name, init_val))
        #                 if init_val:
        #                     try:
        #                         float(init_val)
        #                         param_values[var_name] = init_val
        #                     except ValueError:
        #                         pass

        # # Try to get computed values using libcellml analyser
        # try:
        #     import utilities.libcellml_helper_funcs as cellml
        #     from libcellml import Analyser, Generator, GeneratorProfile
        #     # parse the model in non-strict mode to allow non CellML 2.0 models
        #     model = cellml.parse_model(self.cellml_path, False)
        #     # resolve imports, in non-strict mode
        #     importer = cellml.resolve_imports(model, os.path.dirname(self.cellml_path), False)
        #     # need a flattened model for analysing
        #     flat_model = cellml.flatten_model(model, importer)
            
        #     analyser = Analyser()
        #     analyser.analyseModel(flat_model)
        #     analysed_model = analyser.model()

        #     if analysed_model:
        #         # Get all variables from the analysed model with component context
        #         var_info = []  # List of (var_name, component_name, initial_value)
        #         for i in range(analysed_model.variableCount()):
        #             analyser_var = analysed_model.variable(i)
        #             if analyser_var:
        #                 # Get the actual variable from the analyser variable
        #                 real_var = analyser_var.variable()
        #                 if real_var:
        #                     var_name = real_var.name()

        #                     component_name = None
        #                     # Try to get the component this variable belongs to
        #                     try:
        #                         parent = real_var.parent()
        #                         if parent:
        #                             component_name = parent.name()
        #                     except:
        #                         pass

        #                     var_info.append((var_name, component_name))

        #         print(f"DEBUG: Found {len(var_info)} variables with component info")
        #         for var_name, comp_name, init_val in var_info[:5]:  # Show first 5
        #             print(f"  {var_name} in {comp_name}: {init_val}")

        #         # Find computed constants (variables with initial values that are computed)
        #         for var_name, comp_name, init_val in var_info:
        #             if var_name and init_val:
        #                 try:
        #                     # If it's already a number, we already have it
        #                     float(init_val)
        #                 except ValueError:
        #                     # It's a computed value - try to evaluate it
        #                     # libcellml analyser should have evaluated these
        #                     computed_val = init_val
        #                     if computed_val and computed_val != var_name:  # Not a self-reference
        #                         try:
        #                             float(computed_val)  # Check if it's now a number
        #                             computed_values[var_name] = computed_val
        #                             print(f"Found computed value: {var_name} = {computed_val}")
        #                         except ValueError:
        #                             pass

        # except Exception as e:
        #     print(f"Could not evaluate computed constants: {e}")

        # # Add computed values to param_values
        # param_values.update(computed_values)

        # print(f"Found {len(connections)} connections and {len(param_values)} parameter values")
        # if connections:
        #     print("Sample connections:", dict(list(connections.items())[:3]))
        # if param_values:
        #     print("Sample param values:", dict(list(param_values.items())[:3]))

        # # Resolve initial value references by tracing connections
        # def resolve_variable_reference(var_name, visited=None, component_context=None):
        #     """Recursively resolve a variable reference to its numeric value"""
        #     if visited is None:
        #         visited = set()

        #     if var_name in visited:
        #         return None  # Circular reference

        #     visited.add(var_name)

        #     # Direct lookup in parameters
        #     if var_name in param_values:
        #         return param_values[var_name]

        #     # Check if it's a computed constant from libcellml
        #     if var_name in computed_values:
        #         return computed_values[var_name]

        #     # Trace through connections - now connections[var_name] is a list of (comp1, comp2, source_var)
        #     if var_name in connections:
        #         sources = connections[var_name]
        #         print(f"DEBUG: {var_name} has {len(sources)} possible sources: {sources}")

        #         # Try to find a source that matches the component context
        #         if component_context:
        #             for comp1, comp2, source_var in sources:
        #                 if component_context in comp1 or component_context in comp2 or component_context in source_var:
        #                     # Avoid self-reference
        #                     if source_var != var_name:
        #                         print(f"DEBUG: Resolving {var_name} -> {source_var} (from {comp1} to {comp2}) using context {component_context}")
        #                         return resolve_variable_reference(source_var, visited, component_context)

        #         # If no context match or no context provided, try the first non-self-referencing source
        #         for comp1, comp2, source_var in sources:
        #             if source_var != var_name:
        #                 print(f"DEBUG: Resolving {var_name} -> {source_var} (from {comp1} to {comp2})")
        #                 return resolve_variable_reference(source_var, visited, component_context)

        #     return None

        # # Create a component-to-variables mapping for context-aware resolution
        # component_vars = {}
        # for comp in root.findall('.//component') + root.findall('.//{http://www.cellml.org/cellml/2.0#}component'):
        #     comp_name = comp.get('name')
        #     if comp_name:
        #         component_vars[comp_name] = []
        #         for var in comp.findall('.//variable') + comp.findall('.//{http://www.cellml.org/cellml/2.0#}variable'):
        #             component_vars[comp_name].append(var)

        # # Update initial_value attributes that contain variable references with component context
        # resolved_count = 0

        # def resolve_with_context(var_name, component_name):
        #     """Resolve a variable reference with component context"""
        #     return resolve_variable_reference(var_name, component_context=component_name)

        # for comp_name, variables in component_vars.items():
        #     for var in variables:
        #         init_val = var.get('initial_value')
        #         var_name = var.get('name')

        #         if init_val and var_name:
        #             try:
        #                 float(init_val)
        #                 # Already numeric, skip
        #                 continue
        #             except ValueError:
        #                 # It's a variable reference, try to resolve it with component context
        #                 resolved = resolve_with_context(init_val, comp_name)
        #                 if resolved:
        #                     var.set('initial_value', resolved)
        #                     print(f"Resolved {comp_name}.{var_name}: {init_val} -> {resolved}")
        #                 else:
        #                     # Could not resolve - we need to compute the computed constant
        #                     computed_success = False

        #                     computed_success = self._compute_computed_constant(var_name, comp_name, flat_model, analysed_model, root, param_values, connections, resolve_variable_reference)
        #                     if computed_success:
        #                         var.set('initial_value', computed_success)
        #                         print(f"Computed {comp_name}.{var_name}: {computed_success}")
        #                     else:
        #                         print(f"ERROR: Could not compute computed constant {var_name} in {comp_name}")
        #                         raise ValueError(f"Could not compute computed constant {var_name} in {comp_name}")

        # Apply max function replacements to the XML - replace with piecewise
        print("Applying max function replacements with piecewise...")
        max_count = 0
        mathml_ns = '{http://www.w3.org/1998/Math/MathML}'

        # Find all max applications
        for apply_elem in root.findall(f'.//{mathml_ns}apply'):
            max_elem = apply_elem.find(f'{mathml_ns}max')
            if max_elem is not None:
                children = list(apply_elem)
                if len(children) >= 3:  # max, arg1, arg2
                    # Check if second argument is 0
                    second_arg = children[2]
                    if (second_arg.tag.endswith('cn') and
                        second_arg.text and second_arg.text.strip() == '0'):
                        # Extract first argument and units
                        first_arg = children[1]
                        units = second_arg.get('{http://www.cellml.org/cellml/2.0#}units', 'dimensionless')
                        
                        # Create piecewise structure
                        piecewise = ET.Element(f'{mathml_ns}piecewise')
                        
                        # Create piece element
                        piece = ET.SubElement(piecewise, f'{mathml_ns}piece')
                        # Copy first argument as the value
                        piece.append(ET.fromstring(ET.tostring(first_arg, encoding='unicode')))
                        
                        # Create condition: first_arg >= 0
                        condition = ET.SubElement(piece, f'{mathml_ns}apply')
                        geq = ET.SubElement(condition, f'{mathml_ns}geq')
                        # Copy first argument again for the condition (left side of >=)
                        first_arg_copy = ET.fromstring(ET.tostring(first_arg, encoding='unicode'))
                        condition.append(first_arg_copy)
                        # Right side: 0
                        cn_zero = ET.SubElement(condition, f'{mathml_ns}cn')
                        cn_zero.set('{http://www.cellml.org/cellml/2.0#}units', units)
                        cn_zero.text = '0'
                        
                        # Create otherwise element
                        otherwise = ET.SubElement(piecewise, f'{mathml_ns}otherwise')
                        cn_zero_otherwise = ET.SubElement(otherwise, f'{mathml_ns}cn')
                        cn_zero_otherwise.set('{http://www.cellml.org/cellml/2.0#}units', units)
                        cn_zero_otherwise.text = '0'
                        
                        # Replace the max apply element with piecewise in place
                        apply_elem.clear()
                        apply_elem.tag = piecewise.tag
                        apply_elem.text = piecewise.text
                        apply_elem.tail = piecewise.tail
                        # Copy all attributes
                        for key, value in piecewise.attrib.items():
                            apply_elem.set(key, value)
                        # Copy all children
                        for child in piecewise:
                            apply_elem.append(child)
                        
                        max_count += 1

        print(f"Replaced {max_count} max functions with piecewise")

        # Convert back to string
        model_string = ET.tostring(root, encoding='unicode', method='xml')

        return model_string

    def _setup_time(self, dt, sim_time, pre_time, start_time=0.0):
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self.tSim = np.linspace(start_time + pre_time, self.stop_time, self.n_steps + 1)

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
            self.last_log = self.simulation.run(self.sim_time, log=log,
                                                log_interval=self.dt)
        except Exception as e:
            print(f"Myokit simulation failed: {e}")
            return False
        return True

    def reset_and_clear(self, only_one_exp=-1):
        self.simulation.reset()
        self.last_log = None

    def reset_states(self):
        self.simulation.reset()
        # Re-apply any RHS changes to constants already set on the model

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
                    var = self.qname_to_var[qname]
                    # Set RHS to constant value
                    var.set_rhs(float(val))
                else:
                    raise ValueError(f"parameter {name} not found")
        # Update cached defaults for future resets
        self.default_states = list(self.simulation.state())

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
            if self.pre_steps > 0:
                # drop pre steps
                data = data[self.pre_steps:]
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
            parts = name.split("/")
            last = parts[-1]
            first = parts[0]
            candidates.append(last)
            candidates.append(name.replace("/", "_"))
            candidates.append(f"{first}.{last}")
            candidates.append(f"{first}_{last}")
        if "." in name:
            parts = name.split(".")
            last = parts[-1]
            first = parts[0]
            candidates.append(last)
            candidates.append(f"{first}_{last}")
            candidates.append(name.replace(".", "_"))
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
            param_cand = f'parameters.{name}'
            if param_cand in self.qname_to_var:
                var = self.qname_to_var[param_cand]
                if var.is_state():
                    return ("state", var.qname())
                return ("var", var.qname())

        return (None, None)

