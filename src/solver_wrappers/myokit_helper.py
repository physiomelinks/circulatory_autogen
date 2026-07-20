import os
import copy
import numpy as np
import myokit
from myokit.formats import cellml as cellml_format
# src_dir = os.path.join(os.path.dirname(__file__), '..')
# sys.path.append(src_dir)
import utilities.libcellml_helper_funcs as cellml
from solver_wrappers.name_resolver import VariableNameResolver
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

        # Forward sensitivity analysis (FSA / CVODES) config. Populated by enable_fsa();
        # when _fsa_enabled the Simulation is built with sensitivities and run() returns
        # (log, sensitivities). See enable_fsa/get_sensitivities.
        self._fsa_enabled = False
        self._fsa_dependent_qnames = None       # ordered myokit qnames of dependents
        self._fsa_dependent_specs = None        # myokit dependent spec strings
        self._fsa_independent_specs = None       # myokit independent spec strings (eligible only)
        self._fsa_eligible_param_names = None    # framework param names, order matches independents
        self._fsa_ineligible_param_names = None  # framework param names needing FD fallback
        self._fsa_independent_state_qnames = None  # state qnames that ARE FSA independents (init params)
        self._last_sensitivities = None
        # Per-sub-experiment sensitivities within one experiment: run() appends, reset_states()
        # clears. get_jac_cost_fsa reads this so multi-sub protocols keep each sub's dy/dp.
        self._fsa_sensitivities_history = []

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
        # Canonical cumulative time axis (see _extract("time")). tSim is built in
        # _setup_time as linspace(start_time + pre_time, stop_time); subtracting
        # pre_time gives the logged (post-pre) cumulative protocol time, matching
        # the OpenCOR backend.
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
        if kind == "state":
            raise ValueError(
                f"Pacing parameter {paced_param_name} must resolve to a non-state variable"
            )
        elif kind in [None, "None"]:
            raise ValueError(
                f"Pacing parameter {paced_param_name} must resolve to a valid variable",
                f"valid variables are: {self.all_qnames}"
            )
        elif kind == "var" or kind == "constant":
            pass
        else:
            raise ValueError(
                f"Pacing parameter {paced_param_name} must resolve to a valid kind, but got {kind}",
                f"valid kinds are: var, constant"
            )

        pace_var = self.model.binding("pace")
        if pace_var is not None and pace_var.qname() != qname:
            pace_var.set_binding(None)

        target_var = self.qname_to_var[qname]
        # Myokit forbids set_binding("pace") when this variable already has the pace
        # binding (common for CellML imports that map a driving variable to pace).
        if target_var.binding() != "pace":
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
        if self._fsa_enabled:
            # CVODES forward sensitivity: sensitivities=(dependents, independents) makes
            # run() return (log, sensitivities), where sensitivities[time][dep][indep] is
            # d(dependent)/d(independent). Only FSA-eligible params are independents; params
            # that appear in a state's initial-value expression raise NotImplementedError in
            # Myokit and are handled by finite differences instead (see enable_fsa).
            self.simulation = myokit.Simulation(
                self.model,
                sensitivities=(self._fsa_dependent_specs, self._fsa_independent_specs),
            )
        else:
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
        if (rtol is None and atol is None) and self._fsa_enabled:
            # CVODES forward sensitivities are only as accurate as the state integration;
            # the default tolerance leaves a noise floor that swamps small sensitivities,
            # so tighten it (unless the user set explicit tolerances) when FSA is active.
            rtol, atol = 1e-8, 1e-8
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
        self._offline_default_state = None
        self._state_overrides = {}
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

    def _describe_myokit_log_configuration(self):
        """Context for debugging empty logs or failed final-state extraction."""
        logical_lt = np.asarray(self.tSim, dtype=float) - float(self.pre_time)
        passed_lt = getattr(self, "_last_log_times_passed_to_myokit", None)
        n_st = getattr(self, "n_steps", None)
        lines = [
            "Time grid / logging context:",
            f"  sim_time={self.sim_time!r}, dt={self.dt!r}, pre_time={self.pre_time!r}",
            (
                "  segment_clock start_time="
                f"{getattr(self, 'start_time', None)!r} (cumulative timeline index for this subexperiment)"
            ),
            f"  n_steps = int(sim_time/dt) = {n_st!r}",
            f"  protocol log schedule linspace(start_time, start_time+sim_time): length={logical_lt.size}"
            + (
                f", min={float(logical_lt.min())!r}, max={float(logical_lt.max())!r}"
                if logical_lt.size
                else " (empty — no output points requested)"
            ),
        ]
        if passed_lt is not None and np.asarray(passed_lt).size > 0:
            pl = np.asarray(passed_lt, dtype=float)
            lines.append(
                f"  log_times passed to Myokit.run (aligned to sim time after reset+pre): "
                f"length={pl.size}, min={float(pl.min())!r}, max={float(pl.max())!r}"
            )
        lines.append(
            f"  duration passed to Simulation.run(): {self.sim_time + 1e-12!r} (sim_time + eps)"
        )
        if self.last_log is not None:
            time_key = None
            try:
                time_key = self.model.time().qname()
            except Exception:
                pass
            if time_key and time_key in self.last_log:
                tser = np.asarray(self.last_log[time_key])
                lines.append(
                    f"  logged time series ({time_key!r}): length={tser.size}"
                )
        return "\n".join(lines)

    def _primary_myokit_log_failure_cause(self):
        """
        One explanation derived from actual time settings / intervals (no speculative list).

        Preconditions: reads sim_time, dt, n_steps, tSim, pre_time, optionally
        _last_integration_interval set immediately before Simulation.run().
        """
        st = float(self.sim_time)
        dt = float(self.dt)
        n_steps = int(getattr(self, "n_steps", int(st / dt) if dt else 0))
        pre_t = float(self.pre_time)
        passed = getattr(self, "_last_log_times_passed_to_myokit", None)
        if passed is not None and np.asarray(passed).size > 0:
            lt = np.asarray(passed, dtype=float)
        else:
            lt = np.asarray(self.tSim, dtype=float) - pre_t
        iv = getattr(self, "_last_integration_interval", None)
        iv_start, iv_end = (iv[0], iv[1]) if iv is not None else (None, None)

        if st == 0:
            return (
                "Likely cause: sim_time is 0, so nothing is integrated and state logs have no samples."
            )
        if st < 0:
            return f"Likely cause: sim_time is negative ({st!r}), which is invalid for integration."
        if dt <= 0:
            return f"Likely cause: dt is not positive ({dt!r}), so the output time grid is invalid."

        if lt.size == 0:
            return (
                "Likely cause: computed log_times is empty (check sim_time and dt producing n_steps)."
            )

        if n_steps == 0 and st > 0:
            return (
                "Likely cause: sim_time is positive but smaller than dt, so "
                f"n_steps = int(sim_time/dt) is 0 (sim_time={st:g} s, dt={dt:g} s): "
                "only one output instant exists; Myokit produced no logged points there."
            )

        if iv_start is not None and iv_end is not None:
            lt_min = float(np.min(lt))
            lt_max = float(np.max(lt))
            tol = max(1e-9 * max(abs(iv_end), abs(iv_start), 1.0), 1e-12)
            if lt_max < iv_start - tol or lt_min > iv_end + tol:
                return (
                    "Likely cause: log_times do not overlap the integration interval "
                    f"(log_times in [{lt_min:g}, {lt_max:g}] s vs integration "
                    f"[{iv_start:g}, {iv_end:g}] s)."
                )

        return (
            "Likely cause: timing looks self-consistent but Myokit returned no logged samples "
            "at the requested log_times (solver/model issue — e.g. failure, stiffness, NaNs)."
        )

    def _validate_myokit_state_logs(self):
        """
        After Simulation.run(), every state should have a non-empty series at log_times.
        Empty series cause index -1 errors when restoring the endpoint state.
        """
        if self.last_log is None:
            raise RuntimeError(
                "Myokit Simulation.run returned no log (last_log is None).\n"
                + self._primary_myokit_log_failure_cause()
                + "\n"
                + self._describe_myokit_log_configuration()
            )
        empty = []
        missing = []
        for qname in self.state_qnames:
            if qname not in self.last_log:
                missing.append(qname)
                continue
            if np.asarray(self.last_log[qname]).size == 0:
                empty.append(qname)
        if missing or empty:
            parts = [
                "Myokit returned no logged samples for at least one state variable, "
                "so the final state cannot be read from the log.",
                self._primary_myokit_log_failure_cause(),
            ]
            if missing:
                parts.append(f"States missing from log: {missing}")
            if empty:
                parts.append(f"States with empty log series: {empty}")
            parts.append(self._describe_myokit_log_configuration())
            raise RuntimeError("\n".join(parts))

    # --------- core API ----------
    def run(self):
        try:
            if self.pre_time > 0:
                # Unlogged warm-up. Deliberately run(log=LOG_NONE) and NOT simulation.pre():
                # pre() resets every initial-value sensitivity row of _s_state to the identity
                # basis vector ("Reset to time 0", myokit cvodessim.pre()), throwing away the
                # d(state)/d(init) that CVODES accumulated across the warm-up. This warm-up is
                # part of *this* experiment and the calibrated initial value lives at its
                # start, so that sensitivity must propagate through it; otherwise a directly
                # calibrated state reports ~identity sensitivity when the true value has
                # decayed towards zero (the system forgets its IC), and the optimiser chases a
                # parameter that barely matters. run() integrates identically -- both advance
                # the clock through _run's `self._time += duration`, so the run_t0 offset below
                # is unaffected -- but leaves the sensitivities alone. pre() IS the correct call
                # for a genuine *offline* warm-up, which really does establish a new initial
                # condition; see issue #269.
                # (Return value discarded: with FSA on this is (log, sensitivities), and the
                # warm-up's own log/sensitivities are not wanted.)
                self.simulation.run(self.pre_time, log=myokit.LOG_NONE)

            log = self._make_log()
            # Use explicit log times so the end-point is included.
            start_time = self.simulation.time()
            eps = self.dt * 1e-9 # run for eps after the end time to make sure the final requested log point is emitted.
            # Logical output grid on the cumulative protocol timeline
            logical_log_times = np.asarray(self.tSim, dtype=float) - float(self.pre_time)
            if logical_log_times.size == 0:
                self._last_log_times_passed_to_myokit = None
                raise RuntimeError(
                    "Cannot run Myokit simulation: no sampling instants (log_times is empty).\n"
                    + self._primary_myokit_log_failure_cause()
                    + "\n"
                    + self._describe_myokit_log_configuration()
                )
            # update_times() calls simulation.reset(), so Myokit's clock starts at t=0 for every
            # segment. pre() advances to t==pre_time. Requested sampling instants must be
            # shifted from cumulative protocol times to simulator-absolute times.
            run_t0 = float(self.simulation.time())
            log_times = logical_log_times + (run_t0 - float(logical_log_times.flat[0]))
            self._last_log_times_passed_to_myokit = np.asarray(log_times, dtype=float).copy()
            duration = float(self.sim_time) + eps
            self._last_integration_interval = (run_t0, run_t0 + duration)
            run_out = self.simulation.run(
                self.sim_time+eps,
                log=log,
                log_times=log_times,
            )
            if self._fsa_enabled:
                # With sensitivities enabled, run() returns (log, sensitivities).
                self.last_log, self._last_sensitivities = run_out
                # Retain this sub-experiment's sensitivities; reset_states() (experiment start)
                # clears the list, so it holds exactly the current experiment's per-sub dy/dp in
                # sub order for get_jac_cost_fsa.
                self._fsa_sensitivities_history.append(np.asarray(self._last_sensitivities))
            else:
                self.last_log = run_out
            self._validate_myokit_state_logs()
            # Restore exact endpoint (without epsilon overshoot) for continued runs.
            end_state = [float(np.asarray(self.last_log[qname])[-1]) for qname in self.state_qnames]
            self.simulation.set_state(end_state)
            self.simulation.set_time(start_time + self.sim_time)
        except Exception as e:
            err = str(e)
            if "out of bounds" in err and "size 0" in err:
                print(
                    "Myokit simulation failed: tried to read the final logged state but a "
                    f"time series was empty (underlying error: {e}).\n"
                    + self._primary_myokit_log_failure_cause()
                    + "\n"
                    + self._describe_myokit_log_configuration()
                )
            else:
                print(f"Myokit simulation failed: {e}")
            return False
        return True

    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        """Run unlogged warmup once; use end state as default for reset_states().

        Currently UNUSED: the param-id path folds offline_pre_time into each experiment's
        first-sub warm-up instead, so it is re-integrated at the current parameter values on
        every evaluation. Freezing one state here made every evaluation start from the steady
        state of the *initial* parameter guess, and made the gradient drop the
        d(steady state)/d(p) term -- invisibly, since AD-vs-FD both perturb the same frozen
        state.

        Two things must be handled if a (correct) offline optimisation is reinstated on top of
        this method (tracked in issue #269):
          1. pre() resets initial-value sensitivity rows of _s_state to the identity. That is
             the RIGHT semantics here -- an offline warm-up really does establish a new initial
             condition -- but it also mutates _s_default_state, so the pristine save/restore
             deleted from reset_states() must be reinstated alongside it.
          2. The frozen state must be invalidated whenever the parameters move, or the same
             silent gradient bias returns.
        """
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self.simulation.reset()
        self.simulation.pre(offline_pre_time)
        new_state = list(self.simulation.state())
        self._offline_default_state = new_state
        self.simulation.set_default_state(new_state)
        self.default_states = list(new_state)
        self.simulation.set_state(new_state)
        self.simulation.set_time(0.0)
        self.last_log = None

    def reset_and_clear(self, only_one_exp=-1):
        # Fully reset to baseline default state (before any pre-simulation).
        if self.last_log is not None:
            self._last_results_dict = self._collect_all_results_dict_from_log()
        self.simulation.set_default_state(self.original_state)
        self.simulation.reset()
        self._state_overrides = {}
        self.last_log = None

    def reset_states(self):
        # Reset to offline pre-time state when configured, otherwise model ICs.
        if self.last_log is not None:
            self._last_results_dict = self._collect_all_results_dict_from_log()
        # New experiment: start a fresh per-sub sensitivity history (each experiment is
        # independent; sensitivities do not carry across experiment boundaries).
        self._fsa_sensitivities_history = []
        # reset() restores _s_state from _s_default_state. Nothing mutates _s_default_state any
        # more -- the warm-up in run() uses run(log=LOG_NONE), not pre(), precisely so the
        # initial-value sensitivity rows survive it -- so the default reset() restores is
        # already the pristine one built at construction. The explicit pristine save/restore
        # that used to live here existed only to undo pre()'s pollution and is now redundant.
        # NOTE: if an offline warm-up using pre() is ever reinstated, that pollution returns and
        # this restore must come back with it (see run_offline_pre_and_set_default_state).
        self.simulation.reset()
        if self._offline_default_state is not None:
            self.simulation.set_state(self._offline_default_state)
        else:
            updated_initial_state = self._get_simulation_model().initial_values(as_floats=True)
            self.simulation.set_state(updated_initial_state)
        for qname, val in self._state_overrides.items():
            if qname in self.state_index:
                self._set_state_value(qname, float(val))
        state = list(self.simulation.state())
        self.simulation.set_default_state(state)
        self.default_states = state
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

    def _set_state_value(self, qname, val):
        """Set a single state's current value. Myokit's Simulation exposes only set_state()
        (the full vector), so read-modify-write it. Does not disturb the CVODES sensitivity
        state (set_state leaves _s_state untouched)."""
        idx = self.state_index[qname]
        state = list(self.simulation.state())
        state[idx] = float(val)
        self.simulation.set_state(state)

    def set_param_vals(self, param_names, param_vals):
        # Phase 1: Pre-scan for any string trace value and rebind pace if the target
        # variable differs from the currently bound one.  This ensures set_constant
        # calls made later in the same invocation are not lost to a mid-loop recreate.
        new_paced_qname = self._find_required_paced_qname(param_names, param_vals)
        if new_paced_qname is not None and new_paced_qname != self.paced_parameter_qname:
            self._rebind_pace_to(new_paced_qname)

        # Phase 2: Apply all parameter values.
        for idx, name_or_list in enumerate(param_names):
            names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
            vals = param_vals[idx]
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            for name, val in zip(names, vals):
                kind, qname = self._resolve_name(name)

                if kind == "state":
                    self._state_overrides[qname] = float(val)
                    self._set_state_value(qname, float(val))
                    # FSA: a params_to_change entry that forces a state to a value makes that
                    # state momentarily independent of the calibration params, so its carried
                    # dy/dp column is stale -- zero it. Skip states that are themselves FSA
                    # independents (calibration state-init params): Myokit seeds those, and a
                    # reset_states() re-seed will restore them anyway.
                    if (self._fsa_enabled
                            and qname not in (self._fsa_independent_state_qnames or set())):
                        s = getattr(self.simulation, '_s_state', None)
                        if s is not None:
                            sidx = self.state_index[qname]
                            for row in s:
                                if sidx < len(row):
                                    row[sidx] = 0.0
                elif kind == "var":
                    if isinstance(val, str):
                        trace_name = val
                        # Validate protocol info exists
                        if self.protocol_info is None or 'protocol_traces' not in self.protocol_info:
                            raise ValueError(
                                "params_to_change entry is a string trace key, but protocol_traces "
                                "not found in protocol_info."
                            )
                        if trace_name not in self.protocol_info['protocol_traces']:
                            raise ValueError(
                                f"Protocol trace '{trace_name}' not found in protocol_traces."
                            )
                        trace = self.protocol_info['protocol_traces'][trace_name]
                        if 'values' not in trace:
                            raise ValueError(
                                f"Protocol trace '{trace_name}' is missing 'values' key."
                            )
                        # After Phase 1 rebind, paced_parameter_qname must match.
                        if qname != self.paced_parameter_qname:
                            raise RuntimeError(
                                f"Internal error: pace rebind should have set paced_parameter_qname "
                                f"to {qname}, but it is {self.paced_parameter_qname}."
                            )
                        protocol = myokit.TimeSeriesProtocol(trace['t'], trace['values'])
                        self.simulation.set_protocol(protocol, label='pace')

                    elif not isinstance(val, (float, np.float64, int)):
                        raise ValueError(
                            f"Parameter value {val} is not a valid type ({type(val)}); "
                            "must be float, np.float64, or int."
                        )
                    else:
                        if self.paced_parameter_qname is not None and qname == self.paced_parameter_qname:
                            # Variable is bound to "pace": use a flat TimeSeriesProtocol so the
                            # value is applied correctly rather than calling set_constant.
                            pace_val = float(val)
                            duration = float(max(self.sim_time if self.sim_time is not None else 1.0, self.dt))
                            protocol = myokit.TimeSeriesProtocol(
                                [0.0, duration],
                                [pace_val, pace_val],
                            )
                            self.simulation.set_protocol(protocol, label='pace')
                        else:
                            self.simulation.set_constant(qname, float(val))
                            # Keep self.model in sync with every set_constant call.
                            # _rebind_pace_to calls _recreate_simulation(), which clones
                            # self.model fresh (Myokit docs: "changes to the original model
                            # object will not affect the simulation"). Without this, any
                            # set_constant calls made *before* the rebind (e.g. ID params
                            # set at the start of each experiment) are lost when the second
                            # trace experiment triggers a rebind, causing the calibration
                            # and post-regeneration runs to use different constant values
                            # (template defaults vs. embedded best-fit values) — the root
                            # cause of the cost mismatch reported in issue #219.
                            try:
                                self.qname_to_var[qname].set_rhs(myokit.Number(float(val)))
                            except Exception:
                                pass
                else:
                    raise ValueError(f"parameter {name} not found")
        # Keep state defaults consistent with model-defined initial values.
        self.default_states = list(self._get_simulation_model().initial_values(as_floats=True))

    def _find_required_paced_qname(self, param_names, param_vals):
        """
        Scan param_names/param_vals for the first string (trace-key) value and return
        the resolved Myokit qname of that parameter, or None if none found.

        Only one paced variable per set_param_vals call is supported; if multiple
        string values are present for *different* variables, a ValueError is raised.
        """
        found_qname = None
        for idx, name_or_list in enumerate(param_names):
            names = name_or_list if isinstance(name_or_list, list) else [name_or_list]
            vals = param_vals[idx]
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            for name, val in zip(names, vals):
                if isinstance(val, str):
                    kind, qname = self._resolve_name(name)
                    if kind != "var":
                        raise ValueError(
                            f"Trace name '{val}' was given for '{name}', but it does not "
                            "resolve to a non-state variable."
                        )
                    if found_qname is not None and qname != found_qname:
                        raise ValueError(
                            f"Multiple different parameters have string trace values in the "
                            f"same set_param_vals call ({found_qname} and {qname}).  Myokit "
                            "supports only one paced variable per simulation segment."
                        )
                    found_qname = qname
        return found_qname

    def _rebind_pace_to(self, qname):
        """
        Dynamically rebind Myokit's 'pace' label to *qname*, preserving the current
        simulation state and time so that multi-experiment protocols with different
        paced variables work correctly.

        Steps:
          1. Save current simulation state and time.
          2. Unbind the previous 'pace' variable (if any) in the template model.
          3. Bind the new variable to 'pace' in the template model.
          4. Recreate the Myokit Simulation (which clones the model with the new binding).
          5. Restore the saved state and time.
        """
        current_state = self.simulation.state()
        current_time = self.simulation.time()

        # Unbind old pace variable from the template model.
        old_pace_var = self.model.binding("pace")
        if old_pace_var is not None:
            old_pace_var.set_binding(None)

        # Bind new variable.
        if qname not in self.qname_to_var:
            raise ValueError(
                f"Cannot bind pace to '{qname}': variable not found in model."
            )
        self.qname_to_var[qname].set_binding("pace")
        self.paced_parameter_qname = qname

        # Recreate Simulation with updated binding (clones the modified model).
        self._recreate_simulation()

        # Restore pre-rebind state and time.
        self.simulation.set_state(current_state)
        self.simulation.set_time(current_time)

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
        # Preserve the current state across reset() so multi-sub-experiment
        # protocols continue from the previous sub-experiment's end state, matching
        # the OpenCOR and solve_ivp backends. Myokit's reset() reverts to the
        # *default* (initial) state; run() only set_state()s the end state, not the
        # default state, so without this restore every sub-experiment after the
        # first would restart from the model initial conditions.
        current_state = list(self.simulation.state())
        # Same idea for the CVODES forward sensitivities. reset() zeros dy/dp (restores
        # _s_default_state); without carrying it, a later sub-experiment's sensitivities would
        # ignore how the parameters shaped the earlier sub's end state (which is this sub's
        # initial condition) -- the cross-sub chain-rule term. Myokit exposes no public setter,
        # so we save/restore the private _s_state (a list indexed [independent][state]). At an
        # experiment start reset_states() has just re-seeded _s_state, so the carried value is
        # the correct fresh identity/zero; at sub boundaries it is the previous sub's end dy/dp.
        carried_s_state = None
        if self._fsa_enabled:
            s = getattr(self.simulation, '_s_state', None)
            if s is not None:
                carried_s_state = [list(row) for row in s]
        self._setup_time(dt, sim_time, pre_time, start_time=start_time)
        # Reset to ensure time/step changes are honored
        self.simulation.reset()
        self.simulation.set_state(current_state)
        if carried_s_state is not None:
            self.simulation._s_state = carried_s_state

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
            # Return the canonical cumulative time axis (tSim), not Myokit's logged
            # clock. Myokit's clock restarts at each sub-experiment's run_t0
            # (update_times -> reset), so the raw log is a per-segment local time and
            # corrupts time-based observables (e.g. first_peak_time) in multi-sub-
            # experiment protocols. tSim is built from start_time/pre_time in
            # _setup_time and is the same length as the logged state/var series, so it
            # stays aligned with the values get_results returns. This is identical to
            # get_time() and to the OpenCOR backend's time axis.
            return self.tSim - self.pre_time
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
        """Resolve a name to (kind, qname) using the unified VariableNameResolver."""
        return VariableNameResolver.resolve_key(
            name,
            [("state", self.state_index), ("var", self.qname_to_var)],
            separator=".",
        )

    # ------------------------------------------------------------------
    # Forward sensitivity analysis (FSA / CVODES)
    # ------------------------------------------------------------------

    def enable_fsa(self, dependent_names, independent_param_names):
        """Rebuild the Simulation with CVODES forward sensitivities.

        dependent_names       : framework variable names of the observable operands
                                (e.g. 'aortic_root/u'), i.e. the traces whose gradient
                                w.r.t. parameters we need.
        independent_param_names : framework names of the parameters to differentiate.

        A parameter that appears inside a state's initial-value expression cannot be a
        CVODES independent (Myokit raises NotImplementedError); such parameters are
        classified as FSA-ineligible and returned so the caller can fall back to finite
        differences. Returns the list of ineligible parameter names.
        """
        dep_specs = []
        dep_qnames = []
        for name in dependent_names:
            _, qname = self._resolve_name(name)
            dep_specs.append(qname)
            dep_qnames.append(qname)

        cand_specs = []
        cand_names = []
        indep_state_qnames = set()
        for name in independent_param_names:
            kind, qname = self._resolve_name(name)
            # A directly-overridden state uses its initial value as the independent;
            # a constant uses the constant itself.
            spec = f'init({qname})' if kind == 'state' else qname
            if kind == 'state':
                indep_state_qnames.add(qname)
            cand_specs.append(spec)
            cand_names.append(name)
        # States that are calibration (FSA-independent) params: Myokit seeds their sensitivity
        # (identity at t=0), so a set_param_vals state override of these must NOT be treated as a
        # protocol re-seed (see set_param_vals). params_to_change state overrides are everything
        # else and DO get their sensitivity column zeroed.
        self._fsa_independent_state_qnames = indep_state_qnames

        eligible_specs, eligible_names, ineligible_names = self._classify_fsa_independents(
            dep_specs, cand_specs, cand_names)

        self._fsa_dependent_specs = dep_specs
        self._fsa_dependent_qnames = dep_qnames
        self._fsa_independent_specs = eligible_specs
        self._fsa_eligible_param_names = eligible_names
        self._fsa_ineligible_param_names = ineligible_names
        self._fsa_enabled = True

        # Rebuild the simulation with sensitivities and refresh derived maps/defaults.
        self._recreate_simulation()
        self._build_variable_maps()
        self._init_defaults()
        # The multi-sub sensitivity carry in update_times() reads and writes Myokit's private
        # CVODES sensitivity matrix, `Simulation._s_state` (a list indexed
        # [independent][state]); Myokit exposes no public setter for it. Fail loudly here if it
        # is missing rather than at the call sites, which all use getattr(..., None) and would
        # otherwise silently skip the carry -- dropping the cross-sub chain-rule term and
        # biasing every gradient low with no warning at all. A Myokit upgrade that renames or
        # removes this attribute must break the build, not the numbers.
        if not hasattr(self.simulation, '_s_state'):
            raise RuntimeError(
                "This Myokit build does not expose Simulation._s_state, the private CVODES "
                "sensitivity matrix that the multi-sub-experiment FSA gradient carries across "
                f"sub-experiment boundaries (myokit {getattr(myokit, '__version__', 'unknown')}). "
                "Verified present in 1.39.1 with layout [independent][state]. Without it the "
                "gradient would be silently wrong rather than absent, so FSA refuses to run. "
                "Either pin a Myokit that provides it, or use model_type 'casadi_python' "
                "(solver_info method 'bdf') for a gradient that does not depend on it.")
        return ineligible_names

    def _classify_fsa_independents(self, dep_specs, cand_specs, cand_names):
        """Split candidate independents into FSA-eligible/ineligible using Myokit itself.

        Tries to build one sensitivity Simulation with all candidates (a single compile
        when nothing is ineligible); only if that fails does it probe each parameter
        individually to find the offenders. This defers to Myokit's own rule for what may
        be a forward-sensitivity independent, rather than re-deriving it structurally.
        """
        if not cand_specs:
            return [], [], []
        try:
            myokit.Simulation(self.model, sensitivities=(dep_specs, list(cand_specs)))
            return list(cand_specs), list(cand_names), []
        except Exception:
            pass

        probe_dep = dep_specs[:1] if dep_specs else dep_specs
        eligible_specs, eligible_names, ineligible_names = [], [], []
        for spec, name in zip(cand_specs, cand_names):
            try:
                myokit.Simulation(self.model, sensitivities=(probe_dep, [spec]))
                eligible_specs.append(spec)
                eligible_names.append(name)
            except Exception:
                ineligible_names.append(name)
        return eligible_specs, eligible_names, ineligible_names

    def get_fsa_ineligible_params(self):
        """Framework names of AD params that fall back to finite differences (or None)."""
        return self._fsa_ineligible_param_names

    def get_sensitivities(self, dependent_names, param_names, sensitivities=None):
        """d(dependent_trace)/d(param) from a FSA run.

        Returns ``{dependent_name: {param_name: np.ndarray}}`` with each array aligned to
        the logged output grid (same length as get_results traces). Only FSA-eligible
        params are present in the inner dicts; ineligible params are omitted (the caller
        handles them by finite differences).

        ``sensitivities`` defaults to the last run's array; pass an explicit array (e.g. one
        retained per sub-experiment in ``_fsa_sensitivities_history``) to map that one instead.
        """
        if sensitivities is None:
            sensitivities = self._last_sensitivities
        if sensitivities is None:
            raise RuntimeError(
                "No sensitivities available: FSA is not enabled or run() has not been called.")
        sens = np.asarray(sensitivities, dtype=float)  # [n_time, n_dep, n_indep]
        dep_index = {q: i for i, q in enumerate(self._fsa_dependent_qnames)}
        indep_index = {n: i for i, n in enumerate(self._fsa_eligible_param_names)}
        out = {}
        for dname in dependent_names:
            _, dqname = self._resolve_name(dname)
            di = dep_index[dqname]
            out[dname] = {}
            for pname in param_names:
                if pname in indep_index:
                    out[dname][pname] = sens[:, di, indep_index[pname]].copy()
        return out

