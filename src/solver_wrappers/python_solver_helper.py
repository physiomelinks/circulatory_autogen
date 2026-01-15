import importlib.util
import numpy as np
from scipy.integrate import solve_ivp
import copy
import sys
DEBUG = False


class SimulationHelper:
    """
    SciPy-based solver for libCellML-generated Python modules.

    Matches the key interface of the OpenCOR SimulationHelper:
    - run()
    - update_times(dt, start_time, sim_time, pre_time)
    - get_results / get_all_results / get_all_variable_names
    - get_init_param_vals / set_param_vals
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.solver_info = solver_info or {}
        # pull optional SciPy solve_ivp settings; default method RK45
        # Check both 'solver' (new) and 'method' (legacy) for backward compatibility
        solver_method = self.solver_info.get('method')
        self.solve_ivp_method = solver_method 
        self.solve_ivp_kwargs = {
            k: v
            for k, v in self.solver_info.items()
            if k in ["rtol", "atol", "max_step", "vectorized", "dense_output", "jac"]
        }
        self._load_model()
        self._init_state()
        self.update_times(dt, 0.0, sim_time, pre_time)

    def set_solve_ivp_method(self, method):
        self.solve_ivp_method = method 
        
    # ---- setup helpers ----
    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        # maps
        self.state_name_to_idx = {info["name"]: idx for idx, info in enumerate(self.STATE_INFO)}
        self.var_name_to_idx = {info["name"]: idx for idx, info in enumerate(self.VARIABLE_INFO)}
        # inverse maps for lookup once we've resolved indices
        self.state_idx_to_name = {idx: info["name"] for idx, info in enumerate(self.STATE_INFO)}
        self.var_idx_to_name = {idx: info["name"] for idx, info in enumerate(self.VARIABLE_INFO)}

        # identify constants and algebraics
        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO) if info["type"].name == "ALGEBRAIC"]

    def _init_state(self):
        self.states = self.model.create_states_array()
        self.rates = self.model.create_states_array()
        self.variables = self.model.create_variables_array()
        self.model.initialise_variables(self.states, self.rates, self.variables)
        self.model.compute_computed_constants(self.variables)
        self._store_constants_defaults()

    def _store_constants_defaults(self):
        self.default_constants = [self.variables[i] for i in self.constant_indices]
        self.default_state_inits = [s for s in self.states]

    # ---- name resolution helpers ----
    def _resolve_name(self, name: str):
        """
        Resolve parameter/variable names that may contain prefixes (e.g., 'global/').
        Tries exact, then last path element, then slash-to-underscore form.
        """
        name = str(name).strip()
        candidates = [name]
        if "/" in name:
            parts = name.split("/")
            last = parts[-1]
            first = parts[0]
            candidates.append(last)
            candidates.append(name.replace("/", "_"))
            candidates.append(f"{first}_{last}")
            candidates.append(f"{last}_{first}")
            candidates.append(f"{first}{last}")
            candidates.append(name.replace("/", ""))  # drop slash
        # also try stripping any 'global_' prefix
        candidates += [c.replace("global_", "") for c in list(candidates)]
        for cand in candidates:
            if cand in self.state_name_to_idx:
                return ("state", self.state_name_to_idx[cand])
            if cand in self.var_name_to_idx:
                return ("var", self.var_name_to_idx[cand])
        return (None, None)

    # ---- timing helpers ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time/dt)
        self.n_steps = int(sim_time/dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt/2, dt)
        # stored portion excludes pre_time
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
    def get_init_param_vals(self, param_names):
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolve_name(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[idx])
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals):
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]

            def _to_list(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                try:
                    import numpy as _np
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                except Exception:
                    pass
                return [x]

            name_or_list = _to_list(name_or_list)
            vals = _to_list(vals)

            for name, val in zip(name_or_list, vals):
                kind, idx_res = self._resolve_name(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[idx_res] = val
                else:
                    raise ValueError(f"parameter name {name} not found in states or variables")
        self.model.compute_computed_constants(self.variables)

    # ---- simulation ----
    def _rhs(self, t, y):
        states = list(y)
        rates = [0.0]*self.STATE_COUNT
        # computed constants already in self.variables
        self.model.compute_rates(t, states, rates, self.variables)
        return rates

    def _post_process(self, sol):
        # store trajectories for requested variables
        self.state_traj = np.asarray(sol.y)
        self.time_full = sol.t
        self.var_traj = {}
        for name, idx in self.var_name_to_idx.items():
            self.var_traj[name] = []

        for ti, state_vec in zip(self.time_full, self.state_traj.T):
            states = list(state_vec)
            rates = [0.0]*self.STATE_COUNT
            vars_copy = copy.copy(self.variables)
            self.model.compute_rates(ti, states, rates, vars_copy)
            self.model.compute_variables(ti, states, rates, vars_copy)
            for name, idx in self.var_name_to_idx.items():
                val = vars_copy[idx]
                self.var_traj[name].append(val)

        # convert to numpy arrays
        for name in self.var_traj:
            self.var_traj[name] = np.asarray(self.var_traj[name])

    def run(self):
        # integrate
        solve_kwargs = dict(
            method=self.solve_ivp_method,
            t_eval=self.t_eval,
            **self.solve_ivp_kwargs,
        )
        if DEBUG:
            sys_stdout = sys.__stdout__
            try:
                if sys_stdout:
                    sys_stdout.write(f"[PY_SOLVE] start t0={self.t_eval[0]} t1={self.t_eval[-1]} dt={self.dt}\n")
                    sys_stdout.flush()
            except Exception:
                pass

        sol = solve_ivp(
            self._rhs,
            (self.t_eval[0], self.t_eval[-1]),
            y0=self.states,
            **solve_kwargs,
        )
        if not sol.success:
            if DEBUG:
                try:
                    sys_stdout.write(f"[PY_SOLVE] fail message={sol.message}\n")
                    sys_stdout.flush()
                except Exception:
                    pass
            return False
        self._post_process(sol)
        # update current state to final
        self.states = list(sol.y[:, -1])
        if DEBUG:
            try:
                sys_stdout.write(f"[PY_SOLVE] end t_last={sol.t[-1]} len={len(sol.t)}\n")
                sys_stdout.flush()
            except Exception:
                pass

        return True

    # ---- results ----
    def get_all_variable_names(self):
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        if name == 'time':
            return self.tSim
        if name in self.state_name_to_idx:
            data = self.state_traj[self.state_name_to_idx[name], :]
            return data[self.pre_steps:]
        if name in self.var_name_to_idx:
            data = self.var_traj[name]
            return data[self.pre_steps:]
        # attempt to resolve common alternative namings (e.g., "a/b" vs "a_b")
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            resolved_name = self.state_idx_to_name[idx_res]
            data = self.state_traj[idx_res, :]
            return data[self.pre_steps:]
        if kind == "var":
            resolved_name = self.var_idx_to_name[idx_res]
            data = self.var_traj[resolved_name]
            return data[self.pre_steps:]
        raise ValueError(f"variable {name} not found")

    def get_results(self, variables_list_of_lists, flatten=False):
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            row = [self._extract(name) for name in variables_list]
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    # ---- reset helpers ----
    def reset_and_clear(self, only_one_exp=-1):
        self._init_state()

    def reset_states(self):
        self.states = copy.copy(self.default_state_inits)
        self.model.compute_computed_constants(self.variables)

    def close_simulation(self):
        # no-op for scipy solver
        pass

