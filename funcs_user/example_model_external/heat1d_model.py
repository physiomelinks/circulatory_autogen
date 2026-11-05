"""Example ``external_python`` model: the 1D heat equation on a fixed grid.

    dT/dt = k * d2T/dx2 ,   x in [0, 1],   T(0, t) = T(1, t) = u_D

solved with an explicit finite-difference scheme that this file owns. That is the whole point of
``model_type: external_python``: CA never sees a right-hand side, never calls an integrator, and
never chooses a step. It asks for a sample grid and gets one back.

Three fixed interior probes are exposed as observables, in the same way a real field solver would
report a handful of sensor locations rather than its whole state vector.

See README.md in this directory for the ``user_inputs.yaml`` settings, and
``src/solver_wrappers/external_simulation_helper.py`` for the contract this class implements.
"""
import numpy as np

# --- the spatial discretisation, fixed for this example -----------------------------------
_N_NODES = 21                                   # 20 intervals over a unit rod
_X = np.linspace(0.0, 1.0, _N_NODES)
_DX = _X[1] - _X[0]

# Interior nodes reported as observables. Deliberately not symmetric about the initial bump, so
# the three probes carry different information and a calibration can tell them apart.
_PROBE_NODES = {"heat/T_p1": 5, "heat/T_p2": 10, "heat/T_p3": 14}

# The explicit scheme is stable only for r = k*dt/dx^2 < 1/2. k is a calibrated parameter, so the
# sub-step count is derived from the *current* k at run time rather than baked in -- see
# _sub_steps. This is the margin it aims for.
_STABILITY_TARGET = 0.4


class Heat1D:
    """1D heat equation with an explicit FTCS scheme and its own time stepping."""

    # Self-description. LITERAL values on purpose: CA and downstream tools read these two
    # attributes straight out of this file by AST, without importing it, to build a parameter
    # table before any solver exists.
    parameters = {
        "heat/k": 0.4,      # thermal diffusivity
        "heat/u_D": 0.0,    # Dirichlet boundary value, applied at both ends
    }
    output_names = ["heat/T_p1", "heat/T_p2", "heat/T_p3"]

    # ---- required contract ----
    def init_solver(self, config):
        """One-off setup. Everything expensive belongs here, not in run()."""
        self.x = _X
        self.dx = _DX
        self._param_vals = dict(self.parameters)
        # config['solver_info']['user_config'] is free-form; this model uses it only to let a
        # caller loosen or tighten the explicit-scheme safety margin.
        user_config = (config.get('solver_info') or {}).get('user_config') or {}
        self.stability_target = float(user_config.get('stability_target', _STABILITY_TARGET))
        # Initial condition: an off-centre Gaussian bump, so the transient is worth looking at
        # and the three probes see it arrive at different times.
        self.initial_T = np.exp(-(((self.x - 0.35) / 0.12) ** 2))
        self.update_times(config['dt'], config['start_time'], config['sim_time'],
                          config['pre_time'])

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Record the sample grid CA wants. Cheap by contract -- no re-setup here."""
        self.dt = dt
        self.start_time = start_time
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.n_samples = int(pre_time / dt) + int(sim_time / dt) + 1
        self._history = None

    def set_param_vals(self, param_dict):
        """A subset of ``parameters`` with new values. No re-initialisation needed: k only
        changes the sub-step count, u_D only the boundary values."""
        self._param_vals.update({name: float(val) for name, val in param_dict.items()})

    def run(self):
        """March the whole grid, pre_time included, from the initial condition every time."""
        k = self._param_vals["heat/k"]
        u_D = self._param_vals["heat/u_D"]
        n_sub = self._sub_steps(k)
        dt_sub = self.dt / n_sub
        r = k * dt_sub / (self.dx ** 2)

        T = self.initial_T.copy()
        T[0] = T[-1] = u_D
        history = np.empty((self.n_samples, T.size))
        history[0] = T

        for sample in range(1, self.n_samples):
            for _ in range(n_sub):
                laplacian = T[2:] - 2.0 * T[1:-1] + T[:-2]
                T[1:-1] = T[1:-1] + r * laplacian
                T[0] = T[-1] = u_D
            if not np.all(np.isfinite(T)):
                # A diverged solve is reported, not raised: the calibration turns False into an
                # infinite cost and carries on with the next candidate.
                print(f"Heat1D diverged at sample {sample} (k={k}, r={r})")
                return False
            history[sample] = T

        self._history = history
        return True

    def get_results(self):
        """One 1D array per output name, on the grid update_times asked for, pre_time included."""
        if self._history is None:
            raise RuntimeError("Heat1D.get_results() called before a successful run()")
        return {name: self._history[:, node] for name, node in _PROBE_NODES.items()}

    # ---- optional contract ----
    def get_init_param_vals(self, names):
        """Current values, in the order asked for. Optional -- CA tracks these itself when a
        class does not implement it; implemented here to show the hook."""
        return [self._param_vals[name] for name in names]

    def reset(self):
        """Drop the last solve. run() rebuilds from the initial condition anyway, so this only
        keeps a stale field from being read back by mistake."""
        self._history = None

    def extra_plots(self):
        """A space-time map of the whole field -- the view CA cannot produce, because it only
        ever sees the three probes."""
        # Imported here, not at module level: a headless calibration run imports this file
        # thousands of times and never draws anything.
        import matplotlib.pyplot as plt

        if self._history is None:
            return []
        fig, ax = plt.subplots(figsize=(6, 4))
        t_end = self.start_time + (self.n_samples - 1) * self.dt
        image = ax.imshow(self._history.T, aspect='auto', origin='lower', cmap='inferno',
                          extent=[self.start_time, t_end, self.x[0], self.x[-1]])
        for name, node in _PROBE_NODES.items():
            ax.axhline(self.x[node], color='white', linewidth=0.8, linestyle='--')
            ax.text(t_end, self.x[node], ' ' + name.split('/')[-1], color='white',
                    va='center', fontsize=8)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('x')
        ax.set_title(f"1D heat equation, k={self._param_vals['heat/k']:.4g}")
        # 2 significant figures on the colorbar: the default formatter happily prints
        # float artefacts (0.30000000000000004) as tick labels.
        fig.colorbar(image, ax=ax, label='T', format='%.2g')
        fig.tight_layout()
        return [fig]

    # ---- internals ----
    def _sub_steps(self, k):
        """How many explicit sub-steps fit in one output step while keeping r below the
        stability limit. Derived from k because k is calibrated: a fixed sub-step count would be
        stable at the default k and blow up at the top of its range."""
        if k <= 0.0:
            return 1
        dt_stable = self.stability_target * (self.dx ** 2) / k
        return max(1, int(np.ceil(self.dt / dt_stable)))


# Required explicit registration: the class object, at module level.
SIM_HELPER = Heat1D
