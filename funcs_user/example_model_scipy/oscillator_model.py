"""Example ``external_python`` model: a damped linear oscillator, solved with scipy.

    x'' + c*x' + k*x = 0        state y = [x, v],  v = x'

with ``c`` (damping) and ``k`` (stiffness) as the calibrated parameters.

This is the *simple* end of ``model_type: external_python``. The sibling example
``funcs_user/example_model_external/`` marches a 1D heat equation with a scheme it wrote itself;
here there is no scheme to write, because the model is an ODE and ``scipy.integrate.solve_ivp``
already integrates ODEs. The class still owns the call -- that is the whole difference from the
retired ``python_user_defined`` type, which took ``rhs`` and made the call for you.

MIGRATING FROM ``python_user_defined``
--------------------------------------
This file is the retired ``funcs_user/example_model/oscillator_wrapper.py`` rewritten under the
surviving contract, so it doubles as the migration guide. The mapping is mechanical:

    PARAMETERS  dict           ->  the literal class attribute `parameters`
    STATES      dict           ->  the initial condition run() starts from (_INITIAL_STATE here)
    OUTPUT_NAMES list          ->  the literal class attribute `output_names`
    rhs(t, y, params)          ->  a method, handed to solve_ivp by your own run()
    compute_outputs(...)       ->  just another entry in the dict get_results() returns
    solver_info.method/rtol/.. ->  solver_info['user_config'], read in init_solver

and the bookkeeping CA used to do on your behalf is the six lines in ``update_times`` and
``run`` below: the sample grid is ``start_time + i*dt`` for ``i`` in ``0..N`` with
``N = int(pre_time/dt) + int(sim_time/dt)``, and ``get_results`` returns the whole of it,
pre_time samples included, because CA discards those itself.

See README.md in this directory for the ``user_inputs.yaml`` settings, and
``src/solver_wrappers/external_simulation_helper.py`` for the contract this class implements.
"""
import numpy as np
from scipy.integrate import solve_ivp

# Initial condition [x, v]: released from unit displacement, at rest. Fixed, not calibrated --
# exactly as the retired wrapper's STATES were.
_INITIAL_STATE = np.array([1.0, 0.0])

# Defaults for the free-form user_config options, so the class runs with none supplied.
_DEFAULT_METHOD = 'RK45'
_DEFAULT_RTOL = 1e-8
_DEFAULT_ATOL = 1e-8


class Oscillator:
    """A damped linear oscillator that integrates itself with scipy ``solve_ivp``."""

    # Self-description. LITERAL values on purpose: CA and downstream tools read these two
    # attributes straight out of this file by AST, without importing it, to build a parameter
    # table before any solver exists.
    parameters = {
        "oscillator/c": 0.5,    # damping
        "oscillator/k": 4.0,    # stiffness
    }
    # Two states and one algebraic quantity. Under the retired type the algebraic one needed a
    # separate `compute_outputs` hook; here it is just another array in the returned dict.
    output_names = ["oscillator/x", "oscillator/v", "oscillator/energy"]

    # ---- required contract ----
    def init_solver(self, config):
        """One-off setup. Nothing is expensive for an ODE this size, so this only reads the
        integrator options out of the free-form user_config -- which is where the retired type's
        ``solver_info.method`` / ``rtol`` / ``atol`` now live, since CA no longer chooses the
        integrator for a model it does not integrate."""
        self._param_vals = dict(self.parameters)
        user_config = (config.get('solver_info') or {}).get('user_config') or {}
        self.method = str(user_config.get('method', _DEFAULT_METHOD))
        self.rtol = float(user_config.get('rtol', _DEFAULT_RTOL))
        self.atol = float(user_config.get('atol', _DEFAULT_ATOL))
        self.update_times(config['dt'], config['start_time'], config['sim_time'],
                          config['pre_time'])

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Record the sample grid CA wants, and nothing else -- it is called on every
        sub-experiment, so it must stay cheap.

        This is the arithmetic CA used to do for a python_user_defined wrapper. Use exactly it,
        not ``np.arange(start, stop, dt)``: the wrapper checks the returned array lengths against
        this count, and a float-edge difference of one sample is a hard error.
        """
        self.dt = dt
        self.start_time = start_time
        self.sim_time = sim_time
        self.pre_time = pre_time
        self.n_samples = int(pre_time / dt) + int(sim_time / dt) + 1
        self.t_eval = start_time + np.arange(self.n_samples) * dt
        self._solution = None

    def set_param_vals(self, param_dict):
        """A subset of ``parameters`` with new values. Nothing to re-initialise: c and k are read
        fresh by the rhs on every call."""
        self._param_vals.update({name: float(val) for name, val in param_dict.items()})

    def run(self):
        """Integrate the whole grid, pre_time included, from the initial condition every time."""
        try:
            solution = solve_ivp(
                self._rhs,
                (self.t_eval[0], self.t_eval[-1]),
                y0=_INITIAL_STATE,
                t_eval=self.t_eval,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            # A diverged solve is reported, not raised: during a calibration this is an ordinary
            # event, and CA turns False into an infinite cost and tries the next candidate.
            print("Oscillator: solve_ivp raised at "
                  f"c={self._param_vals['oscillator/c']}, k={self._param_vals['oscillator/k']}")
            return False
        if not solution.success or not np.all(np.isfinite(solution.y)):
            print(f"Oscillator: solve_ivp failed ({solution.message}) at "
                  f"c={self._param_vals['oscillator/c']}, k={self._param_vals['oscillator/k']}")
            return False
        self._solution = solution
        return True

    def get_results(self):
        """One 1D array per output name, on the grid update_times asked for, pre_time included.

        CA slices the leading ``int(pre_time/dt)`` samples off itself, so returning them is not
        optional -- a short array raises rather than being padded.
        """
        if self._solution is None:
            raise RuntimeError("Oscillator.get_results() called before a successful run()")
        x, v = self._solution.y
        k = self._param_vals["oscillator/k"]
        return {
            "oscillator/x": x,
            "oscillator/v": v,
            # The algebraic output: total mechanical energy (unit mass). No separate hook needed.
            "oscillator/energy": 0.5 * (v ** 2 + k * x ** 2),
        }

    # ---- optional contract ----
    def get_init_param_vals(self, names):
        """Current values, in the order asked for. Optional -- CA tracks these itself when a
        class does not implement it; implemented here to show the hook."""
        return [self._param_vals[name] for name in names]

    def reset(self):
        """Drop the last solve. run() rebuilds from the initial condition anyway, so this only
        keeps a stale trajectory from being read back by mistake."""
        self._solution = None

    # ---- internals ----
    def _rhs(self, t, y):
        """dy/dt for y = [x, v]. This is the retired wrapper's module-level ``rhs``, now a
        method: the parameters come off self instead of being passed in."""
        x, v = y
        c = self._param_vals["oscillator/c"]
        k = self._param_vals["oscillator/k"]
        return [v, -c * v - k * x]


# Required explicit registration: the class object, at module level.
SIM_HELPER = Oscillator


if __name__ == '__main__':
    # Drives the class the way CA does, with no CA involved. This is how the values in
    # oscillator_obs_data.json were produced -- run it after changing the physics and paste the
    # numbers back in.
    TRUE_C, TRUE_K = 0.7, 5.0
    model = Oscillator()
    model.init_solver({'dt': 0.05, 'start_time': 0.0, 'sim_time': 10.0, 'pre_time': 0.0,
                       'solver_info': {}})
    model.set_param_vals({"oscillator/c": TRUE_C, "oscillator/k": TRUE_K})
    assert model.run() is True
    results = model.get_results()
    print(f"samples: {len(results['oscillator/x'])} (expected {model.n_samples})")
    print(f"at c={TRUE_C}, k={TRUE_K}:")
    print(f"  mean(x)          = {np.mean(results['oscillator/x']):.8f}")
    print(f"  min(x)           = {np.min(results['oscillator/x']):.8f}")
    print(f"  max_minus_min(v) = "
          f"{np.max(results['oscillator/v']) - np.min(results['oscillator/v']):.8f}")
    print(f"  mean(energy)     = {np.mean(results['oscillator/energy']):.8f}")
