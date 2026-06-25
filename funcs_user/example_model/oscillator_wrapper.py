"""
Example user-defined model wrapper: a damped linear oscillator.

    x'' + c*x' + k*x = 0

state vector y = [x, v] with v = x'. Parameters are the damping ``c`` and the
stiffness ``k``; the framework integrates ``rhs`` with scipy solve_ivp.

This demonstrates ``model_type: python_user_defined``. See README.md in this
directory for the user_inputs.yaml settings and how SA / calibration /
identifiability are run against it. Names use the canonical ``component/variable``
form ("oscillator/c" etc.) so they line up with oscillator_params_for_id.csv and
oscillator_obs_data.json.
"""

# Calibratable constants (overridden by oscillator_params_for_id.csv during a run).
PARAMETERS = {
    "oscillator/c": 0.5,   # damping
    "oscillator/k": 4.0,   # stiffness
}

# Initial conditions for [x, v].
STATES = {
    "oscillator/x": 1.0,   # initial displacement
    "oscillator/v": 0.0,   # initial velocity
}

# Observables referenced by the obs_data operands. Both are states here, so no
# compute_outputs() is needed (an algebraic example is shown commented out below).
OUTPUT_NAMES = ["oscillator/x", "oscillator/v"]


def rhs(t, y, params):
    x, v = y
    c = params["oscillator/c"]
    k = params["oscillator/k"]
    return [v, -c * v - k * x]


# Example of an algebraic output (total mechanical energy). Uncomment this and add
# "oscillator/energy" to OUTPUT_NAMES to expose it as an observable:
#
# def compute_outputs(t, y, params):
#     x, v = y
#     k = params["oscillator/k"]
#     return {"oscillator/energy": 0.5 * (v ** 2 + k * x ** 2)}
