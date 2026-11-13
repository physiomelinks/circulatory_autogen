"""Error metrics for comparing a solver's trajectories with a converged reference.

Every array is (states, time points). Each metric returns one value per state, divided by that
state's range over the reference, so states of very different sizes (pressures in Pa, volumes
in m^3) are compared on one scale. ``worst`` names the state behind the largest value.

The gates come from ``TOLERANCES``, one row per backend: a test reads its gate there rather
than writing its own literal, so loosening one backend is a visible, reviewed change.
"""
import numpy as np

# Gates for forward runs at the matrix's solver settings (adaptive methods at rtol 1e-8).
#   scaled_rms: the RMS error over the run. The gate for smooth trajectories.
#   shift_max:  the largest error once each point may move up to SHIFT_STEPS output steps in
#               time (see shift_tolerant_max). The gate on models with valves and events.
#   features:   the error in each state's mean, min and max over the run: what a calibration's
#               obs_data operations read.
TOLERANCES = {
    'CVODE_myokit':      {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    'CVODE_libopencor':  {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    'BE_batched_c':      {'scaled_rms': 1e-3, 'shift_max': 2e-2, 'features': 5e-3},
    'CVODES_batched_c':  {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    'solve_ivp':         {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    'casadi_integrator': {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    'aadc_semi_implicit': {'scaled_rms': 1e-2, 'shift_max': 5e-2, 'features': 1e-2},
    'cpp_CVODE':         {'scaled_rms': 1e-4, 'shift_max': 1e-3, 'features': 1e-3},
}

# How far (in output steps) shift_tolerant_max may move a point to meet the reference.
SHIFT_STEPS = 3

# A reference counts as converged when two tolerances 10x apart agree this closely.
REFERENCE_CONVERGED = 1e-6

# Two runs at the same parameters on a deterministic backend: equal to this, relative.
ROUND_TRIP = 1e-12


def _ranges(ref):
    ref = np.asarray(ref, dtype=float)
    span = np.ptp(ref, axis=1)
    size = np.max(np.abs(ref), axis=1)
    # A constant state is compared relative to its size; an all-zero one absolutely.
    return np.where(span > 1e-12 * np.maximum(size, 1e-300), span, np.where(size > 0, size, 1.0))


def scaled_rms(x, ref):
    """||x - ref||_2 / (sqrt(N) * range(ref)), per state."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    return np.sqrt(np.mean((x - ref) ** 2, axis=1)) / _ranges(ref)


def shift_tolerant_max(x, ref, k=SHIFT_STEPS):
    """max over t of (min over shifts |s| <= k steps of |x(t) - ref(t+s)|), per state, / range.

    A valve that opens one output step later than in the reference makes a plain maximum error
    as large as the jump. That measures the timing of one event, which a tolerance change
    moves, rather than the solution's accuracy, and it is why this metric exists.
    """
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    n = ref.shape[1]
    best = np.full(x.shape, np.inf)
    for s in range(-k, k + 1):
        lo, hi = max(0, -s), min(n, n - s)
        diff = np.full(x.shape, np.inf)
        diff[:, lo:hi] = np.abs(x[:, lo:hi] - ref[:, lo + s:hi + s])
        best = np.minimum(best, diff)
    return np.max(best, axis=1) / _ranges(ref)


def feature_error(x, ref):
    """The largest error in each state's mean, min and max over the run, / range."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    errors = [np.abs(f(x, axis=1) - f(ref, axis=1)) for f in (np.mean, np.min, np.max)]
    return np.max(errors, axis=0) / _ranges(ref)


def worst(errors, names):
    """(largest value, the name of its state)."""
    i = int(np.argmax(errors))
    return float(errors[i]), names[i]


def convergence_order(errors, steps):
    """The slope of log(error) against log(step): the observed order of a fixed-step method."""
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])
