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
#
# Each gate is about 10x the largest error measured on 2026-11-15 (in brackets: the worst model)
# over five runs: locally on SUNDIALS 7.7 (Myokit 1.39.1/casadi 3.7.2 and Myokit 1.40/casadi
# 3.8.2/libOpenCOR) and in CI on SUNDIALS 6.4.1 (test 3.12 solver-matrix, test-min-deps). SUNDIALS
# 6 vs 7 moved errors by up to 2x. A gate already under 10x the measured error is left as it was.
TOLERANCES = {
    # measured rms 6.5e-6 (3compartment), shift 1.0e-4 (3compartment), features 1.3e-5 (cvs_model_0d)
    'CVODE_myokit':      {'scaled_rms': 1e-5, 'shift_max': 1e-3, 'features': 1e-4},
    # measured rms 1.7e-6 (3compartment), shift 2.4e-5 (3compartment), features 3.6e-6 (cvs_model_0d)
    'CVODE_libopencor':  {'scaled_rms': 1e-5, 'shift_max': 5e-4, 'features': 5e-5},
    # trbdf2 at rtol 1e-6: rms 3.7e-4, shift 9.5e-4, features 5.0e-4 (all cvs_model_0d); cost 2.0e-5
    'BE_batched_c':      {'scaled_rms': 1e-3, 'shift_max': 1e-2, 'features': 5e-3},
    # measured rms 3.4e-6 (cvs_model_0d), shift 4.1e-5 (3compartment), features 8.4e-6 (cvs_model_0d)
    'CVODES_batched_c':  {'scaled_rms': 1e-5, 'shift_max': 5e-4, 'features': 1e-4},
    # Lotka_Volterra only: rms 7.2e-8, shift 2.2e-7, features 6.3e-8
    'solve_ivp':         {'scaled_rms': 1e-6, 'shift_max': 5e-6, 'features': 1e-6},
    # measured rms 4.4e-6 (3compartment), shift 4.2e-5 (3compartment), features 6.2e-6 (cvs_model_0d)
    'casadi_integrator': {'scaled_rms': 1e-5, 'shift_max': 5e-4, 'features': 1e-4},
    # bdf_newton, max_step 1e-4, measured locally only (no AADC in CI): rms 2.9e-4, shift 2.6e-3,
    # features 1.1e-4 (all 3compartment)
    'aadc_semi_implicit': {'scaled_rms': 5e-3, 'shift_max': 5e-2, 'features': 2e-3},
    # model_type cpp on 3compartment: rms 1.4e-5, shift 1.5e-4, features 5.5e-6
    'cpp_CVODE':         {'scaled_rms': 1e-4, 'shift_max': 1e-3, 'features': 1e-4},
}

# 3compartment's calibration cost against CVODE_myokit's, relative: BE_batched_c measured 2.0e-5,
# the other backends at most 2.7e-6 (CVODE_libopencor).
COST_TOLERANCE = {'BE_batched_c': 5e-4}
COST_TOLERANCE_DEFAULT = 5e-5

# control_phys_asd's features from casadi_python against a converged CVODE_myokit run: measured
# 1.1e-6.
DIVISION_GUARD = 2e-5

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
