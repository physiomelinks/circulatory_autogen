"""Converged reference solutions for the solver matrix (tests/test_solver_matrix.py).

``reference_solution`` runs CVODE_myokit twice, at rtol 1e-10 and 1e-11 (atol ten times
smaller for the second), and fails unless the two agree to ``REFERENCE_CONVERGED`` under
``shift_tolerant_max``. A reference is only as good as that check: a comparison against a
reference that is itself off says nothing about the backend under test.

Each reference is cached as ``.npz`` under ``$CUFLYNX_TEST_CACHE`` (default
``tests/test_outputs/references``). The file name is a hash of the flattened CellML, the run
settings, and the Myokit and SUNDIALS versions, so a new model, setting or library is a new
reference. CI keeps the folder between runs with actions/cache.
"""
import contextlib
import hashlib
import io
import json
import os

import numpy as np

from _solver_metrics import REFERENCE_CONVERGED, shift_tolerant_max, worst

CACHE_DIR = os.environ.get('CUFLYNX_TEST_CACHE') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'test_outputs', 'references')


def state_names(cellml_path):
    """The model's states as ``component/variable``, which every backend resolves."""
    import libcellml
    import libcuflynx.utilities.libcellml_helper_funcs as cellml
    with contextlib.redirect_stdout(io.StringIO()):
        model = cellml.parse_model(cellml_path, False)
        importer = cellml.resolve_imports(model, os.path.dirname(os.path.abspath(cellml_path)),
                                          False)
        flat = cellml.flatten_model(model, importer)
    analyser = libcellml.Analyser()
    analyser.analyseModel(flat)
    analysed = analyser.analyserModel()
    return [f'{analysed.state(i).variable().parent().name()}/{analysed.state(i).variable().name()}'
            for i in range(analysed.stateCount())]


def run_states(helper, names):
    """Run a helper and return its states as an array of shape (states, time points)."""
    with contextlib.redirect_stdout(io.StringIO()):
        assert helper.run(), 'the solver failed'
    values = helper.get_results([[n] for n in names], flatten=True)
    return np.asarray(values, dtype=float).reshape(len(names), -1)


def _myokit_versions():
    import myokit
    try:
        sundials = myokit.Sundials.version()
    except Exception:  # noqa: BLE001 - recorded, not needed
        sundials = 'unknown'
    return myokit.__version__, str(sundials)


def _flat_cellml_text(cellml_path):
    flat = cellml_path[:-len('.cellml')] + '_flat.cellml'
    with open(flat if os.path.exists(flat) else cellml_path, 'rb') as f:
        return f.read()


def _myokit_run(cellml_path, names, sim_time, dt, rtol, atol):
    from libcuflynx.solver_wrappers import get_simulation_helper
    with contextlib.redirect_stdout(io.StringIO()):
        helper = get_simulation_helper(
            model_path=cellml_path, solver='CVODE_myokit', model_type='cellml', dt=dt,
            sim_time=sim_time, pre_time=0.0,
            solver_info={'rtol': rtol, 'atol': atol, 'MaximumStep': dt / 10,
                         'MaximumNumberOfSteps': 10 ** 7})
    try:
        return run_states(helper, names)
    finally:
        helper.close_simulation()


def reference_key(cellml_path, sim_time, dt, atol):
    """The cache key: what the reference depends on."""
    myokit_version, sundials_version = _myokit_versions()
    digest = hashlib.sha256(_flat_cellml_text(cellml_path))
    digest.update(json.dumps([sim_time, dt, atol, myokit_version, sundials_version]).encode())
    return digest.hexdigest()[:20]


def reference_solution(cellml_path, sim_time, dt, atol):
    """(state names, converged states of shape (states, time points)) for a model."""
    names = state_names(cellml_path)
    stem = os.path.basename(cellml_path)[:-len('.cellml')]
    path = os.path.join(CACHE_DIR, f'{stem}-{reference_key(cellml_path, sim_time, dt, atol)}.npz')
    if os.path.exists(path):
        cached = np.load(path)
        if list(cached['names']) == names:
            return names, cached['states']
    coarse = _myokit_run(cellml_path, names, sim_time, dt, 1e-10, atol)
    fine = _myokit_run(cellml_path, names, sim_time, dt, 1e-11, atol / 10)
    error, state = worst(shift_tolerant_max(coarse, fine), names)
    assert error < REFERENCE_CONVERGED, (
        f'the {stem} reference is not converged: rtol 1e-10 and 1e-11 differ by {error:.2e} in '
        f'{state}. Change its atol in the MODELS table, or shorten the run.')
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.npz'
    np.savez(tmp, names=np.array(names), states=fine)
    os.replace(tmp, path)
    return names, fine
