"""The FEniCSx heat example: does the shipped ``external_python`` flagship actually work?

``funcs_user/heat_fenics/heat_fenics_model.py`` is the example users are told to copy, and it
is written against an API (dolfinx) that renames things between minor releases. So it needs
tests that run the real library rather than a stub -- which is what the ``test-fenics-emulator``
job in ``.github/workflows/tests.yml`` exists for: it installs ``fenics-dolfinx`` from
conda-forge and runs this file.

Six layers, in increasing order of how much of CA they involve:

* **the shipped obs_data** -- ``heat_fenics_obs_data.json`` names the run window its six
  values were computed on, in its ``protocol_info``. Plain JSON, so this one runs without
  dolfinx: it is a statement about the fixture, not about the solver.
* **smoke** -- load the file, set it up, run it, and check the record grid is exactly the
  length the contract promises. Also that a second ``run()`` at the same parameters is
  bit-identical, because a calibration reuses one instance for thousands of samples.
* **physics sanity** -- more diffusivity, faster relaxation towards the boundary conditions,
  asserted as a monotonicity over the part of the calibration box where the plate actually
  cools within the window (the saturated bottom end is excluded, and says why).
* **plots** -- ``extra_plots`` returns two Figures, headless.
* **emulator round trip** -- CA trains a surrogate against this model through the ordinary
  ``do_emulation`` path, and the surrogate agrees with the solver at a held-out theta. This
  is the first layer that touches CA's plumbing, so it is the one that would catch the
  external backend and this example drifting apart.
* **calibration through the emulator** -- the step after that: ``use_emulator`` makes an
  ordinary genetic-algorithm calibration evaluate the trained surrogate instead of the
  solver, and it has to finish with best-fit parameters inside the params_for_id box. It
  also pins the trap that path has already fallen into once (in the GUI): the emulator's
  fingerprint covers ``protocol_info``'s ``pre_times``/``sim_times``, so training on one
  timeline and calibrating on another is refused as stale rather than answered.

Everything that touches the solver skips cleanly without dolfinx (and the last two without
autoemulate), so the file collects on a machine with neither -- and the obs_data layer still
runs there.
"""
import importlib.util
import json
import os
import shutil
import time

import numpy as np
import pytest


# dolfinx is a conda-forge package that most CA environments do not have, so every test that
# actually solves something skips through this. It is *not* module level: the shipped obs_data
# is plain JSON, and the test that it names its window is worth running everywhere rather than
# only on the one CI job with FEniCSx installed.
def _require_dolfinx():
    return pytest.importorskip('dolfinx')


_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'heat_fenics'))
_MODEL_PATH = os.path.join(_EXAMPLE_DIR, 'heat_fenics_model.py')
_RESOURCE_FILES = ('heat_fenics_params_for_id.csv', 'heat_fenics_obs_data.json')
_OBS_DATA_PATH = os.path.join(_EXAMPLE_DIR, 'heat_fenics_obs_data.json')
_PARAMS_FOR_ID_PATH = os.path.join(_EXAMPLE_DIR, 'heat_fenics_params_for_id.csv')

#: A deliberately small grid for the tests: 50 steps on an 8x8 mesh is milliseconds once the
#: forms are compiled, and none of the assertions below need the shipped resolution. The
#: window tracks the calibration box -- at k in [0.001, 0.2] the plate's time constant runs
#: from ~51 s to ~0.25 s, so a 1 s window shows real cooling at the fast end without being
#: dominated by it at the slow end.
_FAST_CONFIG = {
    'dt': 0.02,
    'sim_time': 1.0,
    'pre_time': 0.0,
    'start_time': 0.0,
    'solver_info': {'user_config': {'nx': 8}},
}

#: How far above the uniform initial temperature a probe is allowed to end up. Backward Euler
#: is unconditionally stable but not monotone: with the consistent (non-lumped) P1 mass matrix
#: and k*dt well below h^2, the sharp boundary layer left by quenching a uniform plate produces
#: a small spatial over-/undershoot, and at the slow end of the box that is the entire signal.
#: Measured at ~0.009 on this 8x8 grid; 0.02 leaves room for a different mesh without letting
#: an actually-heating plate through.
_OVERSHOOT_TOL = 0.02


#: The window ``heat_fenics_obs_data.json``'s six values were computed on, and therefore the
#: window its ``protocol_info`` has to name. From the README's "Time scales" section: 100 steps
#: of 0.02 s, which at the default k = 0.05 is about two time constants of the plate.
_SHIPPED_WINDOW = {'pre_time': 0.0, 'sim_time': 2.0, 'dt': 0.02}


def _shipped_obs_data():
    """The whole shipped obs_data document: its ``protocol_info`` and its ``data_items``."""
    with open(_OBS_DATA_PATH) as handle:
        return json.load(handle)


def _shipped_obs_items():
    """The data_items the example actually ships, in file order.

    Read rather than counted in the assertions, so that adding or dropping an observable in
    ``heat_fenics_obs_data.json`` -- which has happened once already, from two items to six --
    does not leave a stale number behind here to fail on.
    """
    return _shipped_obs_data()['data_items']


def _load_model_class():
    """Load the example exactly the way CA's external backend does: by file path."""
    _require_dolfinx()
    spec = importlib.util.spec_from_file_location('heat_fenics_model_under_test', _MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, 'SIM_HELPER'), (
        f'{_MODEL_PATH} must expose SIM_HELPER -- that is how CA finds the solver class')
    return module.SIM_HELPER


@pytest.fixture(scope='module')
def model_class():
    return _load_model_class()


@pytest.fixture
def model(model_class):
    """A model set up on the fast grid, closed afterwards."""
    instance = model_class()
    instance.init_solver(dict(_FAST_CONFIG))
    yield instance
    instance.close()


def _expected_num_samples(config):
    """The contract's own arithmetic, spelled out so a mismatch names the rule it broke."""
    return int(config['pre_time'] / config['dt']) + int(config['sim_time'] / config['dt']) + 1


# ---------------------------------------------------------------------------------------
# The shipped obs_data. No dolfinx: this is a statement about a JSON file.
# ---------------------------------------------------------------------------------------
@pytest.mark.unit
def test_the_shipped_obs_data_carries_the_window_its_values_were_computed_on():
    """``heat_fenics_obs_data.json`` must name its own run window in ``protocol_info``.

    The six values in it are the model's output on the README's ``dt = 0.02`` /
    ``sim_time = 2.0`` grid -- ``min(T_p*)`` is *the temperature reached by the end of the
    window*, so read off a shorter one it is a different number entirely. Carrying the window
    in the file is what keeps the two from being separately editable.

    It is also the only place the CUFLynx GUI looks when it sizes a run, so an example with no
    ``protocol_info`` cannot be run from the GUI at all. (CA itself still accepts a bare list
    of data_items -- that is about the parser, and is pinned in
    ``tests/test_operation_kwargs.py``. This is about what the example ships.)

    Deliberately outside the dolfinx skip: nothing here solves anything, and this is exactly
    the assertion that must not go quiet on the machines without FEniCSx -- which is most of
    them.
    """
    shipped = _shipped_obs_data()

    protocol = shipped['protocol_info']
    # pre_times is per experiment; sim_times is per experiment, per subexperiment. One
    # experiment of one 2 s stretch, from t = 0, with no spin-up to discard.
    assert protocol['pre_times'] == [_SHIPPED_WINDOW['pre_time']]
    assert protocol['sim_times'] == [[_SHIPPED_WINDOW['sim_time']]]

    # And it still parses -- with no pre_time/sim_time offered, so the file has to be
    # self-sufficient rather than falling back on a yaml's window.
    from parsers.PrimitiveParsers import ObsAndParamDataParser

    parsed = ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=_OBS_DATA_PATH)
    assert parsed['protocol_info']['pre_times'] == [_SHIPPED_WINDOW['pre_time']]
    assert parsed['protocol_info']['sim_times'] == [[_SHIPPED_WINDOW['sim_time']]]
    assert len(parsed['gt_df']) == len(shipped['data_items'])
    # The window and dt divide into the whole number of steps the README quotes.
    assert _expected_num_samples(_SHIPPED_WINDOW) == 101


@pytest.mark.integration
@pytest.mark.slow
def test_the_example_declares_itself_without_being_imported(model_class):
    """``parameters`` and ``output_names`` must be literals a tool can read by AST.

    CA's tooling lists an external model's parameters *without importing it*, so that a
    machine with no dolfinx can still show the model in a form. A computed default would
    read as an arbitrary expression and the listing would come back empty.
    """
    import ast

    with open(_MODEL_PATH) as handle:
        tree = ast.parse(handle.read())

    literals = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name) and target.id in ('parameters',
                                                                      'output_names'):
                        # ast.literal_eval raises on anything that is not a plain literal.
                        literals[target.id] = ast.literal_eval(statement.value)

    assert literals.get('parameters') == model_class.parameters
    assert literals.get('output_names') == model_class.output_names
    assert set(model_class.parameters) == {'heat/k', 'heat/u_D'}
    assert model_class.output_names == ['heat/T_p1', 'heat/T_p2', 'heat/T_p3']


@pytest.mark.integration
@pytest.mark.slow
def test_smoke_run_produces_the_promised_record_grid(model):
    """One run, on the grid the contract defines, inside a wall-time a CI runner can afford."""
    start = time.perf_counter()
    assert model.run() is True, 'the reference run reported divergence'
    elapsed = time.perf_counter() - start
    # Generous on purpose: the point is to catch an accidental O(minutes) run (a forgotten
    # mesh refinement, a re-assembly per step), not to benchmark the runner. Measured well
    # under a second on an 8x8 mesh once the forms are compiled.
    assert elapsed < 30.0, f'a 50-step 8x8 solve took {elapsed:.1f} s'

    results = model.get_results()
    assert set(results) == set(model.output_names), (
        'get_results must be keyed by exactly the declared output_names')

    expected = _expected_num_samples(_FAST_CONFIG)
    for name, trace in results.items():
        assert isinstance(trace, np.ndarray), f'{name} must be a numpy array'
        assert trace.ndim == 1, f'{name} must be 1-D, got shape {trace.shape}'
        assert len(trace) == expected, (
            f'{name} has {len(trace)} samples; the contract says '
            f'int(pre_time/dt) + int(sim_time/dt) + 1 = {expected}')
        assert np.all(np.isfinite(trace))

    # The initial condition is a uniform plate, so every interior probe starts at
    # INITIAL_TEMP. If a probe were mislocated onto a boundary dof this is what would go
    # wrong -- it would start at a boundary value instead.
    for name in model.output_names:
        assert results[name][0] == pytest.approx(1.0, abs=1e-9), (
            f'{name} does not start at the uniform initial temperature; it is probably '
            f'sitting on a boundary dof')

    # p1 sits nearer the driven left edge than p3, so with u_D above the fixed edge
    # temperature it must run warmer. (Under the old symmetric bump the two were identical;
    # they are independent observables now, which is what the shipped obs_data relies on.)
    model.set_param_vals({'heat/u_D': 0.4})
    assert model.run() is True
    warmed = model.get_results()
    assert np.mean(warmed['heat/T_p1']) > np.mean(warmed['heat/T_p3']), (
        'the probe nearer the driven left edge is not warmer than the far one -- is u_D '
        'applied to the left edge only, or did the boundary split fail?')


@pytest.mark.integration
@pytest.mark.slow
def test_pre_time_samples_are_included(model):
    """``get_results`` returns the pre-time samples too; CA discards them, not the model."""
    model.update_times(dt=0.02, start_time=0.0, sim_time=1.0, pre_time=0.5)
    assert model.run() is True

    config = dict(_FAST_CONFIG, pre_time=0.5)
    expected = _expected_num_samples(config)
    trace = model.get_results()['heat/T_p2']
    assert len(trace) == expected, (
        f'with pre_time the grid must grow to {expected} samples, got {len(trace)}')
    assert len(model.get_time()) == expected


@pytest.mark.integration
@pytest.mark.slow
def test_runs_are_repeatable_and_parameters_need_no_reinit(model):
    """Two runs at the same parameters must be identical, and changing k must not re-init.

    A calibration calls ``set_param_vals`` then ``run`` thousands of times on one instance.
    If ``run`` did not restart from the initial condition, sample 500's cost would depend on
    sample 499's parameters -- an error that shows up as a calibration that "almost works".
    """
    model.set_param_vals({'heat/k': 0.05, 'heat/u_D': 0.0})
    assert model.run() is True
    first = model.get_results()['heat/T_p2'].copy()

    # A different parameter set in between, so a stale state would be visible.
    model.set_param_vals({'heat/k': 0.15})
    assert model.run() is True
    other = model.get_results()['heat/T_p2'].copy()
    assert not np.allclose(first, other), 'changing heat/k changed nothing -- is k in the form?'

    model.set_param_vals({'heat/k': 0.05, 'heat/u_D': 0.0})
    assert model.run() is True
    again = model.get_results()['heat/T_p2']
    assert again == pytest.approx(first, abs=1e-12), (
        'the same parameters gave a different trace, so run() is not restarting from the '
        'initial condition')


@pytest.mark.integration
@pytest.mark.slow
def test_an_unknown_parameter_is_rejected_by_name(model):
    with pytest.raises(ValueError, match='heat/k'):
        model.set_param_vals({'heat/not_a_parameter': 1.0})


@pytest.mark.integration
@pytest.mark.slow
def test_more_diffusivity_relaxes_faster_towards_the_boundary_value(model):
    """The physical statement, over the part of the box where it is the dominant effect.

    ``u_t = k Δu`` on a plate quenched through its boundary relaxes at a rate proportional
    to ``k``, so the final centre temperature must fall monotonically as ``k`` rises, from
    the uniform initial temperature towards the steady conduction profile. Stated as a
    monotonicity rather than as a value, so it holds on any mesh and any step size -- but
    only where there is cooling to order: see the note on the sweep below.
    """
    # The bottom of the shipped box (heat/k = 0.001) is deliberately left out. The model's own
    # docstring documents that region: below roughly k = 0.005 the plate barely cools on this
    # window -- at k = 0.001 it keeps ~96% of its heat -- so every observable saturates at the
    # initial temperature. Both k = 0.001 and k = 0.01 come back a hair *above* the initial 1.0
    # (measured 1.0080 and 1.0090 on this grid): the consistent P1 mass matrix over-/undershoots
    # slightly across the boundary layer, and at k*dt << h^2 that artefact is all there is to
    # see. So their ordering is set by discretisation error rather than by diffusivity, and the
    # difference the assertion is about is smaller than the error in measuring it. From k = 0.01
    # up, real cooling dominates and the ordering is physics again.
    sweep = (0.01, 0.05, 0.1, 0.2)  # the shipped calibration box, minus its saturated bottom
    model.set_param_vals({'heat/u_D': 0.0})
    finals = []
    for k in sweep:
        model.set_param_vals({'heat/k': k})
        assert model.run() is True, f'the solve diverged at heat/k = {k}'
        finals.append(float(model.get_results()['heat/T_p2'][-1]))

    assert all(later < earlier for earlier, later in zip(finals, finals[1:])), (
        f'final centre temperature should fall as k rises, got {finals} for k = {sweep}')
    # ... and every one of them is on its way down from the initial 1.0, towards the fixed
    # edge temperature of 0 (u_D is 0 here too, so every edge is at 0) -- "down from 1.0"
    # allowing the same small overshoot described above, which is why this is a tolerance and
    # not a bare `< 1.0`. The lower end carries no tolerance: nothing pulls the centre below
    # the coldest boundary value, so a negative final temperature is a real failure.
    assert finals[-1] > 0.0, (
        f'the fastest-diffusing case ended at {finals[-1]}, at or below the fixed edge '
        f'temperature of 0 that it is only ever approaching from above')
    assert max(finals) < 1.0 + _OVERSHOOT_TOL, (
        f'a final centre temperature of {max(finals)} is above the uniform initial 1.0 by more '
        f'than the {_OVERSHOOT_TOL} of discretisation overshoot this grid is allowed -- the '
        f'plate is being heated, not quenched')

    # The left edge is the only driven one, so raising u_D must raise where the field ends
    # up -- and it must do so without touching the other three edges.
    model.set_param_vals({'heat/k': 0.2, 'heat/u_D': 0.4})
    assert model.run() is True
    shifted = float(model.get_results()['heat/T_p2'][-1])
    assert shifted > finals[-1], (
        f'raising u_D from 0.0 to 0.4 left the final centre temperature at {shifted} '
        f'(was {finals[-1]}) -- is u_D actually applied on the left edge?')
    # A driven edge at u_D < 0 must pull the plate below the all-zero-boundary case, which
    # a boundary condition applied to the wrong facets (or to none) could not do.
    model.set_param_vals({'heat/u_D': -0.4})
    assert model.run() is True
    cooled = float(model.get_results()['heat/T_p2'][-1])
    assert cooled < finals[-1], (
        f'u_D = -0.4 gave a final centre temperature of {cooled}, not below the '
        f'{finals[-1]} of the all-zero boundary')


@pytest.mark.integration
@pytest.mark.slow
def test_extra_plots_returns_two_figures(model):
    """The optional hook CA surfaces in the GUI: two field snapshots, headless."""
    from matplotlib.figure import Figure

    assert model.run() is True
    figures = model.extra_plots()

    assert isinstance(figures, list)
    assert len(figures) == 2, f'expected the mid-time and final-time fields, got {figures}'
    for figure in figures:
        assert isinstance(figure, Figure)
        assert figure.axes, 'a returned Figure has nothing drawn on it'


@pytest.mark.integration
@pytest.mark.slow
def test_extra_plots_before_a_run_says_so(model):
    with pytest.raises(RuntimeError, match='run'):
        model.extra_plots()


# ---------------------------------------------------------------------------------------
# The emulator round trip: CA's own plumbing, end to end, against this model.
# ---------------------------------------------------------------------------------------

#: The timeline every emulator run in this file uses -- training *and* the calibration that
#: consumes the emulator, from one definition so the two cannot drift apart.
#:
#: They have to agree. ``emulators/emulator_bundle.py:fingerprint()`` hashes ``protocol_info``'s
#: ``pre_times`` and ``sim_times`` along with the parameters and the obs operations, so an
#: emulator trained at one ``sim_time`` and then used at another is (correctly) refused with
#: ``EmulatorQualityError: emulator is stale``. That is a real bug users have hit through the
#: GUI, and the point of stating the timeline once here is that this file cannot reproduce it
#: by accident -- it reproduces it deliberately, at the end of the calibration test.
#:
#: ``pre_time``/``sim_time`` here are not yaml keys any more: the obs_data's ``protocol_info``
#: is the run window, so ``_copy_resources`` writes them into the copy it hands these tests.
#: ``dt`` still is a yaml key.
#:
#: A coarse grid on a coarse mesh: 10 steps of an 8x8 problem per training sample, over the 1 s
#: window the k in [0.001, 0.2] box actually shows cooling in -- a third of the shipped window,
#: which none of the assertions below need. The features still move over the whole
#: params_for_id box, which is all the emulator needs, and a few dozen of them take seconds
#: rather than minutes.
_EMULATOR_TIMELINE = {'pre_time': 0.0, 'sim_time': 1.0, 'dt': 0.1}


def _copy_resources(temp_output_dir, sim_time=None, subdir='heat_fenics_resources'):
    """The example's CSV/JSON in a per-test directory, so a run never writes back into the
    repo's ``funcs_user/heat_fenics``.

    The copy's ``protocol_info`` is rewritten to ``sim_time`` on the way (default:
    ``_EMULATOR_TIMELINE``'s). The shipped file names the 2 s window its six values were
    computed on; these tests want the shorter one, and since the window now lives in the
    obs_data rather than in ``user_inputs.yaml``, wanting a different window means writing a
    different obs_data. That is also how the stale-emulator check below gets a *second*
    timeline to ask for -- a ``sim_time`` key in the config would no longer change anything.
    """
    if sim_time is None:
        sim_time = _EMULATOR_TIMELINE['sim_time']
    resources_dir = os.path.join(temp_output_dir, subdir)
    os.makedirs(resources_dir, exist_ok=True)
    for name in _RESOURCE_FILES:
        shutil.copy(os.path.join(_EXAMPLE_DIR, name), os.path.join(resources_dir, name))

    obs_path = os.path.join(resources_dir, 'heat_fenics_obs_data.json')
    obs_data = _shipped_obs_data()
    obs_data['protocol_info']['pre_times'] = [_EMULATOR_TIMELINE['pre_time']]
    obs_data['protocol_info']['sim_times'] = [[sim_time]]
    obs_data['protocol_info']['comment'] = (
        f'Test copy: the window has been shortened to {sim_time} s. The values below are the '
        f'shipped ones, computed on the 2 s window, so nothing here may assert that they are '
        f'recovered.')
    with open(obs_path, 'w') as handle:
        json.dump(obs_data, handle, indent=2)
    return resources_dir


def _emulator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """The ordinary ``do_emulation`` config, pointed at the FEniCSx example.

    Nothing here is special-cased for an external model beyond the three keys that name it
    (``model_type``, ``solver``, ``external_model_path``) -- which is the point.
    """
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    resources_dir = _copy_resources(temp_output_dir)
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'heat_fenics',
        # Never read for an external model (it is a generation-only path), but the parser
        # builds an absolute path from it unconditionally.
        'input_param_file': 'heat_fenics_parameters.csv',
        'model_type': 'external_python',
        'solver': 'external',
        'external_model_path': _MODEL_PATH,
        'resources_dir': resources_dir,
        'param_id_method': 'genetic_algorithm',
        # See _EMULATOR_TIMELINE: shared with the calibration that uses the emulator, because
        # the bundle's fingerprint covers it. pre_time/sim_time are the fallback the parser
        # only reaches for when an obs_data has no protocol_info -- this one has, written by
        # _copy_resources -- and are set to the same values so the two cannot disagree.
        **_EMULATOR_TIMELINE,
        'solver_info': {'user_config': {'nx': 8}},
        'DEBUG': False,
        'do_uq': False,
        'do_ia': False,
        'do_ad': False,
        'do_sensitivity': False,
        'plot_predictions': False,
        'model_out_names': ['heat/T_p1', 'heat/T_p2', 'heat/T_p3'],
        'param_id_obs_path': os.path.join(resources_dir, 'heat_fenics_obs_data.json'),
        'param_id_output_dir': temp_output_dir,
        'generated_models_dir': temp_generated_models_dir,
        'do_emulation': True,
        # RadialBasisFunctions rather than the full autoemulate search: fast, deterministic,
        # and the pick test_uq_on_emulator.py already settled on for a smooth response.
        # 32 Sobol samples over a two-parameter box: still seconds of solver time (each is a
        # 10-step 8x8 solve), and enough design for the six features the shipped obs_data now
        # asks for -- mean and min of each of the three probes -- rather than the two it used
        # to have.
        'emulator_settings': {'models': 'RadialBasisFunctions', 'num_train_samples': 32,
                              'sample_type': 'sobol', 'random_seed': 0, 'n_iter': 2,
                              'n_splits': 2},
    })
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


@pytest.mark.integration
@pytest.mark.slow
def test_the_obs_data_and_params_for_id_describe_this_model(model_class):
    """The shipped resource files must name parameters and outputs the model really has.

    A typo here is invisible until a calibration is half an hour in, and then surfaces as a
    KeyError from inside the cost function.
    """
    import csv

    with open(_PARAMS_FOR_ID_PATH) as handle:
        rows = [{key.strip(): (value or '').strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]
    names = [f'{row["vessel_name"]}/{row["param_name"]}' for row in rows]
    assert set(names) == set(model_class.parameters), (
        f'params_for_id names {names} do not match the model\'s '
        f'{sorted(model_class.parameters)}')
    for row in rows:
        assert float(row['min']) < float(row['max'])

    obs = _shipped_obs_items()
    assert obs, 'the obs_data file is empty'
    for item in obs:
        assert item['data_type'] == 'constant', (
            'the emulator round trip needs scalar features, so every data_item here must be '
            'data_type constant')
        for operand in item['operands']:
            assert operand in model_class.output_names, (
                f'obs_data operand {operand} is not one of {model_class.output_names}')


@pytest.mark.integration
@pytest.mark.slow
def test_an_emulator_trained_on_the_fenics_model_agrees_with_it(
        base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Train a surrogate through CA's ordinary path, then check it against the solver.

    This is the test that ties the three pieces together: the ``external_python`` backend
    must drive the FEniCSx model well enough that CA's feature extraction, its design over
    ``params_for_id`` and its emulator training all work unchanged. Any of them coming loose
    -- a permuted parameter column, features read off the wrong output, a model that quietly
    keeps state between samples -- moves the emulator off the solver, and none of them would
    show in the held-out score alone.

    The comparison is against the *solver's own* features at a held-out theta rather than
    against a hardcoded number, so it does not need to know what the right answer is.
    """
    _require_dolfinx()
    pytest.importorskip('autoemulate')

    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer

    config = _emulator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    trainer = EmulatorTrainer.init_from_dict(config)
    if trainer.rank != 0:
        trainer.train()
        return

    bundle = trainer.train()
    assert bundle is not None

    # One emulator output per shipped data_item, in the file's own order. Counted from the
    # JSON rather than written down, because that count has already changed once (two items
    # to six, when min was added alongside mean for each probe).
    expected_items = _shipped_obs_items()
    assert len(bundle.feature_labels) == len(expected_items), (
        f'expected one scalar feature per shipped data_item '
        f'({len(expected_items)}: {[item["variable"] for item in expected_items]}), got '
        f'{bundle.feature_labels}')
    # ... and each one names the operation and the output it was reduced from, in that order.
    # This is what makes the comparison below a real test: a feature vector permuted against
    # the obs_data would otherwise still be the right length and still be finite.
    for label, item in zip(bundle.feature_labels, expected_items):
        assert item['operation'] in label and item['operands'][0] in label, (
            f'emulator feature {label!r} does not correspond to data_item '
            f'{item["variable"]!r} ({item["operation"]} of {item["operands"]}) -- the feature '
            f'order has drifted from the obs_data order')

    mins = np.asarray(trainer.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(trainer.pid.param_id_info['param_maxs'], dtype=float)

    # A held-out point well inside the training box, where a surrogate fitted on a few dozen
    # samples is best supported. Not a design point: nothing here should be reading back a
    # memorised training target.
    theta = mins + 0.42 * (maxs - mins)

    predicted = np.asarray(bundle.predict(theta), dtype=float).reshape(-1)
    _, operands_list, _ = trainer.pid.get_cost_obs_and_pred_from_params(theta)
    simulated = np.asarray(trainer.pid.get_obs_output_dict(operands_list[0])['const'],
                           dtype=float).reshape(-1)

    assert np.all(np.isfinite(predicted))
    assert predicted.shape == simulated.shape
    assert predicted.size == len(expected_items)

    # Loose on purpose. Every feature lives on an O(1) scale (each probe starts at 1.0 and
    # relaxes towards a steady profile between u_D and 0, so both the means and the minima sit
    # inside [-0.5, 1.0]), and a few dozen Sobol samples over a two-parameter box is a small
    # design, so this is checking "the same function" rather than "a good emulator". A permuted
    # or unscaled feature vector misses by far more than this.
    assert predicted == pytest.approx(simulated, abs=0.15), (
        f'at heat/k={theta[0]:.4g}, heat/u_D={theta[1]:.4g} the emulator predicts '
        f'{predicted} but the FEniCSx solver gives {simulated}')


# ---------------------------------------------------------------------------------------
# One step further: an actual calibration, evaluated through the emulator.
# ---------------------------------------------------------------------------------------

def _calibration_config(trained_config):
    """The same problem again with ``use_emulator``, so the cost reads the surrogate.

    Derived from the very config the emulator was trained with, rather than rebuilt beside it.
    That is not tidiness: ``fingerprint()`` hashes the parameter bounds, the obs operations
    *and* ``protocol_info``'s ``pre_times``/``sim_times``, so any of them drifting between the
    two configs makes CA refuse the emulator as stale. Copying is how this test guarantees the
    only difference is the flag.
    """
    config = dict(trained_config)
    config.update({
        'do_emulation': False,
        'use_emulator': True,
        # DEBUG picks the documented quick-run GA population (28 per generation), and the call
        # budget then buys three generations of it. Every one of those ~84 evaluations is a
        # matrix multiply against the surrogate rather than a FEniCSx solve, which is the whole
        # reason a calibration through an emulator is affordable in CI.
        'DEBUG': True,
        'optimiser_options': {'num_calls_to_function': 90, 'cost_convergence': 1.0e-4,
                              'max_patience': 3, 'cost_type': 'gaussian_MLE'},
        # min_r2 None disables the held-out quality gate. Whether this particular fit is
        # accurate is the previous test's question, asked there against the solver itself; here
        # the emulator is a stand-in whose only job is to be evaluable, and a quality threshold
        # would turn a small-design R2 into a failure of the calibration path.
        'emulator_settings': dict(trained_config['emulator_settings'], min_r2=None),
    })
    return config


@pytest.mark.integration
@pytest.mark.slow
def test_a_calibration_through_the_emulator_completes_with_parameters_in_the_box(
        base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """Train a surrogate of the FEniCSx model, then calibrate on it -- the whole chain.

    The round trip above stops at "the emulator agrees with the solver". This goes the step a
    user actually takes next: ``use_emulator: true`` and an ordinary genetic-algorithm
    calibration, which must reach best-fit parameters without ever calling dolfinx again.

    What is asserted is that it *ran*, and that what it produced is a usable parameter vector:
    finite, one slot per params_for_id entry, inside the box the emulator was trained over.
    Deliberately not that it recovers a truth -- the shipped ``obs_data`` values are
    finite-difference estimates from a different discretisation, not this model's own output on
    this grid, so there is no theta that reproduces them and a recovery assertion would be
    measuring the fixture's provenance rather than CA.
    """
    _require_dolfinx()
    pytest.importorskip('autoemulate')

    from mpi4py import MPI

    from libcuflynx.emulators.emulator_bundle import EmulatorQualityError
    from libcuflynx.emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir
    from libcuflynx.param_id.paramID import CVS0DParamID

    comm = MPI.COMM_WORLD

    config = _emulator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    trainer = EmulatorTrainer.init_from_dict(config)
    trainer.train()          # every rank simulates its share; rank 0 writes the bundle
    if comm.Get_size() > 1:
        comm.Barrier()

    calibration = _calibration_config(config)
    # The run finds the emulator by resolving the same settings the trainer wrote it under --
    # nobody names a path twice -- so if that resolution ever diverged, this test would be
    # calibrating against a bundle it did not train.
    assert resolve_emulator_dir(calibration) == resolve_emulator_dir(config)
    assert os.path.isdir(resolve_emulator_dir(config)), 'the trainer wrote no emulator'

    pid = CVS0DParamID.init_from_dict(calibration)
    assert pid.param_id.emulates_features is True, (
        'use_emulator did not put the emulator behind the cost -- this calibration is running '
        'the FEniCSx solver, and proves nothing about the emulator path')

    pid.run()

    if comm.Get_rank() != 0:
        return

    mins = np.asarray(pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(pid.param_id_info['param_maxs'], dtype=float)
    best = np.asarray(pid.get_best_param_vals(), dtype=float).reshape(-1)

    assert best.size == mins.size == 2, (
        f'expected one best-fit value per params_for_id entry (heat/k, heat/u_D), got {best}')
    assert np.all(np.isfinite(best)), f'the calibration returned non-finite parameters: {best}'
    # Inside the box on both counts: the params_for_id bounds the GA searches, which are also
    # the box the emulator was trained over -- outside it the bundle would have refused to
    # predict at all rather than extrapolate.
    assert np.all(best >= mins) and np.all(best <= maxs), (
        f'best-fit parameters {best} left the params_for_id box '
        f'[{mins}, {maxs}] the emulator was trained over')
    assert np.isfinite(pid.param_id.best_cost), (
        f'the calibration finished with a non-finite cost: {pid.param_id.best_cost}')

    saved = np.load(os.path.join(pid.output_dir, 'best_param_vals.npy'))
    assert np.asarray(saved, dtype=float).reshape(-1) == pytest.approx(best), (
        'best_param_vals.npy disagrees with the parameters the run reports')

    # The trap this test exists to catch, pinned as a contract. emulator_bundle.fingerprint()
    # covers protocol_info's pre_times/sim_times, so the *same* emulator asked to serve a
    # different timeline must be refused -- not silently answered, which is what would make a
    # calibration at the wrong sim_time look like a successful one. (Found through the GUI:
    # train at one sim_time, calibrate at another, EmulatorQualityError. Everything above uses
    # _EMULATOR_TIMELINE for both halves precisely so it never happens by accident here.)
    #
    # The other timeline is a second obs_data naming a doubled window, because that is now the
    # only place a window is stated: a `sim_time` key in the config is the fallback the parser
    # never reaches for once a protocol_info exists, so perturbing it would perturb nothing and
    # this assertion would pass on an emulator that was never asked to be stale. The file name
    # is unchanged, so resolve_emulator_dir still points at the bundle just trained.
    # one_rank because only rank 0 gets here: CVS0DParamID's constructor barriers otherwise,
    # and a barrier one rank reaches alone is a hang, not a failure.
    stale_resources = _copy_resources(temp_output_dir,
                                      sim_time=2.0 * _EMULATOR_TIMELINE['sim_time'],
                                      subdir='heat_fenics_resources_other_window')
    stale = dict(calibration,
                 param_id_obs_path=os.path.join(stale_resources, 'heat_fenics_obs_data.json'),
                 one_rank=True)
    assert resolve_emulator_dir(stale) == resolve_emulator_dir(config), (
        'the stale check must reach the bundle just trained, or it proves nothing')
    with pytest.raises(EmulatorQualityError, match='stale'):
        CVS0DParamID.init_from_dict(stale)
