"""The FEniCSx heat example: does the shipped ``external_python`` flagship actually work?

``funcs_user/heat_fenics/heat_fenics_model.py`` is the example users are told to copy, and it
is written against an API (dolfinx) that renames things between minor releases. So it needs
tests that run the real library rather than a stub -- which is what the ``test-fenics-emulator``
job in ``.github/workflows/tests.yml`` exists for: it installs ``fenics-dolfinx`` from
conda-forge and runs this file.

Four layers, in increasing order of how much of CA they involve:

* **smoke** -- load the file, set it up, run it, and check the record grid is exactly the
  length the contract promises. Also that a second ``run()`` at the same parameters is
  bit-identical, because a calibration reuses one instance for thousands of samples.
* **physics sanity** -- a monotonicity that no amount of discretisation error can flip:
  more diffusivity, faster relaxation towards the boundary conditions.
* **plots** -- ``extra_plots`` returns two Figures, headless.
* **emulator round trip** -- CA trains a surrogate against this model through the ordinary
  ``do_emulation`` path, and the surrogate agrees with the solver at a held-out theta. This
  is the only layer that touches CA's plumbing, so it is the one that would catch the
  external backend and this example drifting apart.

Everything here skips cleanly without dolfinx (and the last one without autoemulate), so the
file collects on a machine with neither.
"""
import importlib.util
import json
import os
import shutil
import time

import numpy as np
import pytest

# dolfinx is a conda-forge package that most CA environments do not have. Nothing below this
# line can be imported without it, so the whole file skips rather than erroring at collection.
dolfinx = pytest.importorskip('dolfinx')


_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'funcs_user', 'heat_fenics'))
_MODEL_PATH = os.path.join(_EXAMPLE_DIR, 'heat_fenics_model.py')
_RESOURCE_FILES = ('heat_fenics_params_for_id.csv', 'heat_fenics_obs_data.json')

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


def _load_model_class():
    """Load the example exactly the way CA's external backend does: by file path."""
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
    """The one physical statement no discretisation error can flip.

    ``u_t = k Δu`` on a plate quenched through its boundary relaxes at a rate proportional
    to ``k``, so the final centre temperature must fall monotonically as ``k`` rises, from
    the uniform initial temperature towards the steady conduction profile. Stated as a
    monotonicity over the calibration box rather than as a value, so it holds on any mesh
    and any step size.
    """
    sweep = (0.001, 0.01, 0.05, 0.1, 0.2)  # the shipped calibration box for heat/k
    model.set_param_vals({'heat/u_D': 0.0})
    finals = []
    for k in sweep:
        model.set_param_vals({'heat/k': k})
        assert model.run() is True, f'the solve diverged at heat/k = {k}'
        finals.append(float(model.get_results()['heat/T_p2'][-1]))

    assert all(later < earlier for earlier, later in zip(finals, finals[1:])), (
        f'final centre temperature should fall as k rises, got {finals} for k = {sweep}')
    # ... and every one of them is on its way down from the initial 1.0, towards the fixed
    # edge temperature of 0 (u_D is 0 here too, so every edge is at 0).
    assert 0.0 < finals[-1] < finals[0] < 1.0

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

def _copy_resources(temp_output_dir):
    """The example's CSV/JSON in a per-test directory, so a run never writes back into the
    repo's ``funcs_user/heat_fenics``."""
    resources_dir = os.path.join(temp_output_dir, 'heat_fenics_resources')
    os.makedirs(resources_dir, exist_ok=True)
    for name in _RESOURCE_FILES:
        shutil.copy(os.path.join(_EXAMPLE_DIR, name), os.path.join(resources_dir, name))
    return resources_dir


def _emulator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """The ordinary ``do_emulation`` config, pointed at the FEniCSx example.

    Nothing here is special-cased for an external model beyond the three keys that name it
    (``model_type``, ``solver``, ``external_model_path``) -- which is the point.
    """
    from parsers.PrimitiveParsers import YamlFileParser

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
        # A coarse grid on a coarse mesh: 10 steps of an 8x8 problem per training sample,
        # over the 1 s window the k in [0.001, 0.2] box actually shows cooling in.
        # The features still move over the whole params_for_id box, which is all the
        # emulator needs, and 20 of them take seconds rather than minutes.
        'pre_time': 0.0,
        'sim_time': 1.0,
        'dt': 0.1,
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
        'emulator_settings': {'models': 'RadialBasisFunctions', 'num_train_samples': 20,
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

    with open(os.path.join(_EXAMPLE_DIR, 'heat_fenics_params_for_id.csv')) as handle:
        rows = [{key.strip(): (value or '').strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]
    names = [f'{row["vessel_name"]}/{row["param_name"]}' for row in rows]
    assert set(names) == set(model_class.parameters), (
        f'params_for_id names {names} do not match the model\'s '
        f'{sorted(model_class.parameters)}')
    for row in rows:
        assert float(row['min']) < float(row['max'])

    with open(os.path.join(_EXAMPLE_DIR, 'heat_fenics_obs_data.json')) as handle:
        obs = json.load(handle)
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
    pytest.importorskip('autoemulate')

    from emulators.emulator_trainer import EmulatorTrainer

    config = _emulator_config(base_user_inputs, temp_output_dir, temp_generated_models_dir)
    trainer = EmulatorTrainer.init_from_dict(config)
    if trainer.rank != 0:
        trainer.train()
        return

    bundle = trainer.train()
    assert bundle is not None
    assert len(bundle.feature_labels) == 2, (
        f'expected the two scalar observables, got {bundle.feature_labels}')

    mins = np.asarray(trainer.pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(trainer.pid.param_id_info['param_maxs'], dtype=float)

    # A held-out point well inside the training box, where a surrogate fitted on 20 samples
    # is best supported. Not a design point: nothing here should be reading back a memorised
    # training target.
    theta = mins + 0.42 * (maxs - mins)

    predicted = np.asarray(bundle.predict(theta), dtype=float).reshape(-1)
    _, operands_list, _ = trainer.pid.get_cost_obs_and_pred_from_params(theta)
    simulated = np.asarray(trainer.pid.get_obs_output_dict(operands_list[0])['const'],
                           dtype=float).reshape(-1)

    assert np.all(np.isfinite(predicted))
    assert predicted.shape == simulated.shape

    # Loose on purpose. Both features live on O(1) scales (the centre probe starts at 1.0 and
    # relaxes towards u_D), and 20 Sobol samples over a two-parameter box is a small design,
    # so this is checking "the same function" rather than "a good emulator". A permuted or
    # unscaled feature vector misses by far more than this.
    assert predicted == pytest.approx(simulated, abs=0.15), (
        f'at heat/k={theta[0]:.4g}, heat/u_D={theta[1]:.4g} the emulator predicts '
        f'{predicted} but the FEniCSx solver gives {simulated}')
