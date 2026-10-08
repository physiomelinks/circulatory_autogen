"""
Coupling a generated C++ 0D model to an external Python model (api transport "python",
libcuflynx.coupling).

The fixture (tests/data/external_coupling) is two capillaries exchanging O2 with tissue:
``cap_k`` (haemodynamics) feeds ``capillary_GE_k`` (O2 exchange), whose
``capillary_to_flux_port`` connects to

* ``microvasc_O2_0d``: one CellML ``well_mixed_volume`` per capillary (a module kept with the
  tests, so they mean the same with the built-in modules or a module library), or
* ``microvasc_O2_ext``: one external row, ``well_mixed_tissue`` (a numpy class), connected to
  both capillaries, with the same equations.

So the coupled run must reproduce the all-CellML model. The FEniCS examples, which follow the
same pattern, are tested in circulatory-autogen-modules.
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from libcuflynx.generators.cpp.api import APIConfigError, validate_api_block

DATA = os.path.join(os.path.dirname(__file__), 'data', 'external_coupling')
MODULES = os.path.join(DATA, 'modules')
T_END = 20.0


def _module_entry():
    with open(os.path.join(MODULES, 'well_mixed_tissue_module_config.json')) as f:
        entry = json.load(f)[0]
    entry['api']['_config_dir'] = MODULES
    return entry


def _generate(tmp_path, prefix, **extra):
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    resources = tmp_path / 'resources'
    resources.mkdir(exist_ok=True)
    for name in os.listdir(os.path.join(DATA, 'resources')):
        if name.startswith(prefix + '_'):
            shutil.copy(os.path.join(DATA, 'resources', name), resources)
    inp = dict(file_prefix=prefix, input_param_file=f'{prefix}_parameters.csv', model_type='cpp',
               resources_dir=str(resources), generated_models_dir=str(tmp_path / 'gen'), pre_time=0.0,
               sim_time=T_END, dt=0.1, DEBUG=False, couple_to_1d=False, external_modules_dir=[MODULES],
               solver_info={'solver': 'CVODE', 'dt_solver': 1e-3, 'MaximumNumberOfSteps': 5000})
    inp.update(extra)
    assert generate_with_new_architecture(False, inp)
    return tmp_path / 'gen' / prefix


def _require_build_tools():
    if shutil.which('cmake') is None:
        pytest.skip('cmake not available')


def _build(model_dir):
    _require_build_tools()
    args = ['cmake', '-S', str(model_dir), '-B', str(model_dir / 'build')]
    if os.environ.get('SUNDIALS_DIR'):
        args.append(f"-DSUNDIALS_DIR={os.environ['SUNDIALS_DIR']}")
    cfg = subprocess.run(args, capture_output=True, text=True)
    if cfg.returncode != 0 and 'SUNDIALS' in cfg.stdout + cfg.stderr:
        pytest.skip('SUNDIALS not found by CMake (set SUNDIALS_DIR)')
    assert cfg.returncode == 0, cfg.stdout + cfg.stderr
    build = subprocess.run(['cmake', '--build', str(model_dir / 'build'), '-j', '4'], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    assert 'warning' not in build.stdout.lower(), build.stdout


def _load(path):
    header = open(path).readline().lstrip('#').split(';')
    cols = {h.split(':', 1)[1].split('[')[0].strip(): i for i, h in enumerate(header) if ':' in h}
    return cols, np.loadtxt(path, ndmin=2)


@pytest.fixture(scope='module')
def zero_d_reference(tmp_path_factory):
    """The all-CellML model run through main0d: tissue O2 and capillary flux of both capillaries."""
    tmp = tmp_path_factory.mktemp('zero_d')
    model_dir = _generate(tmp, 'microvasc_O2_0d')
    _build(model_dir)
    out = tmp / 'out'
    run = subprocess.run([str(model_dir / 'build' / 'main0d'), '-outDir', str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    cs, s = _load(out / 'sol0D_states.txt')
    cv, v = _load(out / 'sol0D_variables.txt')
    return {'t': s[:, 0],
            'C': np.column_stack([s[:, cs[f'volume_{k}/C_P']] for k in (0, 1)]),
            'J': np.column_stack([v[:, cv[f'capillary_GE_{k}/flux_O2_c']] for k in (0, 1)])}


@pytest.fixture(scope='module')
def coupled_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('coupled')
    return _generate(tmp, 'microvasc_O2_ext')


def _with_settings(model_dir, tmp_path, **settings):
    """A copy of the generated model whose external model runs with other api settings."""
    copy_dir = tmp_path / 'model'
    shutil.copytree(model_dir, copy_dir, ignore=shutil.ignore_patterns('build'))
    path = copy_dir / 'external_models.json'
    info = json.loads(path.read_text())
    info['external_models'][0].update(settings)
    path.write_text(json.dumps(info))
    return copy_dir


def _max_rel_error(result, ref):
    C = result.exchange['tissue/C_t']
    err = 0.0
    for k in (0, 1):
        c = np.interp(ref['t'], result.times, C[:, k])
        err = max(err, np.max(np.abs(c - ref['C'][:, k])) / np.ptp(ref['C'][:, k]))
    return err


# ---------------------------------------------------------------------------------------------
# api block and exchange table
# ---------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_python_api_block_is_valid():
    assert validate_api_block(_module_entry()['api'])


@pytest.mark.unit
@pytest.mark.parametrize('mutate, message', [
    (lambda a: a.update(role='consumer'), "must have role 'provider'"),
    (lambda a: a.pop('python'), 'needs "python"'),
    (lambda a: a['python'].update(file='missing.py'), 'not found'),
    (lambda a: a.update(coupling_step=0.1), 'unknown key'),
    (lambda a: a.update(coupling_dt=-1), 'positive number'),
    (lambda a: a.update(relaxation=1.5), r'\(0, 1\]'),
    (lambda a: a.update(variables={'C_t': {'direction': 'sideways'}}), 'direction must be one of'),
])
def test_malformed_python_api_blocks_are_rejected(mutate, message):
    api = copy.deepcopy(_module_entry()['api'])
    mutate(api)
    with pytest.raises(APIConfigError, match=message):
        validate_api_block(api)


@pytest.mark.integration
def test_generation_writes_the_exchange_table(coupled_model):
    """Directions are inferred from the connected CellML variables: GE_capillary's ub_O2_t is a
    boundary condition (set by the external model), flux_O2_c is computed (read by it). The port
    connects to two capillaries, so each variable has two values, in module-array order."""
    info = json.loads((coupled_model / 'external_models.json').read_text())
    (ext,) = info['external_models']
    assert ext['class'] == 'WellMixedTissue' and os.path.isfile(ext['file'])
    assert ext['parameters'] == pytest.approx({'V': 8e-14, 'M': -0.0011830357142857144, 'k_reduce': 50.0,
                                               'C_init': 0.02})
    by_var = {v['variable']: v for v in ext['variables']}
    assert by_var['C_t']['direction'] == 'from_external'
    assert by_var['J_c']['direction'] == 'to_external'
    for v in by_var.values():
        assert v['neighbours'] == ['capillary_GE_0', 'capillary_GE_1']
    source = (coupled_model / 'model0d_capi.cpp').read_text()
    assert 'h->model.setExternal(V_' in source and '// capillary_GE_0/ub_O2_t' in source
    assert '// capillary_GE_1/flux_O2_c' in source
    assert 'add_library(model0d_capi SHARED' in (coupled_model / 'CMakeLists.txt').read_text()


@pytest.mark.integration
def test_check_lists_the_exchange(coupled_model, capsys):
    from libcuflynx.coupling.runner import main
    assert main([str(coupled_model), '--check']) == 0
    out = capsys.readouterr().out
    assert 'C_t' in out and 'external -> 0D' in out and 'J_c' in out and '0D -> external' in out
    assert 'capillary_GE_0, capillary_GE_1' in out


# ---------------------------------------------------------------------------------------------
# coupled runs against the all-CellML model
# ---------------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_coupled_run_reproduces_the_cellml_model(coupled_model, zero_d_reference, tmp_path):
    from libcuflynx.coupling import run_coupled
    _require_build_tools()
    result = run_coupled(coupled_model, output_dir=tmp_path / 'out', record=['capillary_GE_0/flux_O2_c'])
    ref = zero_d_reference
    assert result.times == pytest.approx(ref['t'])
    # tissue O2 rises from 0.02 to ~0.11 mM over 20 s; explicit coupling at dt 0.01 s
    assert np.ptp(ref['C'][:, 0]) > 0.05
    assert _max_rel_error(result, ref) < 2e-3
    J = result.exchange['tissue/J_c']
    assert np.max(np.abs(np.interp(ref['t'], result.times, J[:, 0]) - ref['J'][:, 0])) < 2e-3 * np.max(np.abs(ref['J']))
    assert result.records['capillary_GE_0/flux_O2_c'] == pytest.approx(J[:, 0])
    # the 0D model writes its usual output files, and the exchange is logged
    cols, s = _load(tmp_path / 'out' / 'sol0D_states.txt')
    assert s.shape[0] == len(ref['t'])
    assert (tmp_path / 'out' / 'coupling_exchange.csv').is_file()
    assert set(result.timings) >= {'total', 'zero_d', 'external', 'setup'}


@pytest.mark.integration
@pytest.mark.slow
def test_explicit_coupling_error_falls_with_the_coupling_step(coupled_model, zero_d_reference, tmp_path):
    """Explicit staggered coupling is first order: halving the coupling step halves the error.
    With subiterations the step is iterated to a trapezoidal coupling, second order: halving
    the step quarters the error."""
    from libcuflynx.coupling import run_coupled
    _require_build_tools()

    def error(dt, **settings):
        model = _with_settings(coupled_model, tmp_path / f'{dt}_{len(settings)}', coupling_dt=dt, **settings)
        return _max_rel_error(run_coupled(model, output_dir=tmp_path / f'out_{dt}_{len(settings)}', verbose=False),
                              zero_d_reference)

    # (steps are also cut at the output times, every 0.1 s, so stay below that)
    explicit = {dt: error(dt) for dt in (0.05, 0.025)}
    assert 1.6 < explicit[0.05] / explicit[0.025] < 2.5, explicit
    iterated = {dt: error(dt, subiterations=10, tol=1e-12) for dt in (0.05, 0.025, 0.0125)}
    assert iterated[0.025] / iterated[0.0125] > 3.0, iterated
    assert iterated[0.05] < explicit[0.05] / 5, (iterated, explicit)


# ---------------------------------------------------------------------------------------------
# errors reach Python instead of ending the process
# ---------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_c_interface_reports_errors(coupled_model):
    from libcuflynx.coupling import Model0dError, Model0dLibrary
    _require_build_tools()
    lib = Model0dLibrary(coupled_model)
    model = lib.create()
    assert model.get('tissue/J_c').shape == (2,)
    with pytest.raises(Model0dError, match='non-finite'):
        model.set('tissue/C_t', [np.nan, 0.1])
    with pytest.raises(Model0dError, match='computed by the 0D model'):
        model.set('tissue/J_c', [0.0, 0.0])
    with pytest.raises(Model0dError, match='2 value'):
        model.set('tissue/C_t', [0.1, 0.1, 0.1])
    model.set('tissue/C_t', [0.05, 0.06])
    model.step(0.1)
    assert model.time == pytest.approx(0.1)
    model.snapshot()
    model.step(0.1)
    model.restore()
    assert model.time == pytest.approx(0.1)
    # a solver the generated model doesn't have: the step fails, the process doesn't exit
    bad = lib.create('no_such_solver')
    with pytest.raises(Model0dError, match='not available'):
        bad.step(0.1)


@pytest.mark.integration
def test_a_bad_external_model_is_reported(coupled_model, tmp_path):
    from libcuflynx.coupling import Model0dError, run_coupled
    _require_build_tools()
    bad = tmp_path / 'bad_model.py'
    bad.write_text('import numpy as np\n'
                   'class Bad:\n'
                   '    def __init__(self, params, neighbours, comm=None, info=None):\n'
                   '        pass\n'
                   '    def initial_outputs(self):\n'
                   '        return {"C_t": np.zeros(3)}\n'
                   '    def step(self, t, dt, inputs):\n'
                   '        return {}\n')
    model = _with_settings(coupled_model, tmp_path / 'm', file=str(bad), **{'class': 'Bad'})
    with pytest.raises(Model0dError, match="needs 2 value"):
        run_coupled(model, output_dir=tmp_path / 'out', verbose=False)
