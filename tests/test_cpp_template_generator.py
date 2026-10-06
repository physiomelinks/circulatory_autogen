"""
The template-based C++ generator (model_type: cpp, libcuflynx/generators/cpp).

Unit tests check the api blocks that describe couplings; the slow integration tests generate
models, build them with CMake and run them. Builds are skipped when CMake (or, for CVODE models,
SUNDIALS) is not available -- set SUNDIALS_DIR to a SUNDIALS install prefix if it is not found
automatically.
"""
import copy
import json
import os
import re
import shutil
import signal
import subprocess

import numpy as np
import pytest
import yaml

from libcuflynx.generators.cpp.api import APIConfigError, resolve_process_apis, validate_api_block, unit_factor
from libcuflynx.utilities.package_resources import builtin_modules_dir


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _coupling_entries(resolved=True):
    """The built-in coupling module configs; with resolved=True each consumer that names a
    process carries that process's channels, as after the configs load."""
    with open(os.path.join(builtin_modules_dir(), 'coupling_modules_config.json')) as f:
        entries = json.load(f)
    if resolved:
        resolve_process_apis([((e['vessel_type'], e['BC_type']), e['api']) for e in entries if 'api' in e])
    return entries


def _fv1d_api(resolved=True):
    return next(e['api'] for e in _coupling_entries(resolved) if e['vessel_type'] == 'FV1D_vessel')


def _fv1d_solver_api():
    return next(e['api'] for e in _coupling_entries(False) if e['vessel_type'] == 'FV1D_solver')


def _generation_inputs(user_inputs_dir, resources_src, tmp_path, file_prefix, solver, **extra):
    """user inputs for one generation, with the resources copied so nothing is written into the
    repository's resources/ (the 1D split writes derived module arrays next to its input)."""
    with open(os.path.join(user_inputs_dir, 'user_inputs.yaml')) as f:
        d = yaml.safe_load(f)
    for k in ['user_inputs_path_override', 'resources_dir', 'generated_models_dir',
              'param_id_output_dir', 'param_id_obs_path']:
        d.pop(k, None)
    resources = tmp_path / 'resources'
    resources.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(resources_src):
        if name.startswith(file_prefix + '_'):
            shutil.copy(os.path.join(resources_src, name), resources)
    d.update({
        'file_prefix': file_prefix,
        'input_param_file': f'{file_prefix}_parameters.csv',
        'model_type': 'cpp',
        'couple_to_1d': False,
        'solver_info': {'solver': solver, 'dt_solver': 1e-4, 'MaximumNumberOfSteps': 5000},
        'resources_dir': str(resources),
        'generated_models_dir': str(tmp_path / 'generated_models'),
    })
    d.update(extra)
    return d


def _cmake_build(src_dir, build_dir):
    """Configure and build; skip the test if CMake or a dependency is missing."""
    if shutil.which('cmake') is None:
        pytest.skip('cmake not available')
    args = ['cmake', '-S', str(src_dir), '-B', str(build_dir)]
    if os.environ.get('SUNDIALS_DIR'):
        args.append(f"-DSUNDIALS_DIR={os.environ['SUNDIALS_DIR']}")
    cfg = subprocess.run(args, capture_output=True, text=True)
    if cfg.returncode != 0:
        if 'SUNDIALS' in cfg.stdout + cfg.stderr or 'cvode' in cfg.stdout + cfg.stderr:
            pytest.skip('SUNDIALS not found by CMake (set SUNDIALS_DIR)')
        if 'nlohmann' in cfg.stdout + cfg.stderr or 'FetchContent' in cfg.stdout + cfg.stderr:
            pytest.skip('nlohmann_json not available and could not be downloaded')
        pytest.fail(f'cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}')
    build = subprocess.run(['cmake', '--build', str(build_dir), '-j', '4'], capture_output=True, text=True)
    assert build.returncode == 0, f'build failed:\n{build.stdout}\n{build.stderr}'
    assert 'warning' not in build.stdout.lower(), build.stdout


def _run_coupled(cpp_dir, tmp_path, timeout=900):
    """Build the generated 0D model and the coupler, then run the coupler with the generated
    coupler_config.json (it launches main0d and the FV 1D solver). Returns the coupler's log."""
    _cmake_build(cpp_dir, cpp_dir / 'build')
    coupler_src = os.path.join(os.path.dirname(__file__), '..', 'src', 'libcuflynx', 'coupler')
    if not os.path.isfile(os.path.join(coupler_src, 'coupler.cpp')):
        pytest.skip('coupler sources need a source checkout')
    _cmake_build(coupler_src, tmp_path / 'coupler_build')
    # The 0D model on its own first, so a solver/build problem is reported as such rather than
    # as a hung coupler (which waits on its pipes for ever if a child process dies).
    alone = subprocess.run([str(cpp_dir / 'build' / 'main0d'), '-tEnd', '0.1', '-tSave', '0',
                            '-outDir', str(tmp_path / 'standalone_out')],
                           capture_output=True, text=True, timeout=120)
    assert alone.returncode == 0, (alone.stdout + alone.stderr)[-3000:]

    log_path = tmp_path / 'coupler.log'
    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen([str(tmp_path / 'coupler_build' / 'coupler'), str(cpp_dir / 'coupler_config.json')],
                                stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the coupler and the 0D/1D processes it launched
            proc.wait()
            returncode = None
    log = log_path.read_text(errors='replace')
    tail = '\n'.join(log.splitlines()[-80:])
    assert returncode is not None, f'coupled run timed out; last output:\n{tail}'
    assert returncode == 0, tail
    assert 'Both processes terminated successfully with status codes : 0 0' in log, tail
    return log


def _load_output(path):
    header = open(path).readline().lstrip('#').split(';')
    cols = {}
    for i, item in enumerate(header):
        if ':' in item:
            cols[item.split(':', 1)[1].split('[')[0].strip()] = i
    return cols, np.loadtxt(path, ndmin=2)


# ---------------------------------------------------------------------------------------------
# unit: api blocks
# ---------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_builtin_fv1d_api_blocks_are_valid():
    with open(os.path.join(builtin_modules_dir(), 'coupling_modules_config.json')) as f:
        entries = json.load(f)
    apis = [e for e in entries if 'api' in e]
    assert {(e['vessel_type'], e['BC_type']) for e in apis} >= {('FV1D_vessel', 'vp'), ('FV1D_volume_sum', 'nn'),
                                                                ('FV1D_solver', 'nn')}
    for e in apis:
        assert e['module_format'] == 'external_api' or e['vessel_type'] == 'FV1D_volume_sum'
        assert validate_api_block(e['api'], e['vessel_type'])
    # the pipes are written down once, in the process; every consumer names it
    for e in apis:
        if e['api']['role'] == 'consumer':
            assert e['api']['process'] == 'FV1D_solver' and 'channels' not in e['api']
    resolve_process_apis([((e['vessel_type'], e['BC_type']), e['api']) for e in apis])
    solver = next(e['api'] for e in apis if e['vessel_type'] == 'FV1D_solver')
    for e in apis:
        if e['api']['role'] == 'consumer':
            assert e['api']['channels'] == solver['channels']
            assert e['api']['process_api'] is solver


@pytest.mark.unit
@pytest.mark.parametrize('mutate, message', [
    (lambda a: a.pop('program'), "needs a 'program'"),
    (lambda a: a.update(coordinator='scheduler'), "needs a 'coordinator'"),
    (lambda a: a.update(transport='cpp_class'), "must use transport 'named_pipe'"),
    (lambda a: a.update(calls=[]), 'lists no calls'),
    (lambda a: a['channels'].clear(), "needs a 'channels' dict"),
])
def test_malformed_process_blocks_are_rejected(mutate, message):
    api = copy.deepcopy(_fv1d_solver_api())
    assert validate_api_block(api)
    mutate(api)
    with pytest.raises(APIConfigError, match=message):
        validate_api_block(api)


@pytest.mark.unit
def test_consumer_must_name_an_existing_process():
    consumer = copy.deepcopy(_fv1d_api(resolved=False))
    consumer['process'] = 'no_such_solver'
    with pytest.raises(APIConfigError, match="names process 'no_such_solver'"):
        resolve_process_apis([(('FV1D_vessel', 'vp'), consumer), (('FV1D_solver', 'nn'), _fv1d_solver_api())])
    # a call on a channel the process doesn't have is caught once the channels are known
    consumer = copy.deepcopy(_fv1d_api(resolved=False))
    consumer['calls'][0]['channel'] = 'nowhere'
    with pytest.raises(APIConfigError, match='undeclared channel'):
        resolve_process_apis([(('FV1D_vessel', 'vp'), consumer), (('FV1D_solver', 'nn'), _fv1d_solver_api())])


@pytest.mark.unit
@pytest.mark.parametrize('mutate, message', [
    (lambda a: a.pop('transport'), "missing 'transport'"),
    (lambda a: a.update(role='both'), 'api role'),
    (lambda a: a['calls'][0].update(when='sometimes'), "unknown 'when'"),
    (lambda a: a['calls'][0].update(channel='nowhere'), 'undeclared channel'),
    (lambda a: a['channels']['dt'].pop('recv'), 'has no recv pipe'),
])
def test_malformed_api_blocks_are_rejected(mutate, message):
    api = copy.deepcopy(_fv1d_api())
    mutate(api)
    with pytest.raises(APIConfigError, match=message):
        validate_api_block(api)


@pytest.mark.unit
def test_fv1d_api_matches_the_coupler_and_1d_solver():
    """The api block only describes the 0D side of the pipe protocol; the coupler (C++) and the
    FV 1D solver (Python) hard-code their side. Catch the two drifting apart."""
    src = os.path.join(os.path.dirname(__file__), '..', 'src', 'libcuflynx')
    coupler = os.path.join(src, 'coupler', 'coupler.cpp')
    if not os.path.isfile(coupler):
        pytest.skip('coupler sources need a source checkout')
    coupler_src = open(coupler).read()
    main1d_src = open(os.path.join(src, 'solver1d', 'main1D.py')).read()
    api = _fv1d_solver_api()
    for channel in api['channels'].values():
        for direction in ('send', 'recv'):
            if direction in channel:
                name = channel[direction].replace('{i}', '')
                assert f'"{name}"' in coupler_src, f'{name} is not a pipe the coupler creates'
    assert f"DATA_LENGTH = {api['message_length']}" in main1d_src


@pytest.mark.unit
def test_readable_names_match_the_python_generator():
    """C index names and Python attribute names come from the same scheme."""
    from libcuflynx.generators.naming import build_symbols, make_identifier
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    pairs = [('heart_module', 'q_lv'), ('parameters', 'R-T 1'), ('heart_module', 'q_lv'), ('1st', 'x')]
    assert build_symbols(pairs) == ['heart_module_q_lv', 'parameters_r_t_1', 'heart_module_q_lv_2', 'n_1st_x']
    assert make_identifier('') == 'unnamed'
    infos = [{'component': c, 'name': n} for c, n in pairs]
    _, attrs = PythonGenerator._build_qualified_symbols(PythonGenerator.__new__(PythonGenerator), infos)
    assert [attrs[i] for i in range(len(pairs))] == build_symbols(pairs)


@pytest.mark.unit
def test_api_unit_factors():
    assert unit_factor({'api_units': 'mmHg'}) == pytest.approx(133.322387415)
    assert unit_factor({'api_units': 'ml'}) == pytest.approx(1e-6)
    assert unit_factor({'api_units': 'ml', 'factor': 2.0}) == 2.0
    with pytest.raises(APIConfigError):
        unit_factor({'api_units': 'furlongs'})


@pytest.mark.unit
def test_cvode_solver_init_supports_sundials_5_to_7():
    from libcuflynx.generators.CVSCppGenerator import CVS0DCppGenerator
    gen = CVS0DCppGenerator.__new__(CVS0DCppGenerator)
    gen.solver, gen.reltol, gen.abstol, gen.dtSolver, gen.nMaxSteps = 'CVODE', 1e-7, 1e-9, 1e-4, 5000
    emitted = gen._build_solver_init_function()
    assert 'SUNDIALS_VERSION_MAJOR >= 7' in emitted
    assert 'SUNContext_Create(SUN_COMM_NULL' in emitted
    assert 'CVodeCreate(CV_BDF)' in emitted  # SUNDIALS 5, no context
    assert 'CVodeSetMaxStep(solver, hmax)' in emitted


# ---------------------------------------------------------------------------------------------
# integration: generate, build, run
# ---------------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_delay_model_builds_and_delays_exactly(user_inputs_dir, resources_dir, tmp_path):
    """delay_info -> an external variable filled from a time-stamped history (RK4, no SUNDIALS)."""
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    inp = _generation_inputs(user_inputs_dir, resources_dir, tmp_path, 'delay_test', 'RK4',
                             pre_time=0.0, sim_time=3.0, dt=0.01)
    inp['solver_info']['dt_solver'] = 1e-3
    assert generate_with_new_architecture(False, inp)
    cpp_dir = tmp_path / 'generated_models' / 'delay_test'
    for name in ('model0d_core.c', 'model0d_core.h', 'model0d.h', 'model0d.cpp', 'main0d.cpp', 'CMakeLists.txt'):
        assert (cpp_dir / name).is_file(), name
    # readable: named indices in the libCellML code and in the wrapper, no bare numbers
    header = (cpp_dir / 'model0d_core.h').read_text()
    assert 'S_delay_module_v = 0,' in header and '} StateIndex;' in header and '} VariableIndex;' in header
    core = (cpp_dir / 'model0d_core.c').read_text()
    compute = core[core.index('void initialiseVariables'):]
    assert 'rates[S_delay_module_v]' in compute
    assert 'externalVariable(voi, states, rates, variables, V_delay_module_v_delay)' in compute
    assert not re.search(r'\b(states|rates|variables)\[\d+\]', compute)
    wrapper = (cpp_dir / 'model0d.cpp').read_text()
    assert 'setExternal(V_delay_module_v_delay, ' in wrapper
    _cmake_build(cpp_dir, cpp_dir / 'build')
    out = tmp_path / 'out'
    run = subprocess.run([str(cpp_dir / 'build' / 'main0d'), '-outDir', str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr

    cs, s = _load_output(out / 'sol0D_states.txt')
    cv, v = _load_output(out / 'sol0D_variables.txt')
    t, V = s[:, 0], s[:, cs['delay/V']]
    V_delay = v[:, cv['delay/V_delay']]
    later = t > 0.6
    assert np.max(np.abs(V_delay[later] - np.interp(t[later] - 0.5, t, V))) < 1e-9
    assert np.ptp(V) > 1.0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize('solver_row', [True, False], ids=['with_FV1D_solver_row', 'without_FV1D_solver_row'])
def test_coupled_fv1d_generation_fills_connection_info(user_inputs_dir, resources_dir, tmp_path, solver_row):
    """The FV1D api block reproduces the connection info the previous generator wrote for
    aortic_bif_hybrid_V1 (two terminals receiving pressure from 1D, sending their inflow state),
    and the FV1D_solver process block gives the coupler's configuration. The FV1D_solver row is
    optional: hybrid arrays written before it existed still get a coupler_config.json."""
    import sys
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    cpp_dir = tmp_path / 'cpp'
    ini = tmp_path / '1d' / 'run000' / 'input.ini'
    inp = _generation_inputs(user_inputs_dir, resources_dir, tmp_path, 'aortic_bif_hybrid_V1', 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini),
                             pre_time=1.0, sim_time=2.0, coupler_pipe_dir=str(tmp_path / 'pipes'))
    array = tmp_path / 'resources' / 'aortic_bif_hybrid_V1_module_array.csv'
    assert 'FV1D_solver' in array.read_text()
    if not solver_row:
        array.write_text(''.join(l for l in array.read_text().splitlines(True) if not l.startswith('FV1D_solver')))
    assert generate_with_new_architecture(False, inp)
    with open(cpp_dir / 'aortic_bif_hybrid_V1_coupler1d0d.json') as f:
        info = json.load(f)
    assert list(info) == ['1', '2']
    for key, cellml_idx, port_idx in (('1', 5, 1), ('2', 12, 4)):
        assert info[key]['cellml_idx'] == cellml_idx
        assert info[key]['port_idx'] == port_idx
        assert info[key]['port_state0_or_var1'] == 0
        assert info[key]['cellml_bc_flow0_or_press1'] == 1
        assert info[key]['R_T_variable_idx'] == -1
        assert 'port_volume_sum' not in info[key]
    source = (cpp_dir / 'model0d.cpp').read_text()
    assert 'zero_to_parent_dt' in source and 'parent_to_zero_2' in source
    # no volume sum in this model, so the volume pipe (which the coupler would not create) is not opened
    assert 'parent_to_zero_vol' not in source

    with open(cpp_dir / 'coupler_config.json') as f:
        config = json.load(f)
    assert config['networkName'] == 'aortic_bif_hybrid_V1'
    assert config['inputFold'] == str(cpp_dir) + '/'
    assert config['ODEsolver'] == 'CVODE'
    assert config['T0'] == pytest.approx(1.1)        # the global period T of the model
    assert config['nCC'] == 3                        # whole periods covering pre_time + sim_time = 3 s
    assert config['tmp_pipe_path'] == str(tmp_path / 'pipes') + '/'
    assert config['solver0d_path'] == str(cpp_dir / 'build' / 'main0d')
    assert config['python_path'] == sys.executable
    assert config['solver1d_path'].endswith(os.path.join('solver1d', 'main1D.py'))
    assert os.path.isfile(config['solver1d_path'])
    assert config['initFile_sim1d_path'] == str(ini)
    # main0d, run without the coupler, defaults to the same run length
    main = (cpp_dir / 'main0d.cpp').read_text()
    assert 'double T0 = 1.1;' in main and 'int nCC = 3;' in main


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_fv1d_simulation_runs(user_inputs_dir, resources_dir, tmp_path):
    """End to end: generate aortic_bif_hybrid_V1 (CVODE), build main0d and the coupler with CMake,
    and run one cardiac cycle coupled to the Python FV 1D solver, as configured by the generated
    coupler_config.json."""
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    cpp_dir = tmp_path / 'gen' / 'cpp'
    ini = tmp_path / '1d' / 'run000' / 'input.ini'
    # one period of T = 1.1 s; the pipe folder doesn't exist yet: the coupler creates it
    inp = _generation_inputs(user_inputs_dir, resources_dir, tmp_path, 'aortic_bif_hybrid_V1', 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini),
                             dt=0.01, pre_time=0.0, sim_time=1.1, coupler_pipe_dir=str(tmp_path / 'pipes'))
    assert generate_with_new_architecture(False, inp)
    _run_coupled(cpp_dir, tmp_path)

    out0d = tmp_path / 'simulation_outputs_cpp' / 'aortic_bif_hybrid_V1'
    cv, v = _load_output(out0d / 'sol0D_variables.txt')
    assert v.shape[0] > 50 and np.all(np.isfinite(v))
    u_in = v[:, cv['terminal_1/u_in']]           # pressure received from the 1D solver [Pa]
    assert 5e3 < np.max(u_in) < 3e4
    sol1d = np.genfromtxt(ini.parent / 'res' / 'sol1D_parent.txt')
    assert np.all(np.isfinite(sol1d))


@pytest.mark.integration
@pytest.mark.slow
def test_provider_api_module_couples_through_ports(user_inputs_dir, resources_dir, tmp_path):
    """An external (api) module in the module array is coupled to a CellML module through
    matching ports: its api functions name its own port variables, which resolve to the CellML
    variables on the other end. The CellML model itself does not include it."""
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    ext_dir = tmp_path / 'external_modules'
    ext_dir.mkdir()
    probe = [{
        'vessel_type': 'api_probe', 'BC_type': 'nn', 'module_format': 'external_api', 'module_file': '',
        'module_type': 'probe_api',
        'entrance_ports': [{'port_type': 'volume_port', 'variables': ['V_heart']}],
        'exit_ports': [], 'general_ports': [],
        'variables_and_units': [['V_heart', 'm3', 'access', 'variable']],
        'api': {'name': 'probe', 'role': 'provider', 'transport': 'cpp_class', 'namespace': 'probe',
                'class_name': 'Probe',
                'functions': [{'name': 'get_heart_volume', 'kind': 'get', 'variable': 'V_heart', 'api_units': 'ml'},
                              {'name': 'get_aortic_pressure', 'kind': 'get', 'variable': 'aortic_root/u',
                               'api_units': 'mmHg'},
                              {'name': 'set_time', 'kind': 'time'},
                              {'name': 'solve_time_step', 'kind': 'step'}]},
    }]
    (ext_dir / 'probe_module_config.json').write_text(json.dumps(probe))
    inp = _generation_inputs(user_inputs_dir, resources_dir, tmp_path, '3compartment', 'CVODE',
                             pre_time=0.0, sim_time=1.0, dt=0.01, external_modules_dir=str(ext_dir))
    array = tmp_path / 'resources' / '3compartment_module_array.csv'
    lines = array.read_text().splitlines()
    lines = [(l.rstrip() + ' api_probe') if l.split(',')[0].strip() == 'heart' else l for l in lines]
    lines.append('api_probe, nn, api_probe, heart, ')
    array.write_text('\n'.join(lines) + '\n')

    assert generate_with_new_architecture(False, inp)
    model_dir = tmp_path / 'generated_models' / '3compartment'
    assert 'api_probe' not in (model_dir / '3compartment.cellml').read_text()
    source = (model_dir / 'circulation_api.cpp').read_text()
    assert '// heart/q_heart' in source
    assert '// aortic_root/u' in source
    assert 'class Probe' in (model_dir / 'circulation_api.h').read_text()
    _cmake_build(model_dir, model_dir / 'build')


# The hybrid model is the 0D one with its three vessels moved to the FV 1D solver, so the two
# should give close (not identical) haemodynamics: the 1D vessels add wave propagation that a
# lumped vessel can't. Compared over the last of N_CYCLES periods.
N_CYCLES = 3
CONVERTED_VESSELS = ['parent', 'daughter_1', 'daughter_2']
# Measured (CVODE, 1D solver defaults): means equal (mass conservation; mean pressure = mean flow x
# terminal resistance), pulse amplitudes within 0.4 %, largest pointwise difference 3.6 % of the
# cycle's range. The tolerances leave room for solver and platform differences.
MEAN_TOL, PULSE_TOL, SHAPE_TOL = 0.02, 0.05, 0.10


def _last_cycle(path, names, t_start):
    cols, data = _load_output(path)
    keep = data[:, 0] >= t_start - 1e-9
    return data[keep, 0], {n: data[keep, cols[n]] for n in names}


@pytest.mark.integration
@pytest.mark.slow
def test_coupled_fv1d_model_stays_close_to_the_0d_model(user_inputs_dir, resources_dir, tmp_path):
    """aortic_bif_0d (all 0D) against the same model with its vessels converted to 1D by
    convert_0d_to_1d and run through the coupler with the generated coupler_config.json:
    terminal pressures and flows over the last period must stay close."""
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    from libcuflynx.scripts.convert_0d_to_1d import convert_0d_to_1d

    # the 0D model, generated and run on its own
    zero = tmp_path / 'zero'
    inp0 = _generation_inputs(user_inputs_dir, resources_dir, zero, 'aortic_bif_0d', 'CVODE', dt=0.01)
    T = 1.1  # the global period T in aortic_bif_0d_parameters.csv
    t_end = N_CYCLES * T
    inp0.update(pre_time=0.0, sim_time=t_end)
    assert generate_with_new_architecture(False, inp0)
    cpp0 = zero / 'generated_models' / 'aortic_bif_0d'
    _cmake_build(cpp0, cpp0 / 'build')
    run0 = subprocess.run([str(cpp0 / 'build' / 'main0d'), '-tEnd', str(t_end), '-tSave', '0',
                           '-outDir', str(zero / 'out')], capture_output=True, text=True, timeout=600)
    assert run0.returncode == 0, (run0.stdout + run0.stderr)[-3000:]

    # the hybrid model: the vessels moved to 1D, plus the FV1D_solver row
    converted = tmp_path / 'converted'
    convert_0d_to_1d('aortic_bif', zero / 'resources', 'aortic_bif_0d_parameters.csv', converted, CONVERTED_VESSELS)
    array = (converted / 'aortic_bif_hybrid_module_array.csv').read_text()
    assert 'FV1D_solver,nn,FV1D_solver' in array
    assert 'K_tube_parent' not in array

    hyb = tmp_path / 'hyb'
    cpp_dir = hyb / 'gen' / 'cpp'
    ini = hyb / '1d' / 'run000' / 'input.ini'
    inph = _generation_inputs(user_inputs_dir, converted, hyb, 'aortic_bif_hybrid', 'CVODE',
                              couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                              cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini),
                              dt=0.01, pre_time=0.0, sim_time=t_end, coupler_pipe_dir=str(hyb / 'pipes'))
    assert generate_with_new_architecture(False, inph)
    config = json.loads((cpp_dir / 'coupler_config.json').read_text())
    assert config['T0'] == pytest.approx(T) and config['nCC'] == N_CYCLES
    _run_coupled(cpp_dir, hyb, timeout=1800)

    # last period of each; the coupled main0d saves from (nCC - 2) T0
    t_start = (N_CYCLES - 1) * T
    names = ['terminal_1/u', 'terminal_2/u', 'terminal_1/v', 'terminal_2/v']
    t0d, zero_vals = _last_cycle(zero / 'out' / 'sol0D_states.txt', [n for n in names if n.endswith('/v')], t_start)
    _, zero_vars = _last_cycle(zero / 'out' / 'sol0D_variables.txt', [n for n in names if n.endswith('/u')], t_start)
    zero_vals.update(zero_vars)
    out_hyb = hyb / 'simulation_outputs_cpp' / 'aortic_bif_hybrid'
    thy, hyb_vals = _last_cycle(out_hyb / 'sol0D_states.txt', [n for n in names if n.endswith('/v')], t_start)
    _, hyb_vars = _last_cycle(out_hyb / 'sol0D_variables.txt', [n for n in names if n.endswith('/u')], t_start)
    hyb_vals.update(hyb_vars)
    # the pressure each terminal receives: from the daughter vessel in 0D, from the 1D solver here
    _, z_in = _last_cycle(zero / 'out' / 'sol0D_variables.txt', ['daughter_1/u'], t_start)
    _, h_in = _last_cycle(out_hyb / 'sol0D_variables.txt', ['terminal_1/u_in'], t_start)
    zero_vals['inlet pressure 1'] = z_in['daughter_1/u']
    hyb_vals['inlet pressure 1'] = h_in['terminal_1/u_in']

    report = {}
    for name in zero_vals:
        z, h = zero_vals[name], hyb_vals[name]
        assert np.all(np.isfinite(h)), name
        report[name] = {
            'mean_rel': abs(np.mean(h) - np.mean(z)) / abs(np.mean(z)),
            'ptp_rel': abs(np.ptp(h) - np.ptp(z)) / np.ptp(z),
            # shape: the hybrid trace resampled onto the 0D times, against the 0D cycle's range
            'max_rel': np.max(np.abs(np.interp(t0d, thy, h) - z)) / np.ptp(z),
        }
    print('\nhybrid vs 0D over the last period:')
    for name, r in report.items():
        print(f"  {name:18s} mean {r['mean_rel']:6.1%}  pulse {r['ptp_rel']:6.1%}  max diff / range {r['max_rel']:6.1%}")
    for name, r in report.items():
        assert r['mean_rel'] < MEAN_TOL, f"{name}: mean differs by {r['mean_rel']:.1%}"
        assert r['ptp_rel'] < PULSE_TOL, f"{name}: pulse differs by {r['ptp_rel']:.1%}"
        assert r['max_rel'] < SHAPE_TOL, f"{name}: largest difference is {r['max_rel']:.1%} of the 0D range"

