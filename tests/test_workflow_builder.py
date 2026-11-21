"""Building a calibration workflow in Python (libcuflynx.calibration_workflow.WorkflowBuilder).

The builder writes the same calibration_workflow.json a user writes by hand, with dict inputs
as files under data/ and functions defined in a notebook as a functions file beside it, so the
runner, pack_workflow and CUFLynx see an ordinary workflow. The models are the algebraic ones
of libcuflynx.external_testing.workflow_library (y = 2 p^2 for lin_a; z = 2 p_A^2 + q^2 for
chain), whose optima are known in closed form.
"""
import inspect
import re
import warnings
import json
import linecache
import os

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (WorkflowBuilder, WorkflowError, library_module,
                                             load_workflow, module_array, pack_workflow, prior,
                                             scale_value, selection, set_value)
from libcuflynx.calibration_workflow.builder import function_source
from libcuflynx.calibration_workflow.runner import MODELS_DIR, STEP_RESULT, workflow_status
from libcuflynx.external_testing.workflow_library import (EMULATOR_SETTINGS, IA_OPTIONS,
                                                          OPTIMISER_SETTINGS, SA_OPTIONS,
                                                          STD_A, STD_C, UQ_OPTIONS, Y_A, Z_C,
                                                          build_workflow_library)
from libcuflynx.param_id.operation_funcs import series_to_constant
from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA, load_schema
from libcuflynx.utilities.user_funcs_file import HEADERS, read_funcs_file

LIBRARY_WORKFLOWS = ('twin_split', 'chain_rest', 'chain_analysed', 'chain_pipeline',
                     'chain_array')


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _obs(item, operand, value, std, operation='mean'):
    '''An obs_data dict as a notebook builds one, with a numpy value.'''
    return {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]], 'params_to_change': {}},
            'data_items': [{'data_item_name': item, 'data_type': 'constant',
                            'unit': 'dimensionless', 'operands': [operand],
                            'operation': operation, 'weight': 1.0, 'value': np.float64(value),
                            'std': std}],
            'prediction_items': []}


def _entries(vessel, param, lo, hi):
    '''The in-code params_for_id form (vessel_name, param_name, min, max).'''
    return [{'vessel_name': vessel, 'param_name': param, 'min': lo, 'max': hi}]


def _chain(library, directory, settings=OPTIMISER_SETTINGS):
    '''chain_rest built in Python: fit_a calibrates lin_a's p (2), then rest calibrates C's q
    in the chain with p fixed (q = sqrt(Z_C - Y_A) = 3).'''
    wf = WorkflowBuilder('chain_built', target=library_module('chain', 'v1', 'rest'),
                         directory=directory, module_library_dirs=[library['modules']],
                         settings=settings)
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.add_calibration('rest', obs_data=_obs('z_c', 'mod__C/z', Z_C, STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0), fixed_from='fit_a')
    return wf


def _calibrated(wf, step):
    with open(wf.step_dir(step) / STEP_RESULT) as f:
        return json.load(f)['calibrated'][0]['value']


def _stale(wf):
    return {s: v['stale'] for s, v in wf.status()['steps'].items()}


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

def _keywords(method):
    return {name for name, p in inspect.signature(method).parameters.items()
            if name != 'self' and p.kind != p.VAR_KEYWORD}


@pytest.mark.unit
def test_every_schema_step_key_is_an_argument_of_the_builder():
    """The schema is the single source of truth: a step key added to it fails this until the
    builder takes it (add_step(**raw) reaches it meanwhile)."""
    schema = load_schema(CALIBRATION_WORKFLOW_SCHEMA)
    adders = [getattr(WorkflowBuilder, name) for name in dir(WorkflowBuilder)
              if name.startswith('add_') and name not in ('add_step', 'add_operation',
                                                           'add_cost_func', 'add_modifier_func')]
    keywords = set().union(*(_keywords(m) for m in adders))
    # the keys a method sets from its own arguments: kind (add_step), set (add_set's
    # entries) and sweep (add_sweep's design, outputs, parameters, zip, checked below)
    keywords |= {'kind'} & _keywords(WorkflowBuilder.add_step)
    keywords |= {'set'} if 'entries' in _keywords(WorkflowBuilder.add_set) else set()
    keywords |= {'sweep'} if 'parameters' in _keywords(WorkflowBuilder.add_sweep) else set()
    defs = schema['$defs']
    for where, keys in (('step', defs['step']['properties']),
                        ('param_reduction_options', defs['param_reduction_options']['properties']),
                        ('sweep', defs['step']['properties']['sweep']['properties'])):
        missing = sorted(set(keys) - keywords)
        assert not missing, f'schema {where} keys with no builder argument: {missing}'


@pytest.mark.unit
@pytest.mark.parametrize('which', LIBRARY_WORKFLOWS)
def test_a_loaded_workflow_is_written_back_unchanged(library, which):
    with open(library[which]) as f:
        raw = json.load(f)
    assert WorkflowBuilder.load(library[which]).to_dict() == raw


@pytest.mark.unit
def test_every_kind_of_step_is_written_as_the_schema_reads_it(library, tmp_path):
    """Each add_* gives the step the file would hold, and the written file parses."""
    screened = library['chain/screened']
    obs = os.path.join(screened, 'screened_obs_data.json')
    pfi = os.path.join(screened, 'screened_params_for_id.json')
    chain = library_module('chain', 'v1', 'screened')
    wf = WorkflowBuilder('every_kind', target=chain, directory=tmp_path / 'wf',
                         module_library_dirs=[library['modules']], settings=OPTIMISER_SETTINGS)
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0), store_distribution=True,
                       settings={'UQ_options': UQ_OPTIONS}, description='A alone')
    wf.add_sensitivity('screen', obs_data=obs, params_for_id=pfi, sa_options=SA_OPTIONS)
    wf.add_reduction('reduce', sensitivity_from='screen',
                     param_reduction_options={'method': 'sobol', 'n_params': 1,
                                              'threshold': 0.05})
    wf.add_emulation('emulate', obs_data=obs, params_for_id_from='reduce',
                     emulator_settings=EMULATOR_SETTINGS, fixed_from=['fit_a'])
    wf.add_calibration('fit', obs_data=obs, params_for_id_from='reduce', fixed_from='fit_a')
    wf.add_calibration('again', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0),
                       priors_from=prior('fit_a', kind='normal', inflate=2.0),
                       store_distribution=True, settings={'UQ_options': UQ_OPTIONS})
    wf.add_identifiability('ia', obs_data=obs,
                           params_for_id=os.path.join(library['chain/rest'],
                                                      'rest_params_for_id.json'),
                           ia_options=IA_OPTIONS, result_from='fit')
    wf.add_set('by_hand', [set_value('C/k', 1.0, reason='the gain is one'),
                           scale_value('A/p', 1.1, of='fit_a', reason='ten percent more')])
    wf.add_sweep('sweep_k', parameters={'C/k': [0.9, 1.1]}, outputs=['C/z'],
                 settings={'sim_time': 1.0}, fixed_from=['fit'])
    wf.add_sweep('on_target', on_target=True, parameters={'C/k': {'values': [1.0]}},
                 obs_data=obs, design='one_at_a_time')
    path = wf.write()

    with open(path) as f:
        raw = json.load(f)
    steps = {s['id']: s for s in raw['steps']}
    assert raw['target'] == chain
    assert raw['module_library_dirs'] == [os.path.relpath(library['modules'], wf.directory)]
    assert steps['fit_a']['obs_data'] == 'data/fit_a_obs_data.json'
    assert steps['again']['obs_data'] == 'data/fit_a_obs_data.json'       # written once
    assert steps['screen'] == {
        'id': 'screen', 'kind': 'sensitivity', 'model': chain,
        'obs_data': os.path.relpath(obs, wf.directory),
        'params_for_id': os.path.relpath(pfi, wf.directory),
        'settings': {'sa_options': SA_OPTIONS}}
    assert steps['reduce'] == {'id': 'reduce', 'kind': 'reduction', 'sensitivity_from': 'screen',
                               'settings': {'param_reduction_options': {
                                   'method': 'sobol', 'n_params': 1, 'threshold': 0.05}}}
    assert steps['emulate']['settings'] == {'emulator_settings': EMULATOR_SETTINGS}
    assert steps['fit']['fixed_from'] == ['fit_a'] and 'params_for_id' not in steps['fit']
    assert steps['again']['priors_from'] == [{'step': 'fit_a', 'kind': 'normal',
                                              'inflate': 2.0}]
    assert steps['ia']['result_from'] == 'fit'
    assert steps['by_hand'] == {'id': 'by_hand', 'kind': 'set', 'set': [
        {'parameter': 'C/k', 'value': 1.0, 'reason': 'the gain is one'},
        {'parameter': 'A/p', 'scale': 1.1, 'of': 'fit_a', 'reason': 'ten percent more'}]}
    assert steps['sweep_k']['sweep'] == {'outputs': ['C/z'], 'parameters': {'C/k': [0.9, 1.1]}}
    assert steps['on_target'] == {'id': 'on_target', 'kind': 'sweep',
                                  'obs_data': os.path.relpath(obs, wf.directory),
                                  'on_target': True,
                                  'sweep': {'design': 'one_at_a_time',
                                            'parameters': {'C/k': {'values': [1.0]}}}}
    assert steps['fit_a']['description'] == 'A alone'

    workflow = load_workflow(path)
    assert [s.kind for s in workflow.steps] == [
        'calibration', 'sensitivity', 'reduction', 'emulation', 'calibration', 'calibration',
        'identifiability', 'set', 'sweep', 'sweep']
    assert workflow.step('fit_a').obs_data == os.path.join(wf.directory, 'data',
                                                           'fit_a_obs_data.json')
    assert WorkflowBuilder.load(path).to_dict() == raw


@pytest.mark.unit
def test_paths_are_relative_to_the_workflow_and_helpers_give_schema_fragments(library, tmp_path,
                                                                              monkeypatch):
    monkeypatch.chdir(library['arrays'])
    wf = WorkflowBuilder('arrays', target=module_array('chain_module_array.json'),
                         directory=tmp_path / 'wf', module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_calibration('a', model=module_array('a_only_module_array.json',
                                                'a_only_parameters.json'),
                       obs_data='data/a_obs.json', params_for_id='data/a_params.json')
    wf.add_calibration('sel', model=selection('C'), obs_data='data/chain_obs.json',
                       params_for_id='data/chain_params.json', fixed_from=['a'])
    raw = wf.to_dict()
    arrays = os.path.relpath(library['arrays'], wf.directory)
    assert raw['target'] == os.path.join(arrays, 'chain_module_array.json')
    assert raw['steps'][0]['model'] == {
        'module_array': os.path.join(arrays, 'a_only_module_array.json'),
        'parameters': os.path.join(arrays, 'a_only_parameters.json')}
    assert raw['steps'][1]['model'] == {'select': ['C']}
    assert raw['steps'][1]['obs_data'] == os.path.join(arrays, 'data', 'chain_obs.json')
    assert prior('fit') == 'fit'
    assert selection('a', 'b') == selection(['a', 'b']) == {'select': ['a', 'b']}
    # the package's attribute select is its submodule, which the helper's name leaves alone
    import libcuflynx.calibration_workflow as package
    import libcuflynx.calibration_workflow.select as select_module
    assert package.select is select_module
    workflow = wf.check()
    assert workflow.step('a').obs_data == os.path.join(library['arrays'], 'data', 'a_obs.json')


@pytest.mark.unit
def test_a_step_is_checked_when_it_is_added_and_the_workflow_when_written(library, tmp_path):
    wf = _chain(library, tmp_path / 'wf')
    with pytest.raises(WorkflowError, match='step "bad"'):
        wf.add_step('bad', kind='calibration', model=library_module('lin_a', 'v1', 'fit_a'),
                    obs_data='x.json', params_for_id='y.json', unknown_key=1)
    with pytest.raises(WorkflowError, match='params_for_id'):
        wf.add_calibration('no_params', obs_data='x.json')
    assert wf.steps == ['fit_a', 'rest']
    wf.add_calibration('late', obs_data='x.json', params_for_id='y.json', fixed_from=['nope'])
    with pytest.raises(WorkflowError, match='unknown step "nope"'):
        wf.write()
    wf.remove_step('late')
    wf.write()


@pytest.mark.unit
def test_no_step_is_named_as_a_name_the_run_directory_uses(library, tmp_path):
    """B13: models/, _target/ and inputs/ are the run's own folders and "target" is the
    target's view, so neither the builder nor the schema takes them as a step id."""
    from libcuflynx.calibration_workflow.builder import _RESERVED_IDS
    from libcuflynx.calibration_workflow.runner import TARGET_DIR, TARGET_VIEW
    from libcuflynx.utilities.cuflynx_study import INPUTS_DIR
    assert _RESERVED_IDS == {MODELS_DIR, TARGET_DIR, TARGET_VIEW, INPUTS_DIR}
    schema = load_schema(CALIBRATION_WORKFLOW_SCHEMA)
    wf = _chain(library, tmp_path / 'wf')
    raw = wf.to_dict()
    for name in sorted(_RESERVED_IDS):
        assert name in schema['$defs']['step']['properties']['id']['not']['enum']
        with pytest.raises(WorkflowError, match=f'a step cannot be named "{name}"'):
            wf.add_calibration(name, obs_data='x.json', params_for_id='y.json')
        written = json.loads(json.dumps(raw))
        written['steps'][0]['id'] = name
        with pytest.raises(WorkflowError):
            load_workflow(written)
    assert wf.steps == ['fit_a', 'rest']


@pytest.mark.unit
def test_step_ids_give_distinct_file_names_on_every_file_system(library, tmp_path):
    """B9: an id goes into file names as the workflow name does, and ids whose files or
    directories would be one are refused when they are added."""
    wf = _chain(library, tmp_path / 'wf')
    obs_data, entries = _obs('y_a', 'mod/y', Y_A, STD_A), _entries('mod', 'p', 0.2, 5.0)
    model = library_module('lin_a', 'v1', 'fit_a')
    wf.add_calibration('a:b fit?', model=model, obs_data=obs_data, params_for_id=entries)
    assert wf.to_dict()['steps'][-1]['params_for_id'] == 'data/a_b_fit_params_for_id.json'
    for refused, why in (('Fit_A', 'differs from the step "fit_a" only by case'),
                         ('a_b*fit', 'both give the file name "a_b_fit"'),
                         ('x' * 81, 'has 81 characters; at most 80'),
                         ('', 'non-empty string'), (3, 'non-empty string')):
        with pytest.raises(WorkflowError, match=re.escape(why)):
            wf.add_calibration(refused, model=model, obs_data=obs_data, params_for_id=entries)
    assert wf.steps == ['fit_a', 'rest', 'a:b fit?']
    wf.add_calibration('x' * 80, model=model, obs_data=obs_data, params_for_id=entries)
    wf.add_calibration('fit_a', model=model, obs_data=obs_data,
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))     # replacing itself
    wf.write()
    assert os.path.isfile(os.path.join(wf.directory, 'data', 'a_b_fit_params_for_id.json'))


@pytest.mark.unit
def test_a_file_of_the_users_in_data_is_not_written_over(library, tmp_path):
    """B4: a step may name its own file in data/; the builder does not write a dict there."""
    wf = _chain(library, tmp_path / 'wf')
    mine = os.path.join(wf.directory, 'data', 'fit_a_obs_data.json')
    os.makedirs(os.path.dirname(mine))
    with open(mine, 'w') as f:
        json.dump(_obs('mine', 'mod/y', 1.0, 1.0), f, default=float)
    wf.add_calibration('other', model=library_module('lin_a', 'v1', 'fit_a'), obs_data=mine,
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    with pytest.raises(WorkflowError, match='step "other" names .*fit_a_obs_data.json as its '
                                            'obs_data.*step "fit_a"'):
        wf.write()
    with open(mine) as f:
        assert json.load(f)['data_items'][0]['data_item_name'] == 'mine'


@pytest.mark.unit
def test_empty_option_dicts_none_settings_and_the_default_model(library, tmp_path):
    """B3, B10: an empty options dict is a dict; add_step defaults the model as the others
    do and takes settings=None; emulator_settings is optional."""
    wf = _chain(library, tmp_path / 'wf')
    inputs = dict(obs_data=_obs('z_c', 'mod__C/z', Z_C, STD_C),
                  params_for_id=_entries('mod__C', 'q', 0.1, 10.0))
    wf.add_sensitivity('sa', sa_options={}, **inputs)
    wf.add_emulation('emu', emulator_settings={}, **inputs)
    wf.add_emulation('emu_default', **inputs)
    wf.add_identifiability('ia', ia_options={}, result_from='rest', **inputs)
    wf.add_step('raw', settings=None, description=None, **inputs)
    steps = {s['id']: s for s in wf.to_dict()['steps']}
    assert steps['sa']['settings'] == {'sa_options': {}}
    assert steps['emu']['settings'] == {'emulator_settings': {}}
    assert 'settings' not in steps['emu_default'] and 'settings' not in steps['raw']
    assert steps['ia']['settings'] == {'ia_options': {}}
    target = library_module('chain', 'v1', 'rest')
    assert all(steps[s]['model'] == target for s in ('sa', 'emu', 'emu_default', 'ia', 'raw'))
    # the kinds that run no model, and a sweep on the target, get none
    wf.add_step('by_hand', kind='set', set=[set_value('mod__C/q', 1.0, reason='a test')])
    wf.add_step('grid', kind='sweep', on_target=True, sweep={'parameters': {'mod__C/q': [1, 2]}})
    steps = {s['id']: s for s in wf.to_dict()['steps']}
    assert 'model' not in steps['by_hand'] and 'model' not in steps['grid']


@pytest.mark.unit
def test_pandas_values_are_written_and_a_nan_is_refused(library, tmp_path):
    """B11, B12: pandas values are converted as numpy ones are; JSON has no NaN."""
    pd = pytest.importorskip('pandas')
    from libcuflynx.calibration_workflow.builder import jsonable
    assert jsonable({'v': pd.Series([1.0, 2.0]), 'i': pd.Index([1, 2]),
                     'a': pd.Series([0.5]).to_numpy()}) == {'v': [1.0, 2.0], 'i': [1, 2],
                                                            'a': [0.5]}
    wf = _chain(library, tmp_path / 'wf')
    obs_data = _obs('y_a', 'mod/y', Y_A, STD_A)
    obs_data['data_items'][0].update(data_type='series', obs_dt=0.5,
                                     value=pd.Series([1.0, 2.0]), std=pd.Series([0.1, 0.1]))
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'), obs_data=obs_data,
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.write()
    with open(os.path.join(wf.directory, 'data', 'fit_a_obs_data.json')) as f:
        assert json.load(f)['data_items'][0]['value'] == [1.0, 2.0]
    for bad in (float('nan'), np.float64('inf')):
        with pytest.raises(WorkflowError, match=r'step "fit_a": obs_data holds '
                                                r'data_items\[0\]\.value \((nan|inf)\)'):
            wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                               obs_data=_obs('y_a', 'mod/y', bad, STD_A),
                               params_for_id=_entries('mod', 'p', 0.1, 5.0))
    with pytest.raises(WorkflowError, match=r'params_for_id holds params\[0\]\.min \(nan\)'):
        wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                           obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                           params_for_id={'version': 1, 'params': [
                               {'targets': ['mod/p'], 'min': float('nan'), 'max': 5.0}]})


@pytest.mark.unit
@pytest.mark.parametrize('writer', ['builder', 'funcs_file'])
def test_two_processes_writing_one_file_do_not_share_a_temporary_file(tmp_path, monkeypatch,
                                                                      writer):
    """B6: ranks that build and write the same workflow at once each write through a
    temporary file of their own. Here a second process writes the file while the first is
    between writing its temporary file and renaming it."""
    from libcuflynx.calibration_workflow import builder
    from libcuflynx.utilities import user_funcs_file
    path = str(tmp_path / 'data' / 'file.py')
    sources = {'f': 'def f(x):\n    return x\n'}
    if writer == 'builder':
        def write():
            return builder._write_if_changed(path, 'content\n')
    else:
        def write():
            return user_funcs_file.write_funcs_file(path, 'operation', sources)
    replace, pid, depth = os.replace, os.getpid(), []

    def replace_after_another_process_wrote(src, dst):
        if not depth:
            depth.append(1)
            monkeypatch.setattr(os, 'getpid', lambda: pid + 1)
            write()                                  # the other rank, start to finish
            monkeypatch.setattr(os, 'getpid', lambda: pid)
        return replace(src, dst)
    monkeypatch.setattr(os, 'replace', replace_after_another_process_wrote)
    write()
    assert depth and os.listdir(tmp_path / 'data') == ['file.py']


@pytest.mark.unit
def test_the_launched_mpiexec_keeps_the_users_mpi_settings():
    """B7: run(ranks=N) starts mpiexec from a process that may itself be under a launcher.
    The variables of that job are left out; what the user set is passed on."""
    from libcuflynx.calibration_workflow.builder import launcher_environment
    users = {'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',
             'OMPI_MCA_rmaps_base_oversubscribe': 'true', 'OMPI_MCA_btl': 'self,vader',
             'OMPI_MCA_orte_tmpdir_base': '/scratch',
             'PRTE_MCA_rmaps_default_mapping_policy': ':oversubscribe',
             'HYDRA_HOST_FILE': '/hosts', 'HYDRA_LAUNCHER': 'ssh', 'I_MPI_PIN': '0',
             'OPAL_PREFIX': '/opt/ompi', 'MPIEXEC_TIMEOUT': '60', 'PATH': '/usr/bin'}
    of_the_job = {'OMPI_COMM_WORLD_RANK': '0', 'OMPI_COMM_WORLD_SIZE': '2',
                  'OMPI_MCA_orte_hnp_uri': 'x', 'OMPI_MCA_orte_local_daemon_uri': 'x',
                  'OMPI_MCA_ess': 'pmi', 'OMPI_MCA_ess_base_jobid': '1',
                  'OMPI_MCA_orte_ess_num_procs': '2', 'OMPI_MCA_pmix': '^s1',
                  'OMPI_MCA_initial_wdir': '/', 'OMPI_UNIVERSE_SIZE': '2',
                  'OMPI_APP_CTX_NUM_PROCS': '2', 'OMPI_FIRST_RANKS': '0', 'OMPI_ARGV': 'x',
                  'OMPI_COMMAND': 'python', 'OMPI_FILE_LOCATION': '/tmp/x',
                  'OMPI_NUM_APP_CTX': '1', 'PMIX_RANK': '0', 'PMIX_NAMESPACE': 'x',
                  'PMIX_SERVER_URI41': 'x', 'PMI_RANK': '0', 'PMI_FD': '5', 'PMI_SIZE': '2',
                  'HYDI_CONTROL_FD': '6'}
    assert launcher_environment({**users, **of_the_job}) == users


@pytest.mark.unit
def test_run_refuses_bad_ranks_and_a_missing_mpiexec_with_a_message(library, tmp_path,
                                                                     monkeypatch):
    """B8."""
    wf = _chain(library, tmp_path / 'wf')
    for ranks in ('2', 0, -1, 2.0, True):
        with pytest.raises(WorkflowError, match='ranks is the number of MPI ranks'):
            wf.run(ranks=ranks)
    with pytest.raises(WorkflowError, match='a progress callable needs ranks=1'):
        wf.run(ranks=2, progress=print)
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(WorkflowError, match=r'run\(ranks=2\) needs mpiexec, which is not on '
                                            r'the PATH'):
        wf.run(ranks=2)
    assert not os.path.exists(wf.output_dir)


@pytest.mark.unit
def test_settings_are_checked_where_they_are_given(library, tmp_path):
    """A misspelt setting is refused by the constructor and by update_settings, not later."""
    target = library_module('chain', 'v1', 'rest')
    with pytest.raises(WorkflowError, match=r"\['param_id_metod'\] is not a setting"):
        WorkflowBuilder('typo', target=target, directory=tmp_path / 'wf',
                        settings={'param_id_metod': 'CMA-ES'})
    with pytest.raises(WorkflowError, match='calibration workflow'):
        WorkflowBuilder('  ', target=target, directory=tmp_path / 'wf')
    wf = _chain(library, tmp_path / 'wf')
    before = wf.to_dict()
    with pytest.raises(WorkflowError, match=r"\['dtt'\] is not a setting"):
        wf.update_settings(dtt=1, dt=0.5)
    assert wf.to_dict() == before                        # nothing of a refused call is kept
    wf.update_settings(optimiser_options={'seed': 7})
    assert wf.settings['optimiser_options'] == {**OPTIMISER_SETTINGS['optimiser_options'],
                                                'seed': 7}
    wf.update_settings(dt=None)
    assert 'dt' not in wf.settings
    # one library given as a string is one library
    one = WorkflowBuilder('one', target=target, directory=tmp_path / 'one',
                          module_library_dirs=library['modules'])
    assert one.module_library_dirs == [library['modules']] and one.target == target


@pytest.mark.unit
def test_check_names_what_a_run_still_needs(library, tmp_path):
    """1.7: no module library for a model that names library modules, and the settings the
    calibration method requires, are reported by check() and plan(), by name."""
    arrays = library['arrays']
    wf = WorkflowBuilder('needs', target=module_array(os.path.join(arrays,
                                                                   'chain_module_array.json')),
                         directory=tmp_path / 'wf')
    with pytest.raises(WorkflowError, match='has no steps yet'):
        wf.check()
    wf.add_calibration('c', obs_data=_obs('z_c', 'C/z', Z_C, STD_C),
                       params_for_id=_entries('C', 'q', 0.1, 10.0))
    for check in (wf.check, wf.plan, wf.run):
        with pytest.raises(WorkflowError) as refused:
            check()
        message = str(refused.value)
        assert 'cannot run yet' in message
        assert 'pass module_library_dirs=[' in message and 'chain_module_array.json' in message
        assert "no optimiser_options ['num_calls_to_function'], which genetic_algorithm (the " \
               "param_id_method when none is set) requires" in message
    assert not os.path.exists(wf.output_dir)
    wf.write()                        # the file itself is complete: a library can be given
    #                                   on the command line
    ready = WorkflowBuilder('needs', target=wf.target, directory=tmp_path / 'wf',
                            module_library_dirs=[library['modules']],
                            settings={'param_id_method': 'genetic_algorithm', 'DEBUG': True,
                                      'debug_optimiser_options': {'num_calls_to_function': 9}})
    ready.add_calibration('c', obs_data=_obs('z_c', 'C/z', Z_C, STD_C),
                          params_for_id=_entries('C', 'q', 0.1, 10.0))
    assert ready.check().name == 'needs'
    ready.update_settings(DEBUG=False)
    with pytest.raises(WorkflowError, match='which genetic_algorithm requires'):
        ready.check()
    ready.update_settings(param_id_method='CMA-ES')
    ready.check()


@pytest.mark.unit
def test_the_genetic_algorithm_without_a_budget_says_so():
    """The engine's own message, for a workflow run without the builder."""
    from libcuflynx.param_id import optimisers
    source = inspect.getsource(optimisers)
    assert "needs optimiser_options.num_calls_to_function" in source


@pytest.mark.unit
def test_relative_paths_are_written_with_forward_slashes_on_every_platform():
    from libcuflynx.calibration_workflow.builder import _forward_slashes
    assert _forward_slashes('..\\lib\\arrays\\chain_module_array.json', sep='\\') == \
        '../lib/arrays/chain_module_array.json'
    assert _forward_slashes('../lib/a b.json') == '../lib/a b.json'


@pytest.mark.unit
def test_the_api_page_lists_every_public_member_and_each_has_a_docstring():
    page = os.path.join(os.path.dirname(__file__), '..', 'tutorial', 'docs', 'api',
                        'calibration-workflow.md')
    with open(page) as f:
        text = f.read()
    listed = re.search(r'WorkflowBuilder\n    options:\n      members:\n((?:        - \w+\n)+)',
                       text).group(1)
    listed = [line.strip()[2:] for line in listed.splitlines()]
    public = [name for name in vars(WorkflowBuilder) if not name.startswith('_')]
    assert sorted(listed) == sorted(public)
    for name in public:
        assert len(inspect.getdoc(getattr(WorkflowBuilder, name)) or '') > 20, name
    for attribute in ('directory (pathlib.Path)', 'path (pathlib.Path)', 'settings (dict)'):
        assert attribute in inspect.getdoc(WorkflowBuilder)


@pytest.mark.unit
def test_repr_and_load_of_a_directory(library, tmp_path):
    wf = _chain(library, tmp_path / 'wf')
    wf.add_operation(double_mean)
    text = repr(wf)
    assert text.splitlines()[0] == (f'<WorkflowBuilder "chain_built": target chain/v1/rest, '
                                    f'2 steps, in {wf.directory}>')
    assert '  fit_a (calibration): not run' in text and 'operation functions: double_mean' in text
    path = wf.write()
    assert WorkflowBuilder.load(wf.directory).to_dict() == WorkflowBuilder.load(path).to_dict()
    with pytest.raises(WorkflowError, match='there is no workflow file'):
        WorkflowBuilder.load(tmp_path / 'nowhere')
    with open(path, 'w') as f:
        f.write('{')
    with pytest.raises(WorkflowError, match='is not valid JSON'):
        WorkflowBuilder.load(path)


@pytest.mark.unit
def test_progress_lines_say_what_happened_to_a_step_that_did_not_run(capsys):
    from libcuflynx.scripts.calibration_workflow_run_script import _print_progress
    for fields in ({'has_result': False}, {'has_result': True, 'current': True},
                   {'has_result': True, 'current': False}, {}):
        _print_progress({'event': 'step_skipped', 'step': 's', **fields})
    assert capsys.readouterr().out.splitlines() == [
        '[workflow] step s: not run',
        '[workflow] step s: done and current, not run again',
        '[workflow] step s: reusing its earlier result',
        '[workflow] step s: reusing its earlier result']


@pytest.mark.unit
def test_the_first_progress_line_names_the_steps_that_run_and_those_kept(capsys):
    from libcuflynx.scripts.calibration_workflow_run_script import _print_progress
    order = ['screen', 'top2', 'fit']
    for running in (['top2'], order, []):
        _print_progress({'event': 'workflow_started', 'order': order, 'running': running,
                         'output_dir': 'out'})
    assert capsys.readouterr().out.splitlines() == [
        '[workflow] running 1 of 3 steps: top2; keeping screen, fit',
        '[workflow] running 3 of 3 steps: screen, top2, fit',
        '[workflow] nothing to run: all 3 steps are done and current. To run them all again: '
        'wf.run(force=True), or the command without --stale-only.']


@pytest.mark.unit
def test_dicts_are_copied_and_written_as_files_only_when_they_change(library, tmp_path):
    wf = _chain(library, tmp_path / 'wf')
    obs = _obs('y_a', 'mod/y', Y_A, STD_A)
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'), obs_data=obs,
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    obs['data_items'][0]['value'] = 1.0              # after it was added: no effect (D10)
    wf.write()
    path = os.path.join(wf.directory, 'data', 'fit_a_obs_data.json')
    with open(path) as f:
        assert json.load(f)['data_items'][0]['value'] == Y_A      # numpy value written
    with open(os.path.join(wf.directory, 'data', 'fit_a_params_for_id.json')) as f:
        assert json.load(f) == {'version': 1, 'params': [{'targets': ['mod/p'], 'min': 0.1,
                                                          'max': 5.0}]}
    stamp = os.stat(path).st_mtime_ns
    assert wf.steps == ['fit_a', 'rest']             # re-added in place (D3)
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.write()
    assert os.stat(path).st_mtime_ns == stamp        # same content: not rewritten
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A + 1, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.write()
    with open(path) as f:
        assert json.load(f)['data_items'][0]['value'] == Y_A + 1
    with pytest.raises(Exception, match='obs_data'):
        wf.add_calibration('bad', obs_data={'data_items': []},
                           params_for_id=_entries('mod', 'p', 0.1, 5.0))


@pytest.mark.unit
def test_the_in_code_params_for_id_entries_read_as_the_engine_reads_them(tmp_path):
    """entries_to_document is a copy of PrimitiveParsers._build_param_id_info_from_df's
    conversion: the written document must give the engine what the entries give it."""
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    from libcuflynx.utilities.params_for_id_file import (entries_to_document,
                                                         write_params_for_id)
    from libcuflynx.calibration_workflow.builder import jsonable
    entries = [
        {'vessel_name': 'aortic_root', 'param_name': 'C', 'min': np.float64(1e-9),
         'max': 5e-8, 'name_for_plotting': 'C_{ao}'},
        {'vessel_name': ['heart__lv', 'b'], 'param_name': 'R', 'min': 1.0, 'max': 2.0,
         'prior': 'normal', 'prior_mean': 1.5, 'prior_std': 0.1},
        {'vessel_name': 'a b', 'param_name': 'L', 'min': 1, 'max': 3, 'param_type': 'const',
         'comment': '', 'unbounded': 'false'},
        {'vessel_name': 'aortic_root', 'param_name': 'C', 'min': 1e-9, 'max': 5e-8},
    ]
    parser = ObsAndParamDataParser()
    expected = parser.get_param_id_info_from_entries(entries)
    path = str(tmp_path / 'x_params_for_id.json')
    write_params_for_id(path, jsonable(entries_to_document(entries)))
    got = parser.get_param_id_info(path)
    assert set(got) == set(expected)
    for key in expected:
        assert str(got[key]) == str(expected[key]), key
    assert [e.get('name') for e in entries_to_document(entries)['params']] == [
        None, None, None, 'aortic_root/C#1']


# --------------------------------------------------------------------------------------------
# functions defined in a notebook
# --------------------------------------------------------------------------------------------

@series_to_constant
def double_mean(x, series_output=False):
    '''Twice the mean: fitting lin_a's y = 2 p^2 to Y_A through it gives p = sqrt(Y_A / 4).'''
    if series_output:
        return 2 * x
    return 2 * np.mean(x)


SCALE = 3.0


def uses_a_session_value(x):
    return SCALE * np.mean(x)


def _cell(source, name='<ipython-input-7-0123456789ab>'):
    '''Run ``source`` as IPython runs a cell: compiled under a cell name whose lines it keeps
    in linecache, in a __main__ namespace. Returns the namespace.'''
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = {'__name__': '__main__'}
    exec(compile(source, name, 'exec'), namespace)
    return namespace


CELL = '''
import numpy as np
import scipy.signal as signal

def _peak_count(x):
    return len(signal.find_peaks(x)[0])

def half_mean(x, series_output=False):
    """Half the mean: lin_a's y = 2 p^2 fitted through it to Y_A gives p = sqrt(Y_A)."""
    if series_output:
        return 0.5 * x
    return 0.5 * np.mean(x) + 0 * _peak_count(np.asarray(x))

half_mean.series_to_constant = True
'''


@pytest.mark.unit
def test_a_function_is_written_with_what_it_needs_and_registers_as_an_operation(tmp_path):
    from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
    cell = _cell(CELL)
    wf = WorkflowBuilder('funcs', target=library_module('lin_a', 'v1', 'fit_a'),
                         directory=tmp_path / 'wf')
    wf.add_operation(double_mean)
    with pytest.raises(WorkflowError, match="'_peak_count' is a function defined in this "
                                            "session"):
        wf.add_operation(cell['half_mean'])
    wf.add_operation(cell['half_mean'], cell['_peak_count'])
    source = wf.funcs_source()
    # the marker set after half_mean's def is written as its decorator
    assert '@series_to_constant\ndef half_mean' in source
    assert '@series_to_constant\ndef double_mean' in source
    assert source.startswith(HEADERS['operation'])
    path = os.path.join(wf.directory, 'funcs_operation_funcs.py')
    assert wf.settings['operation_funcs_external_path'] == path
    os.makedirs(wf.directory)
    with open(path, 'w') as f:
        f.write(source)
    funcs = get_operation_funcs_dict_for_mode('numpy', external_path=path)
    x = np.array([1.0, 3.0, 2.0])
    assert funcs['double_mean'](x) == pytest.approx(double_mean(x)) == 4.0
    assert funcs['half_mean'](x) == pytest.approx(cell['half_mean'](x)) == 1.0
    assert funcs['half_mean'].series_to_constant and funcs['double_mean'].series_to_constant
    assert '_peak_count' not in funcs
    # the layout CUFLynx's funcs dialog reads: the listed functions, and the extra imports
    sources, imports = read_funcs_file(path, 'operation')
    assert list(sources) == ['double_mean', 'half_mean', '_peak_count']
    assert imports == ['import scipy.signal as signal']
    assert 'CUFLYNX_OPERATIONS = ["double_mean", "half_mean", "_peak_count"]' in source

    # a builder loaded from the file keeps the functions and can add more
    wf.add_calibration('fit_a', obs_data='x.json', params_for_id='y.json')
    wf.write()
    again = WorkflowBuilder.load(wf.path)
    again.add_operation(double_mean)
    assert again.funcs_source() == source


def _funcs_builder(tmp_path, name='funcs'):
    wf = WorkflowBuilder(name, target=library_module('lin_a', 'v1', 'fit_a'),
                         directory=tmp_path / 'wf')
    wf.add_calibration('fit_a', obs_data='x.json', params_for_id='y.json')
    return wf


def extra_mean(x, series_output=False):
    return x if series_output else np.mean(x)


@pytest.mark.unit
def test_a_second_builder_regenerates_a_functions_file_nobody_edited(tmp_path):
    """Running a script again, or Restart & Run All (1.1): the file carries the checksum of
    what was generated, so a builder that never saw it may write it again."""
    from libcuflynx.utilities.user_funcs_file import CHECKSUM_PREFIX, is_unedited
    wf = _funcs_builder(tmp_path)
    wf.add_operation(double_mean)
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    with open(funcs) as f:
        first = f.read()
    assert first.count(CHECKSUM_PREFIX) == 1 and is_unedited(first)

    again = _funcs_builder(tmp_path)
    again.add_operation(double_mean)            # the same file: nothing to refuse
    again.write()
    with open(funcs) as f:
        assert f.read() == first
    # and with other functions: the unedited file is replaced
    third = _funcs_builder(tmp_path)
    third.add_operation(double_mean, extra_mean)
    third.write()
    with open(funcs) as f:
        text = f.read()
    assert 'def extra_mean' in text and is_unedited(text)
    # the original builder lacks extra_mean, which the file now holds: it is refused
    # until told to replace the file
    with pytest.raises(WorkflowError, match=r"\['extra_mean'\].*overwrite=True"):
        wf.write()
    wf.write(overwrite=True)
    with open(funcs) as f:
        assert f.read() == first


@pytest.mark.unit
def test_a_functions_file_edited_by_hand_is_refused_with_an_instruction_that_works(tmp_path):
    """1.2: the refusal names overwrite=True, and add_operation(f, overwrite=True) or
    write(overwrite=True) then goes through."""
    from libcuflynx.utilities.user_funcs_file import is_unedited
    wf = _funcs_builder(tmp_path)
    wf.add_operation(double_mean)
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    with open(funcs, 'a') as f:
        f.write('\n\ndef by_hand(x):\n    return x\n')
    with open(funcs) as f:
        assert not is_unedited(f.read())
    message = (re.escape(funcs) + r' was edited by hand.*wf\.add_operation\(f, overwrite=True\)'
               r'.*wf\.write\(overwrite=True\)')
    for refused in (wf.write, wf.status, lambda: wf.add_operation(extra_mean),
                    lambda: _funcs_builder(tmp_path).add_operation(extra_mean)):
        with pytest.raises(WorkflowError, match=message):
            refused()
        with open(funcs) as f:
            assert 'def by_hand' in f.read()             # not overwritten
    # adding what gives the file as it is on disk is never refused
    wf.add_operation(double_mean, overwrite=True)
    wf.write()
    with open(funcs) as f:
        text = f.read()
    assert 'def by_hand' not in text and is_unedited(text)

    # write(overwrite=True) does the same
    with open(funcs, 'a') as f:
        f.write('\n# a note\n')
    with pytest.raises(WorkflowError, match='was edited by hand'):
        wf.add_operation(extra_mean)
    wf.add_operation(extra_mean, overwrite=True)
    with open(funcs) as f:
        assert '# a note' in f.read()                    # add_operation writes nothing
    wf.write()
    with open(funcs) as f:
        assert '# a note' not in f.read()
    # a file no builder wrote is refused the same way
    with open(funcs, 'w') as f:
        f.write('def mine(x):\n    return x\n')
    with pytest.raises(WorkflowError, match=message):
        _funcs_builder(tmp_path).add_operation(double_mean)
    fresh = _funcs_builder(tmp_path)
    fresh.add_operation(double_mean, overwrite=True)
    fresh.write()
    assert fresh.functions() == ['double_mean']


@pytest.mark.unit
def test_a_fresh_builder_does_not_drop_functions_another_writer_added(tmp_path):
    """A functions file with a valid checksum that holds a function this builder lacks (added
    through write_funcs_file) is refused; a re-run with the same functions is not."""
    from libcuflynx.utilities.user_funcs_file import write_funcs_file
    wf = _funcs_builder(tmp_path)
    wf.add_operation(double_mean)
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    # the same functions again (a notebook re-run after a restart), and more, need no overwrite
    for functions in ((double_mean,), (double_mean, extra_mean)):
        again = _funcs_builder(tmp_path)
        again.add_operation(*functions)
        again.write()
    sources, imports = read_funcs_file(funcs, 'operation')
    sources['dialog_op'] = '@series_to_constant\ndef dialog_op(x, series_output=False):\n' \
                           '    return np.max(x)\n'
    write_funcs_file(funcs, 'operation', sources, imports)
    fresh = _funcs_builder(tmp_path)
    fresh.add_operation(double_mean, extra_mean)       # adding never refuses: a later add may supply it
    with pytest.raises(WorkflowError, match=r"\['dialog_op'\].*overwrite=True"):
        fresh.write()
    assert 'dialog_op' in read_funcs_file(funcs, 'operation')[0]
    fresh.add_operation(double_mean, extra_mean, overwrite=True)
    fresh.write()
    assert 'dialog_op' not in read_funcs_file(funcs, 'operation')[0]


@pytest.mark.unit
def test_a_function_added_by_a_later_cell_is_not_refused_when_an_earlier_cell_adds(tmp_path):
    """Restart & Run All: an early cell adds A, a later one adds my_max; the file already holds
    both from the last run. Nothing is refused until write(), by when both are added."""
    first = _funcs_builder(tmp_path)
    first.add_operation(double_mean, extra_mean)
    first.write()
    funcs = first.settings['operation_funcs_external_path']
    fresh = _funcs_builder(tmp_path)
    fresh.add_operation(double_mean)                   # does not raise
    fresh.add_operation(extra_mean)
    fresh.write()                                      # no overwrite needed
    assert set(read_funcs_file(funcs, 'operation')[0]) == {'double_mean', 'extra_mean'}


@pytest.mark.unit
def test_load_keeps_a_hand_edited_functions_file_and_never_drops_its_statements(tmp_path):
    """1.3: load() reads a hand-edited file's functions and may rewrite it, but not when the
    file holds statements a rewrite would drop (a constant, a class)."""
    wf = _funcs_builder(tmp_path)
    wf.add_operation(double_mean)
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    with open(funcs, 'a') as f:
        f.write('\n\ndef by_hand(x):\n    return x\n')
    again = WorkflowBuilder.load(wf.directory)           # the directory, or the file
    assert again.functions() == ['double_mean', 'by_hand']
    again.add_operation(extra_mean)
    again.write()
    with open(funcs) as f:
        assert 'def by_hand' in f.read() and again.functions()[-1] == 'extra_mean'

    with open(funcs) as f:
        text = f.read()
    text = text.replace('CUFLYNX_OPERATIONS', 'GAIN = 2.0  # mine\n\nclass Shape:\n    pass\n\n'
                        'CUFLYNX_OPERATIONS')
    with open(funcs, 'w') as f:
        f.write(text)
    loaded = WorkflowBuilder.load(wf.path)
    loaded.write()                                       # nothing to rewrite: the file stays
    assert loaded.status is not None
    with open(funcs) as f:
        assert f.read() == text
    with pytest.raises(WorkflowError, match=r'GAIN = 2\.0  # mine \(line \d+\); class Shape: '
                                            r'\(line \d+\)'):
        loaded.remove_operation('by_hand')
        loaded.write()
    with open(funcs) as f:
        assert f.read() == text
    loaded.write(overwrite=True)
    with open(funcs) as f:
        assert 'GAIN' not in f.read()


@pytest.mark.unit
def test_cuflynx_reads_and_rewrites_a_file_with_the_checksum_line(tmp_path):
    """CUFLynx's funcs dialog (apps/api/user_funcs.py) reads a file by its list marker and its
    top-level defs and rewrites it with write_funcs_file: the checksum line is a comment to
    both, and the rewritten file carries its own."""
    import ast
    from libcuflynx.utilities.user_funcs_file import (LIST_MARKERS, is_unedited,
                                                      unkept_statements, write_funcs_file)
    wf = _funcs_builder(tmp_path)
    wf.add_operation(double_mean, extra_mean)
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    with open(funcs) as f:
        text = f.read()
    tree = ast.parse(text)                               # as CUFLynx's _parse_existing does
    order = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and n.targets[0].id == LIST_MARKERS['operation']][0]
    assert order == ['double_mean', 'extra_mean']
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == order
    assert unkept_statements(funcs, 'operation') == []
    sources, imports = read_funcs_file(funcs, 'operation')
    write_funcs_file(funcs, 'operation', {'extra_mean': sources['extra_mean']}, imports)
    with open(funcs) as f:
        rewritten = f.read()
    assert is_unedited(rewritten) and 'def double_mean' not in rewritten
    # with Windows line ends the checksum still holds
    assert is_unedited(rewritten.replace('\n', '\r\n'))
    # so a builder may regenerate what CUFLynx wrote
    wf.add_operation(double_mean)
    wf.write()


@pytest.mark.unit
def test_functions_are_added_all_or_none_removed_and_listed(tmp_path):
    """B5: a call that fails on one function adds none, and does not name a file."""
    wf = _funcs_builder(tmp_path)
    with pytest.raises(WorkflowError, match='lambda'):
        wf.add_operation(double_mean, lambda x: x)
    assert wf.functions() == [] and 'operation_funcs_external_path' not in wf.settings
    assert wf.funcs_source() is None
    with pytest.raises(WorkflowError, match=r"takes a function, not a name \('mean'\)"):
        wf.add_operation('mean')
    wf.add_operation(double_mean, extra_mean)
    assert wf.functions() == ['double_mean', 'extra_mean'] and wf.functions('cost') == []
    wf.remove_operation('double_mean')
    assert wf.functions() == ['extra_mean'] and 'def double_mean' not in wf.funcs_source()
    with pytest.raises(WorkflowError, match=r"no operation function \['double_mean'\]"):
        wf.remove_operation(double_mean)
    wf.remove_operation(extra_mean)
    assert 'operation_funcs_external_path' not in wf.settings
    assert 'operation_funcs_external_path' not in wf.to_dict().get('settings', {})


K_SESSION = 3.0


class SessionType:
    pass


def default_from_the_session(x, k=K_SESSION, series_output=False):
    return x if series_output else k * np.mean(x)


def annotated_with_a_session_type(x: SessionType, series_output=False) -> SessionType:
    return x if series_output else np.mean(x)


def default_from_a_module(x, k=np.pi, *, scale: float = os.sep, series_output=False):
    return x if series_output else k * np.mean(x)


@pytest.mark.unit
def test_a_default_or_annotation_using_a_session_name_is_refused(tmp_path):
    """B1: the def statement evaluates its defaults and annotations where the file is
    imported, so the names in them are needed there too."""
    with pytest.raises(WorkflowError, match="'K_SESSION' is a float defined in this session"):
        function_source(default_from_the_session)
    with pytest.raises(WorkflowError, match="'SessionType' is a type defined in this session"):
        function_source(annotated_with_a_session_type)
    source, imports = function_source(default_from_a_module)
    assert imports == ['import os']
    path = tmp_path / 'ops.py'
    from libcuflynx.utilities.user_funcs_file import write_funcs_file
    write_funcs_file(str(path), 'operation', {'default_from_a_module': source}, imports)
    namespace = {}
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    assert namespace['default_from_a_module'](np.ones(3)) == pytest.approx(np.pi)


SUBMODULE_CELL = """
import numpy as np
import wsgiref.util
import xml.dom.minidom as minidom
import email

def through_its_package(x, series_output=False):
    wsgiref.util.FileWrapper, minidom.parseString, email.mime.text.MIMEText
    return x if series_output else np.mean(x)
"""


@pytest.mark.unit
def test_a_submodule_used_through_its_package_is_imported(tmp_path):
    """B2: ``import pkg`` does not load ``pkg.sub``; the file imports the submodule."""
    import email.mime.text  # noqa: F401 - loaded, so the session's ``email`` has it
    import subprocess
    import sys
    cell = _cell(SUBMODULE_CELL, name='<ipython-input-3-00aa00aa00aa>')
    source, imports = function_source(cell['through_its_package'])
    assert imports == ['import email', 'import email.mime.text', 'import wsgiref',
                       'import wsgiref.util', 'import xml.dom.minidom as minidom']
    from libcuflynx.utilities.user_funcs_file import write_funcs_file
    path = write_funcs_file(str(tmp_path / 'ops.py'), 'operation',
                            {'through_its_package': source}, imports)
    done = subprocess.run(
        [sys.executable, '-c', 'import runpy, sys; import numpy as np; '
         'print(runpy.run_path(sys.argv[1])["through_its_package"](np.ones(3)))', path],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '1.0'


@pytest.mark.unit
def test_functions_that_cannot_live_in_a_file_are_refused(tmp_path):
    wf = WorkflowBuilder('funcs', target=library_module('lin_a', 'v1', 'fit_a'),
                         directory=tmp_path / 'wf')
    with pytest.raises(WorkflowError, match='lambda'):
        wf.add_operation(lambda x: x)
    offset = 1.0

    def closure(x):
        return x + offset
    with pytest.raises(WorkflowError, match=r"\['offset'\] from the function"):
        wf.add_operation(closure)
    with pytest.raises(WorkflowError, match="'SCALE' is a float defined in this session"):
        wf.add_operation(uses_a_session_value)
    namespace = {'__name__': '__main__'}
    exec('def no_source(x):\n    return x\n', namespace)
    with pytest.raises(WorkflowError, match='source of no_source cannot be found'):
        wf.add_operation(namespace['no_source'])
    # one functions file per workflow (D5)
    wf.update_settings(operation_funcs_external_path=str(tmp_path / 'mine.py'))
    with pytest.raises(WorkflowError, match='already names operation_funcs_external_path'):
        wf.add_operation(double_mean)
    assert 'def double_mean' in function_source(double_mean)[0]


# --------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_run_only_from_step_and_edited_dicts_make_their_steps_stale(library, tmp_path):
    wf = _chain(library, tmp_path / 'wf')
    wf.run(only='fit_a', progress=False)
    status = wf.status()['steps']
    assert status['fit_a']['status'] == 'done' and status['rest']['status'] == 'not_run'
    wf.run(from_step='rest', progress=False)
    assert wf.status()['complete']
    assert _calibrated(wf, 'fit_a') == pytest.approx(2.0, rel=1e-4)
    assert _calibrated(wf, 'rest') == pytest.approx(3.0, rel=1e-4)

    # the same dict again (re-running a cell) changes nothing
    wf.add_calibration('rest', obs_data=_obs('z_c', 'mod__C/z', Z_C, STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0), fixed_from='fit_a')
    assert _stale(wf) == {'fit_a': False, 'rest': False}
    # an edited dict makes its step stale, and the steps downstream of it
    wf.add_calibration('rest', obs_data=_obs('z_c', 'mod__C/z', Z_C + 1, STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0), fixed_from='fit_a')
    assert _stale(wf) == {'fit_a': False, 'rest': True}
    wf.add_calibration('fit_a', model=library_module('lin_a', 'v1', 'fit_a'),
                       obs_data=_obs('y_a', 'mod/y', Y_A, STD_A),
                       params_for_id=_entries('mod', 'p', 0.1, 4.0))
    assert _stale(wf) == {'fit_a': True, 'rest': True}
    with pytest.raises(WorkflowError, match='has changed since "fit_a" ran'):
        wf.run(only='rest', progress=False)

    images = wf.plot_step('fit_a')
    assert images and all(os.path.isfile(p) for p in images)
    model = wf.model('rest')
    assert os.path.isfile(model['model_path'])
    # generated in the run's models/ folder, which no step's output can be
    assert str(model['model_path']).startswith(
        os.path.join(wf.output_dir, MODELS_DIR, 'rest') + os.sep)


@pytest.mark.integration
def test_run_keeps_the_steps_that_are_done_and_current(library, tmp_path, capsys):
    """1.5: run() with no only/from_step runs what has not run or is stale, and says which
    steps it kept; force=True runs everything. result() reads what a step recorded."""
    from libcuflynx.scripts import calibration_workflow_run_script as cli
    wf = _chain(library, tmp_path / 'wf')
    with pytest.raises(WorkflowError, match=f'step "fit_a" has not run into '
                                            f'{re.escape(str(wf.output_dir))}'):
        wf.result('fit_a')
    events = []
    wf.run(only='fit_a', progress=events.append)
    # a step that never ran is not said to be reused
    assert [(e['step'], e['has_result']) for e in events if e['event'] == 'step_skipped'] == [
        ('rest', False)]

    def ran(**kwargs):
        events.clear()
        result = wf.run(progress=events.append, **kwargs)
        assert result['complete']
        return ([e['step'] for e in events if e['event'] == 'step_started'],
                [e['step'] for e in events if e['event'] == 'step_skipped' and e['current']])

    assert ran() == (['rest'], ['fit_a'])
    finished = {s: v['finished'] for s, v in wf.status()['steps'].items()}
    values = {s: _calibrated(wf, s) for s in wf.steps}
    assert ran() == ([], ['fit_a', 'rest'])
    assert {s: v['finished'] for s, v in wf.status()['steps'].items()} == finished
    assert {s: _calibrated(wf, s) for s in wf.steps} == values

    # the default progress prints one line for each step kept
    capsys.readouterr()
    wf.run()
    out = capsys.readouterr().out
    assert '[workflow] step fit_a: done and current, not run again' in out
    assert '[workflow] step rest: done and current, not run again' in out

    # the result a step recorded
    fit_a = wf.result('fit_a')
    assert fit_a['kind'] == 'calibration' and fit_a['values'] == {'mod/p': values['fit_a']}
    assert fit_a['best_cost'] == wf.status()['steps']['fit_a']['best_cost']
    assert fit_a['step_dir'] == wf.step_dir('fit_a')
    assert os.path.isdir(fit_a['param_id_output_dir'])
    with pytest.raises(WorkflowError, match='has no step "nope"'):
        wf.result('nope')

    # an edited step and the steps after it run again; the ones before are kept
    wf.add_calibration('rest', obs_data=_obs('z_c', 'mod__C/z', Z_C, 2 * STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0), fixed_from='fit_a')
    assert ran() == (['rest'], ['fit_a'])
    # the command line's --stale-only is the same; without it every step runs
    assert cli.build_parser().parse_args([str(wf.path), '--stale-only']).stale_only
    assert not cli.build_parser().parse_args([str(wf.path)]).stale_only
    assert ran(force=True) == (['fit_a', 'rest'], [])
    assert _calibrated(wf, 'fit_a') == pytest.approx(values['fit_a'], rel=1e-12)


@pytest.mark.integration
def test_result_of_a_reduction_names_the_chosen_parameters(library, tmp_path):
    wf = WorkflowBuilder('reduced', target=library_module('chain', 'v1', 'rest'),
                         directory=tmp_path / 'wf', module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_sensitivity('screen', obs_data=_obs('z_c', 'mod__C/z', Z_C, STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0)
                       + _entries('mod__A', 'p', 1.9, 2.1), sa_options=SA_OPTIONS)
    wf.add_reduction('keep', sensitivity_from='screen', method='sobol', n_params=1)
    wf.run(progress=False)
    keep = wf.result('keep')
    assert keep['kind'] == 'reduction' and len(keep['parameters']) == 1
    assert keep['parameters'] == [row['parameter'] for row in keep['selected']]
    assert os.path.isfile(keep['params_for_id']) and os.path.isfile(keep['report'])
    assert os.path.isdir(wf.result('screen')['sensitivity_dir'])
    with pytest.raises(WorkflowError, match='is a reduction step; plot_step plots'):
        wf.plot_step('keep')


@pytest.mark.integration
def test_a_notebook_operation_reaches_the_run_and_editing_its_file_makes_steps_stale(
        library, tmp_path):
    cell = _cell(CELL, name='<ipython-input-8-fedcba987654>')
    wf = WorkflowBuilder('ops', target=library_module('lin_a', 'v1', 'fit_a'),
                         directory=tmp_path / 'wf', module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_operation(double_mean, cell['half_mean'], cell['_peak_count'])
    wf.add_calibration('double', obs_data=_obs('y_a', 'mod/y', Y_A, STD_A, 'double_mean'),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.add_calibration('half', obs_data=_obs('y_a', 'mod/y', Y_A, STD_A, 'half_mean'),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    wf.run(progress=False)
    assert _calibrated(wf, 'double') == pytest.approx(np.sqrt(Y_A / 4), rel=1e-4)
    assert _calibrated(wf, 'half') == pytest.approx(np.sqrt(Y_A), rel=1e-4)
    assert _stale(wf) == {'double': False, 'half': False}

    # a function edited in its cell and added again makes the steps stale
    edited = _cell(CELL.replace('0 * _peak_count', '0.0 * _peak_count'),
                   name='<ipython-input-9-abcdefabcdef>')
    wf.add_operation(edited['half_mean'])
    assert _stale(wf) == {'double': True, 'half': True}

    # so does editing the file itself, for a run of the file (the command line, CUFLynx)
    path = wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    wf.add_operation(cell['half_mean'])
    wf.write()
    assert not any(v['stale'] for v in workflow_status(path, wf.output_dir)['steps'].values())
    with open(funcs, 'a') as f:
        f.write('\n# edited\n')
    assert all(v['stale'] for v in workflow_status(path, wf.output_dir)['steps'].values())
    # a result recorded before the functions file was an input is not stale for it
    for step in wf.steps:
        result_path = wf.step_dir(step) / STEP_RESULT
        with open(result_path) as f:
            result = json.load(f)
        del result['inputs']['operation_funcs']
        with open(result_path, 'w') as f:
            json.dump(result, f)
    assert not any(v['stale'] for v in workflow_status(path, wf.output_dir)['steps'].values())

    # the files a CUFLynx export carries: the workflow, its data and its functions
    pack = tmp_path / 'pack'
    packed = pack_workflow(path, str(pack))
    sources = set(packed['sources'])
    assert os.path.realpath(funcs) in sources
    for name in ('double_obs_data.json', 'double_params_for_id.json', 'half_obs_data.json'):
        assert os.path.realpath(os.path.join(wf.directory, 'data', name)) in sources
    # a packed workflow (as a CUFLynx archive holds it) loads with its packed paths kept
    with open(packed['workflow']) as f:
        raw = json.load(f)
    assert raw['steps'][0]['obs_data'].startswith('inputs/')
    assert WorkflowBuilder.load(packed['workflow']).to_dict() == raw


@pytest.mark.integration
def test_two_ranks_run_the_written_workflow_with_its_operation(library, tmp_path):
    """run(ranks=2) launches its own mpiexec, so this runs on one rank (not marked mpi, which
    the suite runs under mpiexec -n 2)."""
    wf = WorkflowBuilder('ranks', target=library_module('lin_a', 'v1', 'fit_a'),
                         directory=tmp_path / 'wf', module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_operation(double_mean)
    wf.add_calibration('double', obs_data=_obs('y_a', 'mod/y', Y_A, STD_A, 'double_mean'),
                       params_for_id=_entries('mod', 'p', 0.1, 5.0))
    result = wf.run(ranks=2)
    assert result['complete']
    assert _calibrated(wf, 'double') == pytest.approx(np.sqrt(Y_A / 4), rel=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['operation', 'cost', 'modifier'])
def test_each_kind_of_functions_file_defines_its_header_names_and_registers(kind, tmp_path):
    """utilities.user_funcs_file is the one layout of a functions file (CUFLynx's dialog
    writes it too): its header defines the names HEADER_NAMES says, and libcuflynx registers
    the functions below it, not the header's decorators."""
    from libcuflynx.calibration_workflow.builder import _is
    from libcuflynx.utilities.user_funcs_file import HEADER_NAMES, write_funcs_file
    namespace = {}
    exec(HEADERS[kind], namespace)
    for name, spec in HEADER_NAMES[kind].items():
        assert _is(namespace[name], spec), name
    sources = {
        'operation': {'span': '@series_to_constant\ndef span(x, series_output=False):\n'
                              '    return x if series_output else np.max(x) - np.min(x)\n'},
        'cost': {'abs_cost': '@differentiable\ndef abs_cost(output, desired_mean):\n'
                             '    return mb.sum(mb.abs(output - desired_mean))\n'},
        'modifier': {'shift': '@modifier_func(inputs={})\ndef shift(theta, baseline):\n'
                              '    return baseline + theta\n'}}[kind]
    path = write_funcs_file(str(tmp_path / f'{kind}_funcs.py'), kind, sources)
    if kind == 'operation':
        from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
        funcs = get_operation_funcs_dict_for_mode('numpy', external_path=path)
        assert funcs['span'](np.array([1.0, 4.0])) == 3.0
        assert 'series_to_constant' not in funcs and 'differentiable' not in funcs
    elif kind == 'cost':
        from libcuflynx.funcs.cost_funcs_user import cost_func_metadata
        metadata = cost_func_metadata(external_path=path)
        assert metadata['abs_cost']['differentiable'] and 'is_MLE' not in metadata
    else:
        from libcuflynx.param_id.modifier_funcs import get_modifier_funcs
        funcs = get_modifier_funcs(path)
        assert 'shift' in funcs and 'modifier_func' not in funcs
    assert read_funcs_file(path, kind) == (sources, [])


def _rule(operands, tangents):
    return tangents[0].mean()


def mean_with_rule(x):
    return np.mean(x)


mean_with_rule.exact_tangent = _rule


@pytest.mark.unit
def test_an_exact_tangent_given_after_the_def_is_written_as_its_decorator():
    source, imports = function_source(mean_with_rule, 'operation',
                                      known={'mean_with_rule', '_rule'})
    assert source.startswith('@exact_tangent(_rule)\ndef mean_with_rule(x):')
    assert imports == ['from libcuflynx.param_id.operation_funcs import exact_tangent']


# --------------------------------------------------------------------------------------------
# options as dicts, paths, logs, clean, show_*, pack overwrite, plot formats
# --------------------------------------------------------------------------------------------

def _screened(library, directory):
    screened = library['chain/screened']
    wf = WorkflowBuilder('screened', target=library_module('chain', 'v1', 'screened'),
                         directory=directory, module_library_dirs=[library['modules']],
                         settings=OPTIMISER_SETTINGS)
    wf.add_sensitivity('screen', obs_data=os.path.join(screened, 'screened_obs_data.json'),
                       params_for_id=os.path.join(screened, 'screened_params_for_id.json'),
                       sa_options=SA_OPTIONS)
    return wf


def _reduction_options(wf, step='reduce'):
    return next(s for s in wf.to_dict()['steps']
                if s['id'] == step)['settings']['param_reduction_options']


@pytest.mark.unit
def test_add_reduction_takes_its_options_as_a_dict_and_the_keywords_warn(library, tmp_path):
    wf = _screened(library, tmp_path / 'wf')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        wf.add_reduction('reduce', sensitivity_from='screen',
                         param_reduction_options={'n_params': 1, 'redundancy_check': True})
    # method is pca unless given
    assert _reduction_options(wf) == {'method': 'pca', 'n_params': 1, 'redundancy_check': True}
    wf.add_reduction('reduce', sensitivity_from='screen')
    assert _reduction_options(wf) == {'method': 'pca'}

    # the keywords of the first version give the same step, with a FutureWarning
    with pytest.warns(FutureWarning, match=r"param_reduction_options=\{'method': 'sobol', "
                                           r"'n_params': 1\}"):
        wf.add_reduction('reduce', sensitivity_from='screen', method='sobol', n_params=1)
    assert _reduction_options(wf) == {'method': 'sobol', 'n_params': 1}
    with pytest.warns(FutureWarning):
        wf.add_reduction('reduce', sensitivity_from='screen', threshold=0.05,
                         param_reduction_options={'method': 'sobol', 'n_params': 1})
    assert _reduction_options(wf) == {'method': 'sobol', 'n_params': 1, 'threshold': 0.05}

    # an option given both ways with different values is refused; the step is unchanged
    with pytest.warns(FutureWarning), pytest.raises(WorkflowError, match=r"\['n_params'\] is "
                                                                         r"given both"):
        wf.add_reduction('reduce', sensitivity_from='screen', n_params=2,
                         param_reduction_options={'method': 'sobol', 'n_params': 1})
    assert _reduction_options(wf)['n_params'] == 1
    # an option the schema does not have is refused where it is given
    with pytest.raises(WorkflowError, match='n_parms'):
        wf.add_reduction('reduce', sensitivity_from='screen',
                         param_reduction_options={'n_parms': 1})


@pytest.mark.unit
def test_paths_are_returned_as_path_objects(library, tmp_path):
    from pathlib import Path
    wf = _chain(library, str(tmp_path / 'wf'))
    assert wf.directory == tmp_path / 'wf' and isinstance(wf.directory, Path)
    assert wf.path == tmp_path / 'wf' / 'calibration_workflow.json'
    assert wf.output_dir == tmp_path / 'wf' / 'workflow_output' / 'chain_built'
    assert isinstance(wf.output_dir, Path)
    written = wf.write()
    assert written == wf.path and isinstance(written, Path) and written.is_file()
    assert wf.step_dir('fit_a') == wf.output_dir / 'fit_a'
    loaded = WorkflowBuilder.load(written)
    assert loaded.directory == wf.directory and isinstance(loaded.path, Path)


@pytest.mark.unit
def test_a_run_given_no_output_directory_goes_beside_the_workflow_file(library, tmp_path,
                                                                        monkeypatch):
    """The command (and run_calibration_workflow) run a written file into the builder's
    output_dir, wherever they are run from."""
    from libcuflynx.calibration_workflow.runner import default_output_dir
    wf = _chain(library, tmp_path / 'wf')
    path = wf.write()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert default_output_dir(load_workflow(path)) == str(wf.output_dir)
    # a workflow given as a dict has no file: the working directory
    in_memory = load_workflow({**wf.to_dict(), 'target': wf.target,
                               'module_library_dirs': wf.module_library_dirs,
                               'steps': [{'id': 'by_hand', 'kind': 'set', 'set': [
                                   {'parameter': 'C/k', 'value': 1.0, 'reason': 'one'}]}]})
    assert default_output_dir(in_memory) == str(elsewhere / 'workflow_output' / 'chain_built')


@pytest.mark.unit
def test_clean_lists_and_moves_only_unused_files_the_builder_wrote(library, tmp_path, capsys):
    wf = _chain(library, tmp_path / 'wf')
    wf.write()
    data = wf.directory / 'data'
    mine = data / 'my_notes.json'
    mine.write_text('{"mine": true}')
    result_file = wf.output_dir / 'rest' / 'step_result.json'
    result_file.parent.mkdir(parents=True)
    result_file.write_text('{}')
    assert wf.clean() == []
    assert 'nothing to clean' in capsys.readouterr().out

    # a removed step's data files are no longer used: listed, and left where they are
    wf.remove_step('rest')
    wf.write()
    unused = [data / 'rest_obs_data.json', data / 'rest_params_for_id.json']
    assert wf.clean() == unused and all(p.is_file() for p in unused)
    out = capsys.readouterr().out
    assert 'data/rest_obs_data.json' in out and 'wf.clean(delete=True) moves them' in out

    # a file another workflow in the directory names is kept
    other = wf.directory / 'other_workflow.json'
    other.write_text(json.dumps({'workflow_name': 'other', 'steps': [
        {'id': 'rest', 'obs_data': 'data/rest_obs_data.json'}]}))
    assert wf.clean() == [data / 'rest_params_for_id.json']
    # so is one that is not as the builder wrote it
    written = unused[1].read_text()
    unused[1].write_text(written + ' ')
    assert wf.clean() == []
    assert 'kept data/rest_params_for_id.json: changed since' in capsys.readouterr().out
    unused[1].write_text(written)
    # a JSON file that cannot be read may name anything: nothing is cleaned
    broken = wf.directory / 'broken.json'
    broken.write_text('{')
    with pytest.raises(WorkflowError, match='broken.json cannot be read'):
        wf.clean(delete=True)
    broken.unlink()
    other.unlink()

    # delete=True moves the files into data/removed/, and deletes nothing
    moved = wf.clean(delete=True)
    removed = data / 'removed' / 'data'
    assert moved == [removed / 'rest_obs_data.json', removed / 'rest_params_for_id.json']
    assert all(p.is_file() for p in moved) and not any(p.exists() for p in unused)
    assert moved[1].read_text() == written
    assert mine.read_text() == '{"mine": true}' and result_file.is_file()
    assert (data / 'fit_a_obs_data.json').is_file()
    assert wf.clean() == [] and wf.clean(delete=True) == []
    # the workflow still writes and reads as before
    assert load_workflow(wf.write()).step('fit_a').obs_data == str(data / 'fit_a_obs_data.json')

    # a second builder of the same workflow cleans what the first wrote
    again = WorkflowBuilder.load(wf.path)
    again.add_calibration('extra', model=library_module('lin_a', 'v1', 'fit_a'),
                          obs_data=_obs('y_a', 'mod/y', Y_A + 1, STD_A),
                          params_for_id=_entries('mod', 'p', 0.1, 4.0))
    again.write()
    assert WorkflowBuilder.load(wf.path).clean() == []
    again.remove_step('extra')
    assert again.clean() == []          # the file on disk still names them, until write()
    again.write()
    assert WorkflowBuilder.load(wf.path).clean() == [data / 'extra_obs_data.json',
                                                     data / 'extra_params_for_id.json']


@pytest.mark.unit
def test_show_plan_and_show_status_print_a_line_for_each_step(library, tmp_path, capsys):
    wf = _chain(library, tmp_path / 'wf')
    assert wf.show_plan() is None
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith('[workflow] chain_built: 2 steps, in run order')
    assert lines[1].split() == ['step', 'kind', 'model', 'takes', 'from', 'gives']
    assert lines[2].split()[:3] == ['fit_a', 'calibration', 'lin_a/v1/fit_a']
    assert lines[3].split()[:4] == ['rest', 'calibration', 'chain/v1/rest', 'fit_a']

    assert wf.show_status() is None
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (f'[workflow] chain_built: 0 of 2 steps are done and current, in '
                        f'{wf.output_dir}')
    assert lines[1].split() == ['step', 'kind', 'status', 'best', 'cost', 'finished']
    assert [line.split() for line in lines[2:]] == [['fit_a', 'calibration', 'not', 'run'],
                                                    ['rest', 'calibration', 'not', 'run']]


@pytest.mark.unit
def test_the_launched_commands_lines_go_to_each_steps_log(tmp_path):
    """run(ranks=N) reads the command's output: the [workflow] lines are shown and start and
    end each step's log; the rest goes to the log only, unless verbose."""
    from libcuflynx.calibration_workflow.builder import _RunLog
    lines = ['[workflow] running 2 of 2 steps: fit, fit 2\n',
             '[workflow] step 1/2: fit (calibration, a.json)\n', 'generation 1\n',
             'a warning\n', '[workflow] step fit done: best cost 1 in 2.0 s\n',
             'between the steps\n',
             '[workflow] step 2/2: fit 2 (calibration, a.json)\n', 'generation 9\n',
             '[workflow] step fit 2 FAILED: no\n', 'Traceback\n']
    for verbose in (False, True):
        shown, out = [], tmp_path / str(verbose)
        log = _RunLog(str(out), echo=shown.append, verbose=verbose, steps=['fit', 'fit 2'])
        for line in lines:
            log.line(line)
        log.close()
        assert shown == (lines if verbose else [x for x in lines if x.startswith('[workflow]')])
        assert (out / 'fit' / 'log.txt').read_text() == 'generation 1\na warning\n'
        assert (out / 'fit 2' / 'log.txt').read_text() == 'generation 9\n'
        assert (out / 'workflow_log.txt').read_text() == ''.join(
            x for x in lines if x.startswith('[workflow]') or x in ('between the steps\n',
                                                                    'Traceback\n'))
        assert sorted(p.name for p in out.iterdir()) == ['fit', 'fit 2', 'workflow_log.txt']


@pytest.mark.unit
def test_pack_workflow_replaces_an_earlier_pack_only_when_told_to(library, tmp_path):
    path = _chain(library, tmp_path / 'wf').write()
    pack = tmp_path / 'pack'
    first = pack_workflow(path, str(pack))
    with pytest.raises(ValueError, match='is not empty; pack into a new directory, or replace '
                                         'an earlier pack with overwrite=True'):
        pack_workflow(path, str(pack))
    stale = pack / 'inputs' / 'gone' / 'old.json'
    stale.parent.mkdir()
    stale.write_text('{}')
    again = pack_workflow(path, str(pack), overwrite=True)
    assert again['files'] == first['files'] and not stale.exists()
    # a directory holding anything a pack does not write is never replaced
    (pack / 'notes.txt').write_text('mine')
    with pytest.raises(ValueError, match=r"holds \['notes.txt'\], which a pack does not write"):
        pack_workflow(path, str(pack), overwrite=True)
    assert (pack / 'notes.txt').read_text() == 'mine'
    assert (pack / 'calibration_workflow.json').is_file()


@pytest.mark.integration
def test_run_prints_the_workflow_lines_and_keeps_all_output_in_logs(library, tmp_path, capsys):
    wf = _chain(library, tmp_path / 'wf')
    wf.run()
    captured = capsys.readouterr()
    printed = [line for line in captured.out.splitlines() if line.strip()]
    assert printed and all(line.startswith('[workflow]') for line in printed), printed
    assert captured.err == ''
    assert '[workflow] step 1/2: fit_a (calibration, lin_a/v1/fit_a)' in printed
    logs = {step: (wf.step_dir(step) / 'log.txt').read_text() for step in wf.steps}
    assert all(logs.values()) and logs['fit_a'] != logs['rest']
    assert not any(line.startswith('[workflow]') for text in logs.values()
                   for line in text.splitlines())
    workflow_log = (wf.output_dir / 'workflow_log.txt').read_text()
    assert all(line in workflow_log for line in printed)
    assert not list(wf.output_dir.glob('step_log.*'))
    # the streams are the caller's again
    print('after the run')
    assert capsys.readouterr().out == 'after the run\n'

    # verbose prints the steps' output too, and the logs are written as before
    wf.run(force=True, verbose=True)
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any(not line.startswith('[workflow]') for line in printed)
    assert all((wf.step_dir(step) / 'log.txt').read_text() for step in wf.steps)
    # progress=False prints nothing, and still keeps the logs
    wf.run(force=True, progress=False)
    assert capsys.readouterr().out == ''
    assert all((wf.step_dir(step) / 'log.txt').read_text() for step in wf.steps)

    wf.show_status()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith('[workflow] chain_built: 2 of 2 steps are done and current')
    assert [line.split()[:3] for line in lines[2:]] == [['fit_a', 'calibration', 'done'],
                                                        ['rest', 'calibration', 'done']]

    # a failing step's output is kept too, and the message says where
    wf.add_calibration('rest', obs_data=_obs('z_c', 'mod__C/nope', Z_C, STD_C),
                       params_for_id=_entries('mod__C', 'q', 0.1, 10.0), fixed_from='fit_a')
    with pytest.raises(Exception):
        wf.run()
    out = capsys.readouterr().out
    assert '[workflow] step rest FAILED' in out
    assert f'[workflow] the output of step rest is in {wf.step_dir("rest") / "log.txt"}' in out
    assert (wf.step_dir('rest') / 'log.txt').is_file()

    # plot_step saves the formats asked for, png by default
    images = wf.plot_step('fit_a')
    assert images and {p.suffix for p in images} == {'.png'} and all(p.is_file() for p in images)
    both = wf.plot_step('fit_a', formats=('pdf', 'png'))
    assert {p.suffix for p in both} == {'.pdf', '.png'} and set(images) <= set(both)
    assert {p.suffix for p in wf.plot_step('fit_a', formats='eps')} == {'.eps'}
    with pytest.raises(WorkflowError, match=r"not \['svg'\]"):
        wf.plot_step('fit_a', formats=('svg',))
