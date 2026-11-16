"""Building a calibration workflow in Python (libcuflynx.calibration_workflow.WorkflowBuilder).

The builder writes the same calibration_workflow.json a user writes by hand, with dict inputs
as files under data/ and functions defined in a notebook as a functions file beside it, so the
runner, pack_workflow and CUFLynx see an ordinary workflow. The models are the algebraic ones
of libcuflynx.external_testing.workflow_library (y = 2 p^2 for lin_a; z = 2 p_A^2 + q^2 for
chain), whose optima are known in closed form.
"""
import inspect
import re
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
    wf.add_reduction('reduce', sensitivity_from='screen', method='sobol', n_params=1,
                     threshold=0.05)
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
def test_no_step_is_named_as_the_runs_folder_of_models(library, tmp_path):
    schema = load_schema(CALIBRATION_WORKFLOW_SCHEMA)
    assert MODELS_DIR in schema['$defs']['step']['properties']['id']['not']['enum']
    wf = _chain(library, tmp_path / 'wf')
    with pytest.raises(WorkflowError, match='step "models"'):
        wf.add_calibration(MODELS_DIR, obs_data='x.json', params_for_id='y.json')


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


@pytest.mark.unit
def test_a_functions_file_changed_outside_the_builder_is_not_overwritten(tmp_path):
    target = library_module('lin_a', 'v1', 'fit_a')
    wf = WorkflowBuilder('funcs', target=target, directory=tmp_path / 'wf')
    wf.add_operation(double_mean)
    wf.add_calibration('fit_a', obs_data='x.json', params_for_id='y.json')
    wf.write()
    funcs = wf.settings['operation_funcs_external_path']
    with open(funcs, 'a') as f:
        f.write('\n\ndef by_hand(x):\n    return x\n')
    with pytest.raises(WorkflowError, match=re.escape(funcs) + ' has changed since this builder'):
        wf.write()
    with pytest.raises(WorkflowError, match='has changed since this builder'):
        wf.add_operation(double_mean)
    with open(funcs) as f:
        assert 'def by_hand' in f.read()                 # not overwritten

    # load() starts from the file as it is, and may then add to it
    again = WorkflowBuilder.load(wf.path)
    again.add_operation(double_mean)
    again.write()
    assert 'def by_hand' in again.funcs_source()

    # a builder that did not write or read the file does not replace it
    fresh = WorkflowBuilder('funcs', target=target, directory=tmp_path / 'wf')
    with pytest.raises(WorkflowError, match='did neither write nor read it'):
        fresh.add_operation(double_mean)


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
    assert model['model_path'].startswith(os.path.join(wf.output_dir, MODELS_DIR, 'rest') + os.sep)


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
