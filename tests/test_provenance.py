"""The provenance of a calibration workflow run: what ran with what, and what set every value."""
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ElementTree

import pytest

from libcuflynx.calibration_workflow import (WorkflowError, run_calibration_workflow,
                                             workflow_status)
from libcuflynx.calibration_workflow.runner import module_library_digest, settings_digest
from libcuflynx.external_testing.workflow_library import build_workflow_library
from libcuflynx.reporting.provenance import (DERIVED_NODE, EARLIER_NODE, FIXED_NODE, GRAPH_FILE,
                                             HAND_NODE,
                                             MEASURED_NODE, SCHEMA_VERSION, SOURCE_NODE, SVG_FILE,
                                             TARGET_NODE, ProvenanceVersionError,
                                             provenance_graph, provenance_svg,
                                             read_provenance_graph, write_provenance)
from libcuflynx.schemas import PROVENANCE_GRAPH_SCHEMA
from libcuflynx.schemas.validation import validate
from libcuflynx.scripts import provenance_script
from libcuflynx.utilities.paths import git_state, repo_root


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def _parameters_path(library):
    return os.path.join(library['chain/rest'], 'rest_parameters.json')


def _mark_values(library):
    """In the chain's parameters, mark k as set by hand."""
    path = _parameters_path(library)
    doc = _read_json(path)
    leaves = _leaves(doc)
    leaves['k'].update(origin='set_by_hand', reason='a gain of one by construction')
    _write_json(path, doc)


def _leaves(doc):
    found = {}

    def walk(node):
        for key, child in node.items():
            if isinstance(child, dict) and 'value' in child:
                found[key] = child
            elif isinstance(child, dict):
                walk(child)
    walk({k: v for k, v in doc.items() if k != 'schema_version'})
    return found


def _nodes(graph):
    return {node['id']: node for node in graph['nodes']}


def _edges(graph):
    return {(e['from'], e['to']): e for e in graph['edges']}


@pytest.mark.integration
def test_a_run_writes_its_provenance_and_says_what_set_each_value(library, tmp_path):
    _mark_values(library)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(library['chain_rest'], output_dir=out)

    # what it ran with
    assert result['libcellml_version']
    assert result['libcuflynx_git'] == git_state(repo_root())
    assert set(result['project_git']) == {'commit', 'dirty'}

    graph = _read_json(os.path.join(out, GRAPH_FILE))
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    assert graph['schema_version'] == SCHEMA_VERSION == 3
    assert graph['producer']['name'] == 'libcuflynx'
    assert graph['workflow']['output_dir'] == out
    assert graph['workflow']['order'] == ['fit_a', 'rest']
    assert graph['code']['libcellml_version'] == result['libcellml_version']

    nodes = _nodes(graph)
    assert {'fit_a', 'rest', SOURCE_NODE, HAND_NODE, TARGET_NODE} <= set(nodes)
    assert MEASURED_NODE not in nodes
    assert nodes['fit_a']['mark'] == nodes['rest']['mark'] == 'calibrated'
    assert [p['name'] for p in nodes['fit_a']['parameters'] if p['from_step'] == 'fit_a']
    assert any(line.startswith('data `') for line in nodes['fit_a']['lines'])
    assert '1 parameter, 1 observable' in nodes['fit_a']['lines']
    assert nodes[SOURCE_NODE]['lines'][0] == '`chain`, parameterisation `rest`'

    # a value set by hand is never left looking calibrated, and gives its value and reason
    hand = nodes[HAND_NODE]
    assert hand['mark'] == 'set_by_hand'
    assert [v['reason'] for v in hand['parameters']] == ['a gain of one by construction']
    assert 'a gain of one by construction' in hand['lines'][0]
    assert (HAND_NODE, TARGET_NODE) in _edges(graph)
    assert all(hand['parameters'][0]['name'] != p['name'] for p in nodes['rest']['parameters'])

    # a value nothing describes is unclassified, never inherited
    assert graph['counts']['inherited'] == 0
    assert graph['counts']['calibrated'] == len(result['merged_parameters'])
    assert graph['counts']['set_by_hand'] == 1

    # both calibrations reach the target
    assert {('fit_a', TARGET_NODE), ('rest', TARGET_NODE)} <= set(_edges(graph))
    assert _edges(graph)[('fit_a', 'rest')]['style'] == 'fixed'
    assert (SOURCE_NODE, 'fit_a') in _edges(graph)
    assert not graph['workflow']['stale']
    assert {e['status'] for n in graph['nodes'] for e in n['inputs']} == {'current'}

    svg = open(os.path.join(out, SVG_FILE)).read()
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith('svg') and root.get('role') == 'img'
    assert '<script' not in svg and '<style' not in svg
    for text in ('SET BY HAND', 'CALIBRATED', 'unclassified', out, 'libCellML'):
        assert text in svg


FIXED = [{'derivation': 'k_gain', 'name': 'R', 'value': 8.314, 'units': 'J_per_K_per_mol',
          'origin': 'measured', 'data_reference': '[CODATA][2018][gas_constant]'},
         {'derivation': 'k_gain', 'name': 'T', 'value': 310.0, 'units': 'kelvin',
          'origin': 'set_by_hand', 'reason': 'body temperature'},
         {'derivation': 'k_gain', 'name': 'F', 'value': 96485.0, 'units': 'C_per_mol'}]


@pytest.mark.integration
def test_fixed_constants_of_a_derivation_are_drawn_with_value_and_origin(library, tmp_path):
    """The fixed constants of the derivation that calculates a value are a box of their own,
    each with its value, units and origin (unclassified when its derivation declares none), and
    an arrow to the derived values; a run with no fixed constants has no such box."""
    path = _parameters_path(library)
    doc = _read_json(path)
    _leaves(doc)['k'].update(origin='derived', derivation='k_gain',
                             data_reference='derived; derivation k_gain',
                             fixed_constants=json.dumps(FIXED))
    _write_json(path, doc)
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_rest'], output_dir=out)

    graph = _read_json(os.path.join(out, GRAPH_FILE))
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    fixed = _nodes(graph)[FIXED_NODE]
    assert fixed['mark'] == 'mixed'
    assert [(v['name'], v['origin']) for v in fixed['parameters']] == [
        ('k_gain/R', 'measured'), ('k_gain/T', 'set_by_hand'), ('k_gain/F', 'unclassified')]
    assert _edges(graph)[(FIXED_NODE, DERIVED_NODE)]['label'] == 'constants of'
    assert graph['counts']['derived'] == 1       # constants are not values of the target
    # drawn left of the model source, so its arrow to the derived values crosses no other arrow
    order = [n['id'] for n in graph['nodes']]
    assert order.index(FIXED_NODE) < order.index(SOURCE_NODE) and order.index(
        SOURCE_NODE) < order.index(DERIVED_NODE)
    # the drawing wraps long lines, so compare the text it shows, joined
    svg = ' '.join(re.findall(r'>([^<]*)</text>', provenance_svg(graph)))
    for text in ('k_gain/R', '8.314 J_per_K_per_mol, measured', '[CODATA][2018][gas_constant]',
                 'k_gain/T', '310.0 kelvin, set by hand: body temperature',
                 'k_gain/F', '96485.0 C_per_mol, unclassified', 'Fixed constants of derivations'):
        assert text in svg

    _leaves(doc)['k'].pop('fixed_constants')
    _write_json(path, doc)
    out = str(tmp_path / 'run_without')
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    nodes = _nodes(_read_json(os.path.join(out, GRAPH_FILE)))
    assert DERIVED_NODE in nodes and FIXED_NODE not in nodes


@pytest.mark.integration
def test_derived_values_and_values_calibrated_before_the_run_have_their_own_marks(library,
                                                                                tmp_path):
    """A value whose parameters entry a derivation calculates (origin derived) is drawn in its
    own box, marked derived, not as measured or unclassified; a value calibrated before the run
    (origin calibrated in its parameters file) is calibrated, in a box of its own."""
    path = _parameters_path(library)
    doc = _read_json(path)
    _leaves(doc)['k'].update(origin='derived', derivation='k_gain',
                             data_reference='derived; derivation k_gain')
    _write_json(path, doc)
    out = str(tmp_path / 'run')
    result = run_calibration_workflow(library['chain_rest'], output_dir=out)

    graph = _read_json(os.path.join(out, GRAPH_FILE))
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    nodes = _nodes(graph)
    derived = nodes[DERIVED_NODE]
    assert derived['mark'] == 'derived' and derived['title'] == 'Derived values'
    assert [(v['name'], v['origin']) for v in derived['parameters']] == [('k_mod__C', 'derived')]
    assert 'derived; derivation k_gain' in derived['lines'][0]
    assert _edges(graph)[(DERIVED_NODE, TARGET_NODE)]['label'] == 'derived'
    assert graph['counts']['derived'] == 1 and graph['counts']['unclassified'] == 0
    assert {HAND_NODE, MEASURED_NODE, EARLIER_NODE}.isdisjoint(nodes)
    svg = provenance_svg(graph)
    assert 'DERIVED' in svg and '1 derived' in svg

    # the target's values as a later run sees them, k calibrated by an earlier workflow
    target = result['target_model']['parameters_path']
    doc = _read_json(target)
    _leaves(doc)['k'].update(origin='calibrated', data_reference='calibrated by workflow old')
    _write_json(target, doc)
    graph = provenance_graph(out)
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    nodes = _nodes(graph)
    assert DERIVED_NODE not in nodes
    earlier = nodes[EARLIER_NODE]
    assert earlier['mark'] == 'calibrated'
    assert [(v['name'], v['from_step']) for v in earlier['parameters']] == [('k_mod__C', None)]
    assert graph['counts']['calibrated'] == len(result['merged_parameters']) + 1
    assert all(v['from_step'] for v in nodes[TARGET_NODE]['parameters'])


@pytest.mark.integration
def test_the_graph_says_which_input_went_stale(library, tmp_path):
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 4.5
    _write_json(obs_path, obs)
    graph = provenance_graph(out)
    nodes = _nodes(graph)
    status = {e['key']: e['status'] for e in nodes['fit_a']['inputs']}
    assert status['obs_data'] == 'stale' and status['params_for_id'] == 'current'
    # the step that took fit_a's value is stale too, though its own inputs are not
    assert nodes['fit_a']['stale'] and nodes['rest']['stale'] and graph['workflow']['stale']
    assert all(e['status'] == 'current' for e in nodes['rest']['inputs'])
    assert 'STALE' in provenance_svg(graph)


@pytest.mark.integration
def test_settings_are_stale_only_where_a_step_s_own_settings_changed(library, tmp_path):
    """The graph judges settings per step, as workflow_status does, not by the file's bytes."""
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    raw = _read_json(library['chain_rest'])

    def stale_steps():
        graph = provenance_graph(out)
        nodes = _nodes(graph)
        drawn = {s: nodes[s]['stale'] for s in ('fit_a', 'rest')}
        settings = {s: {e['key']: e['status'] for e in nodes[s]['inputs']}['settings']
                    for s in ('fit_a', 'rest')}
        status = workflow_status(library['chain_rest'], out)['steps']
        assert drawn == {s: status[s]['stale'] for s in drawn}
        assert graph['workflow']['stale'] == any(drawn.values())
        return {s for s, stale in drawn.items() if stale}, settings, graph

    # the same workflow, re-saved with other formatting: nothing is stale
    with open(library['chain_rest'], 'w') as f:
        json.dump(raw, f, indent=4)
    stale, settings, graph = stale_steps()
    assert stale == set() and set(settings.values()) == {'current'}
    assert 'STALE' not in provenance_svg(graph)

    # rest's own settings change: rest is stale, fit_a before it is not
    steps = {s['id']: s for s in raw['steps']}
    steps['rest']['settings'] = {'optimiser_options': {'num_calls_to_function': 401}}
    _write_json(library['chain_rest'], raw)
    stale, settings, _ = stale_steps()
    assert stale == {'rest'} and settings == {'fit_a': 'current', 'rest': 'stale'}

    # fit_a's change reaches rest, which takes its value; only fit_a's settings changed
    del steps['rest']['settings']
    steps['fit_a']['settings'] = {'optimiser_options': {'num_calls_to_function': 401}}
    _write_json(library['chain_rest'], raw)
    stale, settings, _ = stale_steps()
    assert stale == {'fit_a', 'rest'} and settings == {'fit_a': 'stale', 'rest': 'current'}


@pytest.mark.integration
def test_a_changed_module_library_is_reported_and_warned_about(library, tmp_path):
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    config = next(os.path.join(root, name) for root, _, files in os.walk(library['modules'])
                  for name in files if name.endswith('_modules_config.json'))
    with open(config, 'a') as f:
        f.write('\n')
    fit_a = _nodes(provenance_graph(out))['fit_a']
    assert {e['key']: e['status'] for e in fit_a['inputs']}['module_library'] == 'stale'
    assert 'module library changed since it ran' in fit_a['lines'] and not fit_a['stale']
    step = workflow_status(library['chain_rest'], out)['steps']['fit_a']
    assert step['library_changed'] and not step['stale']
    with pytest.warns(UserWarning, match='module library has changed since "fit_a" ran'):
        run_calibration_workflow(library['chain_rest'], output_dir=out, only='rest')


@pytest.mark.integration
def test_changed_settings_stop_a_later_step_reusing_a_result(library, tmp_path):
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_rest'], output_dir=out)
    raw = _read_json(library['chain_rest'])
    raw['settings']['param_id_method'] = 'genetic_algorithm'
    _write_json(library['chain_rest'], raw)
    with pytest.raises(WorkflowError, match='its settings has changed since "fit_a" ran'):
        run_calibration_workflow(library['chain_rest'], output_dir=out, from_step='rest')


@pytest.mark.integration
def test_an_upstream_change_stops_a_step_two_steps_on(library, tmp_path):
    """fit_a changed, screen and fit took it in: the identifiability step after them refuses."""
    out = str(tmp_path / 'run')
    run_calibration_workflow(library['chain_analysed'], output_dir=out)
    obs_path = os.path.join(library['lin_a/fit_a'], 'fit_a_obs_data.json')
    obs = _read_json(obs_path)
    obs['data_items'][0]['value'] = 4.5
    _write_json(obs_path, obs)
    with pytest.raises(WorkflowError, match='has changed since "fit_a" ran'):
        run_calibration_workflow(library['chain_analysed'], output_dir=out, only='ia')


@pytest.mark.integration
def test_the_command_writes_both_files_and_a_partial_run_still_has_a_graph(library, tmp_path,
                                                                           capsys):
    out = str(tmp_path / 'run')
    with pytest.raises(WorkflowError):
        run_calibration_workflow(library['chain_rest'], output_dir=out, only='rest')
    run_calibration_workflow(library['chain_rest'], output_dir=out, only='fit_a')
    os.remove(os.path.join(out, GRAPH_FILE))
    assert provenance_script.main([out]) == 0
    assert os.path.join(out, GRAPH_FILE) in capsys.readouterr().out
    nodes = _nodes(_read_json(os.path.join(out, GRAPH_FILE)))
    assert nodes['rest']['status'] == 'not_run' and nodes['rest']['lines'] == ['not run']
    assert nodes['fit_a']['status'] == 'done'


def _box(node_id, kind='step'):
    return {'id': node_id, 'kind': kind, 'title': node_id, 'lines': [], 'mark': None,
            'status': 'done', 'stale': False, 'counts': {}, 'parameters': [], 'inputs': []}


def _another_producers_graph():
    git = {'commit': 'abcdef1234567890', 'dirty': False}
    graph = {'schema_version': 3,
             'workflow': {'name': 'w', 'output_dir': '/runs/w', 'file': None, 'file_status': 'current',
                          'order': ['a'], 'complete': True, 'finished': None, 'stale': False},
             'producer': {'name': 'opinf', 'version': '0.1', 'git': git},
             'code': {'libcuflynx_version': None, 'libcuflynx_git': git, 'project_git': git,
                      'libcellml_version': None},
             'counts': {'calibrated': 0, 'measured': 0, 'inherited': 0, 'set_by_hand': 0,
                        'unclassified': 0},
             'nodes': [_box('a')], 'edges': []}
    return graph


@pytest.mark.unit
def test_another_producer_validates_and_is_named_in_the_footer():
    graph = _another_producers_graph()
    validate(PROVENANCE_GRAPH_SCHEMA, graph, source='graph', what='provenance graph')
    assert 'opinf 0.1: abcdef1234' in provenance_svg(graph)


@pytest.mark.unit
@pytest.mark.parametrize('version', [1, 2, 4, None])
def test_a_graph_of_a_version_it_does_not_know_is_refused(tmp_path, version):
    graph = _another_producers_graph()
    path = str(tmp_path / GRAPH_FILE)
    _write_json(path, graph)
    assert read_provenance_graph(path) == graph
    graph['schema_version'] = version
    _write_json(path, graph)
    message = rf'schema_version {version!r}; .* reads provenance graph schema_version 3 only'
    with pytest.raises(ProvenanceVersionError, match=message):
        read_provenance_graph(path)
    with pytest.raises(ProvenanceVersionError, match=message):
        provenance_svg(graph)


@pytest.mark.unit
def test_an_arrow_past_a_row_crosses_no_box():
    long_name = '/a/very/long/output/directory/' * 6
    graph = {'schema_version': 3, 'workflow': {'name': 'w', 'output_dir': long_name},
             'code': {}, 'counts': {},
             'nodes': [_box('a'), _box('b'), _box('c')],
             'edges': [{'from': 'a', 'to': 'b', 'label': '', 'style': 'order'},
                       {'from': 'b', 'to': 'c', 'label': '', 'style': 'order'},
                       {'from': 'a', 'to': 'c', 'label': 'skips', 'style': 'merge'}]}
    svg = provenance_svg(graph)
    boxes = re.findall(r'<rect x="(\d+)" y="\d+" width="(\d+)"', svg)
    right_edge = max(int(x) + int(width) for x, width in boxes)
    lane = re.search(r'V\d+ H(\d+) V\d+ H\d+ V\d+', svg)
    assert lane and int(lane.group(1)) > right_edge
    # the footer wraps inside the drawing, however long the directory
    view_width = int(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
    footer = re.findall(r'<text x="16"[^>]*>([^<]*)</text>', svg)
    assert len(footer) > 3 and max(len(line) for line in footer) * 6.5 < view_width


@pytest.mark.unit
def test_a_folder_without_a_repository_has_no_commit(tmp_path):
    assert git_state(str(tmp_path)) == {'commit': None, 'dirty': None}
    assert git_state(None) == {'commit': None, 'dirty': None}


@pytest.mark.unit
def test_a_repository_records_its_commit_and_tracked_changes(tmp_path):
    def git(*args):
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', '-C',
                        str(tmp_path), *args], check=True, capture_output=True)
    git('init')
    (tmp_path / 'tracked').write_text('one')
    git('add', 'tracked')
    git('commit', '-m', 'first')
    state = git_state(str(tmp_path))
    assert re.fullmatch('[0-9a-f]{40}', state['commit']) and state['dirty'] is False
    (tmp_path / 'untracked').write_text('output')
    assert git_state(str(tmp_path))['dirty'] is False
    (tmp_path / 'tracked').write_text('two')
    assert git_state(str(tmp_path))['dirty'] is True


@pytest.mark.unit
def test_a_settings_digest_ignores_key_order():
    assert settings_digest({'a': 1, 'b': 2}) == settings_digest({'b': 2, 'a': 1})
    assert settings_digest({'a': 1}) != settings_digest({'a': 2})


@pytest.mark.unit
def test_a_library_digest_follows_module_files_only(tmp_path):
    (tmp_path / 'x_modules.cellml').write_text('<model/>')
    (tmp_path / 'notes.png').write_text('a plot')
    before = module_library_digest([str(tmp_path)])
    (tmp_path / 'notes.png').write_text('another plot')
    assert module_library_digest([str(tmp_path)]) == before
    (tmp_path / 'x_modules.cellml').write_text('<model name="m"/>')
    assert module_library_digest([str(tmp_path)]) != before


@pytest.mark.unit
def test_write_provenance_refuses_a_folder_that_is_no_run(tmp_path):
    with pytest.raises(WorkflowError, match='is not a calibration workflow run'):
        write_provenance(str(tmp_path))


def test_helper_items_are_not_counted_as_observables(tmp_path):
    """A data_item with weight 0 only feeds other items, so a step's observable count leaves it
    out. A large std does not make an item a helper."""
    from libcuflynx.reporting.provenance import _data_item_counts
    path = tmp_path / 'obs_data.json'
    _write_json(path, {'data_items': [
        {'variable': 'a', 'value': 1.0, 'std': 0.1},
        {'variable': 'b', 'value': 1.0, 'std': 0.1, 'weight': 2.0},
        {'variable': 'a_half', 'value': 1.0, 'std': 1e15},
        {'variable': 'b_sat', 'value': 1.0, 'std': 0.1, 'weight': 0},
        {'variable': 'series', 'value': [1.0, 2.0], 'std': [0.1, 1e15]},
        {'variable': 'series_helper', 'value': [1.0, 2.0], 'std': [1e15, 1e15]},
        {'variable': 'zero_weights', 'value': [1.0, 2.0], 'std': 0.1, 'weight': [0, 0]},
    ]})
    result = {'inputs': {'obs_data': {'path': str(path)}}}
    assert _data_item_counts(result) == (5, 2)
    assert _data_item_counts({'inputs': {}}) == (0, 0)


def test_only_weight_zero_makes_a_helper_item(tmp_path):
    """An item may have a value and std of very large order and still be scored: only weight 0
    marks a helper."""
    from libcuflynx.reporting.provenance import _data_item_counts
    path = tmp_path / 'obs_data.json'
    _write_json(path, {'data_items': [
        {'variable': 'huge', 'value': 1e20, 'std': 1e20, 'weight': 1},
        {'variable': 'unscored', 'value': 1.0, 'std': 0.1, 'weight': 0},
    ]})
    assert _data_item_counts({'inputs': {'obs_data': {'path': str(path)}}}) == (1, 1)
