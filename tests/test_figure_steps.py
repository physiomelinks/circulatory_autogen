"""The ``figure`` kind of a calibration workflow: a Python function drawing from step results.

The fixture library is libcuflynx.external_testing.workflow_library: in ``chain``, C reads A's
output, z = 2 p_A^2 + k q^2. The workflow here fits A, fits C, sweeps q of C at A's fit, and
draws one figure from the sweep, A's values, C's obs_data and C's params_for_id.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (WorkflowError, plan_workflow,
                                             run_calibration_workflow, workflow_status)
from libcuflynx.calibration_workflow import figures
from libcuflynx.calibration_workflow.pack import pack_workflow
from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.runner import (WorkflowStepError,
                                                    write_calibrated_parameters_to)
from libcuflynx.calibration_workflow.spec import load_workflow, parse_workflow
from libcuflynx.external_testing.workflow_library import Y_A, build_workflow_library
from libcuflynx.reporting.methods import workflow_methods_latex
from libcuflynx.reporting.provenance import provenance_graph, provenance_svg

SCRIPT = '''"""z of the chain against q, with A's fitted p in the title."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import style


def draw(curves, fit, data, chosen, out_dir):
    q = curves['points'][:, 0]
    z = curves['trace_0'][:, 0, -1]
    figure, axes = plt.subplots(figsize=(3, 2), dpi=60)
    axes.plot(q, z, color=style.COLOUR, marker='o')
    axes.axhline(data['data_items'][0]['value'], color='k')
    axes.set_title('p = %.3f' % fit['mod/p'])
    plot = os.path.join(out_dir, 'z_against_q.png')
    figure.savefig(plot)
    plt.close(figure)
    table = os.path.join(out_dir, 'z_against_q.csv')
    with open(table, 'w') as f:
        f.write('q,z\\n' + ''.join('%r,%.10g\\n' % (a, b) for a, b in zip(q, z)))
        f.write('fitted,' + ';'.join(str(entry['targets']) for entry in chosen['params']) + '\\n')
    return [plot, table]


def main(argv=None):
    from libcuflynx.calibration_workflow.figures import figure_main
    return figure_main(draw, argv)


if __name__ == '__main__':
    main()
'''
STYLE = "COLOUR = 'tab:red'\n"
INPUTS = {'curves': {'sweep': 'scan'}, 'fit': {'values': 'fit_a'}, 'data': {'obs_data': 'rest'},
          'chosen': {'params_for_id': 'rest'}}


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, raw):
    with open(path, 'w') as f:
        json.dump(raw, f, indent=1)
    return path


def _figure_step(**keys):
    return dict({'id': 'fig', 'kind': 'figure', 'function': 'figures/z_against_q.py:draw',
                 'inputs': dict(INPUTS), 'files': ['figures/style.py']}, **keys)


def _with_figure(library, script=SCRIPT):
    '''chain_rest with a sweep of q at A's fit and the figure step; the workflow file.'''
    raw = _read_json(library['chain_rest'])
    rest = raw['steps'][1]
    raw['steps'] += [
        {'id': 'scan', 'kind': 'sweep', 'model': rest['model'], 'fixed_from': ['fit_a'],
         'settings': {'sim_time': 1.0},
         'sweep': {'outputs': ['mod__C/z'], 'parameters': {'mod__C/q': {'values': [1.0, 2.0]}}}},
        _figure_step()]
    directory = os.path.join(os.path.dirname(library['chain_rest']), 'figures')
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'z_against_q.py'), 'w') as f:
        f.write(script)
    with open(os.path.join(directory, 'style.py'), 'w') as f:
        f.write(STYLE)
    return _write_json(library['chain_rest'], raw)


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    '''One run of the workflow with the figure: ``(library, workflow path, output dir)``.'''
    root = tmp_path_factory.mktemp('figure_run')
    library = build_workflow_library(str(root / 'library'))
    path, out = _with_figure(library), str(root / 'out')
    run_calibration_workflow(path, output_dir=out)
    return library, path, out


def _edit(library, edit):
    path = _with_figure(library)
    raw = _read_json(path)
    edit(raw, next(s for s in raw['steps'] if s['id'] == 'fig'))
    return raw, path


# --------------------------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_figure_step_depends_on_the_steps_its_inputs_name(library):
    workflow = load_workflow(_with_figure(library))
    step = workflow.step('fig')
    assert (step.kind, step.model, step.depends_on) == ('figure', None, ['scan', 'fit_a', 'rest'])
    assert step.figure['function'] == 'draw'
    assert step.figure['script'].endswith(os.path.join('figures', 'z_against_q.py'))
    assert step.figure['inputs']['curves'] == {'kind': 'sweep', 'step': 'scan'}
    assert [s.id for s in workflow.topological_order()][-1] == 'fig'
    plan = {s['id']: s for s in plan_workflow(workflow)['steps']}
    assert plan['fig']['figure']['inputs']['fit'] == {'kind': 'values', 'step': 'fit_a'}
    assert plan['fig']['depends_on'] == ['scan', 'fit_a', 'rest']


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda raw, fig: fig['inputs'].update(curves={'sweep': 'fit_a'}),
     'reads "sweep" from "fit_a", a calibration step; it is read from a sweep or simulate step'),
    (lambda raw, fig: fig['inputs'].update(curves={'traces': 'scan'}), 'not one of'),
    (lambda raw, fig: fig['inputs'].update(curves={'sweep': 'nowhere'}),
     'takes from unknown step "nowhere"'),
    (lambda raw, fig: fig['inputs'].update(curves={'sweep': 'scan', 'values': 'fit_a'}),
     'has too many'),
    (lambda raw, fig: fig['inputs'].update(out_dir={'sweep': 'scan'}),
     '"out_dir" is where the function writes'),
    (lambda raw, fig: fig['inputs'].update({'my curves': {'sweep': 'scan'}}), 'is not in the form described'),
    (lambda raw, fig: fig['inputs'].update(post={'distribution': 'fit_a'}),
     'does not store one'),
    (lambda raw, fig: fig['inputs'].update(data={'obs_data': 'scan'}), 'which has none'),
    (lambda raw, fig: fig.update(model=raw['steps'][1]['model']), r"has no \['model'\]"),
    (lambda raw, fig: fig.update(settings={'dt': 0.1}), r"has no \['settings'\]"),
    (lambda raw, fig: fig.update(fixed_from=['fit_a']), 'a figure step has no "fixed_from"'),
    (lambda raw, fig: fig.pop('inputs'), "'inputs' is a required property"),
    (lambda raw, fig: fig.update(function='figures/z_against_q.py'), 'is not in the form described'),
    (lambda raw, fig: raw['steps'][1].update(files=['a.py']), 'are for a figure step'),
    (lambda raw, fig: (fig.update(inputs={'curves': {'sweep': 'scan'}}),
                       raw['steps'][1].update(fixed_from=['fit_a', 'fig'])),
     '"fixed_from" names "fig", a figure step'),
])
def test_invalid_figure_steps_say_why(library, edit, message):
    raw, path = _edit(library, edit)
    with pytest.raises(WorkflowError, match=message):
        parse_workflow(raw, path=path)


@pytest.mark.unit
@pytest.mark.parametrize('script, message', [
    (None, 'does not exist'),
    (SCRIPT.replace('def draw(', 'def plot('), 'defines no function "draw"'),
    (SCRIPT.replace('chosen, out_dir)', 'out_dir)'),
     r"parameters of \"draw\" are \['curves', 'data', 'fit', 'out_dir'\]"),
    (SCRIPT.replace('chosen, out_dir)', 'chosen, out_dir, **more)'), r'\*args or \*\*kwargs'),
    (SCRIPT.replace('def main(', 'def run(').replace('    main()', '    run()'), 'has no "main"'),
    (SCRIPT + 'def broken(:\n', 'is not valid Python'),
])
def test_the_script_is_checked_without_running_it(library, script, message):
    path = _with_figure(library, script or SCRIPT)
    if script is None:
        os.remove(os.path.join(os.path.dirname(path), 'figures', 'style.py'))
    with pytest.raises(WorkflowError, match=message):
        resolve_workflow(load_workflow(path))


@pytest.mark.unit
def test_a_figure_step_leaves_the_workflow_digest_as_it_was(library):
    '''So adding or changing a figure after a run keeps the run's values current.'''
    from libcuflynx.calibration_workflow.runner import workflow_digest
    path = _with_figure(library)
    with_figure = workflow_digest(path)
    raw = _read_json(path)
    raw['steps'] = [s for s in raw['steps'] if s['id'] != 'fig']
    assert workflow_digest(_write_json(path, raw)) == with_figure
    raw['steps'][-1]['settings']['sim_time'] = 2.0
    assert workflow_digest(_write_json(path, raw)) != with_figure


@pytest.mark.unit
def test_every_input_kind_is_in_the_schema():
    from libcuflynx.schemas import CALIBRATION_WORKFLOW_SCHEMA, load_schema
    names = load_schema(CALIBRATION_WORKFLOW_SCHEMA)['$defs']['figure_input']['propertyNames']
    assert names['enum'] == list(figures.INPUT_KINDS)


def _run_alone(script, run_dir, out_dir, *more):
    done = subprocess.run([sys.executable, script, run_dir, '--out-dir', out_dir, *more],
                          capture_output=True, text=True, timeout=120, env=dict(os.environ))
    assert done.returncode == 0, done.stdout + done.stderr
    return out_dir


# --------------------------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_figure_is_drawn_from_the_loaded_results(finished):
    _, path, out = finished
    result = _read_json(os.path.join(out, 'fig', 'step_result.json'))
    assert (result['kind'], result['status'], result['model']) == ('figure', 'done', None)
    directory = os.path.join(out, 'fig', 'figure')
    assert result['figure_dir'] == directory
    assert result['figures'] == [os.path.join(directory, 'z_against_q.png'),
                                 os.path.join(directory, 'z_against_q.csv')]
    assert result['function'] == 'z_against_q.py:draw'
    assert sorted(result['inputs']) == ['file:style.py', 'script', 'settings']
    with open(result['figures'][1]) as f:
        rows = f.read().splitlines()
    assert rows[0] == 'q,z'
    np.testing.assert_allclose([float(row.split(',')[1]) for row in rows[1:3]],
                               [Y_A + 1.0, Y_A + 4.0], rtol=1e-6)
    assert rows[3] == "fitted,['mod/C/q']"
    status = workflow_status(path, out)
    assert status['complete'] and not status['steps']['fig']['stale']


@pytest.mark.integration
def test_the_provenance_and_the_methods_show_the_figure(finished):
    _, path, out = finished
    graph = provenance_graph(out)
    node = next(n for n in graph['nodes'] if n['id'] == 'fig')
    assert (node['kind'], node['status'], node['stale'], node['mark']) == (
        'analysis', 'done', False, None)
    assert node['lines'][:2] == ['`z_against_q.py:draw`', '`curves`: sweep of `scan`']
    assert {(e['from'], e['label'], e['style']) for e in graph['edges'] if e['to'] == 'fig'} == {
        ('scan', 'sweep', 'reads'), ('fit_a', 'values', 'reads'),
        ('rest', 'obs_data + params_for_id', 'reads')}
    assert [(e['to'], e['style']) for e in graph['edges'] if e['from'] == 'fig'] == [
        ('fig/files', 'writes')]
    files = next(n for n in graph['nodes'] if n['id'] == 'fig/files')
    assert (files['title'], files['status']) == ('2 files written', 'done')
    assert {e['key'] for e in node['inputs']} == {'settings', 'script', 'file:style.py'}
    assert all(e['status'] == 'current' for e in node['inputs'])
    assert 'z_against_q.py:draw' in provenance_svg(graph)
    latex = workflow_methods_latex(path, run_dir=out)
    assert 'This step draws figures with' in latex and '1 figure step' in latex


@pytest.mark.integration
def test_the_script_run_on_its_own_writes_the_same_files(finished, tmp_path):
    library, path, out = finished
    script = os.path.join(os.path.dirname(path), 'figures', 'z_against_q.py')
    alone = str(tmp_path / 'alone')
    done = subprocess.run([sys.executable, script, out, '--out-dir', alone],
                          capture_output=True, text=True, timeout=120, env=dict(os.environ))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.split() == [os.path.join(alone, 'z_against_q.png'),
                                   os.path.join(alone, 'z_against_q.csv')]
    assert figures.output_differences(os.path.join(out, 'fig', 'figure'), alone) == []
    # the inputs given by hand, with no step looked up, load the same objects
    by_hand = str(tmp_path / 'by_hand')
    done = subprocess.run(
        [sys.executable, script, out, '--out-dir', by_hand]
        + [arg for name, source in INPUTS.items() for kind, step in source.items()
           for arg in ('--input', f'{name}={kind}:{step}')],
        capture_output=True, text=True, timeout=120, env=dict(os.environ))
    assert done.returncode == 0, done.stdout + done.stderr
    assert figures.output_differences(alone, by_hand) == []


@pytest.mark.unit
def test_a_step_error_reaches_the_other_ranks():
    import pickle
    error = pickle.loads(pickle.dumps(WorkflowStepError('fig', 'its function raised')))
    assert (error.step_id, str(error)) == ('fig', 'step "fig": its function raised')


@pytest.mark.unit
def test_a_figure_is_drawn_with_matplotlibs_own_settings(tmp_path):
    '''Whatever changed the settings before (importing libcuflynx does), and they are put back.'''
    import matplotlib
    seen = {}

    def draw(out_dir):
        seen['width'] = matplotlib.rcParams['lines.linewidth']
        path = os.path.join(out_dir, 'a.txt')
        with open(path, 'w') as f:
            f.write('a')
        return path

    with matplotlib.rc_context({'lines.linewidth': 7.0}):
        assert figures.call_figure(draw, {}, str(tmp_path)) == [str(tmp_path / 'a.txt')]
        assert matplotlib.rcParams['lines.linewidth'] == 7.0
    assert seen['width'] == matplotlib.rcParamsOrig['lines.linewidth']


@pytest.mark.unit
def test_differing_outputs_are_named(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, height in (('a', 1.0), ('b', 2.0)):
        os.makedirs(tmp_path / name)
        figure, axes = plt.subplots(figsize=(1, 1), dpi=30)
        axes.plot([0, 1], [0, height])
        figure.savefig(tmp_path / name / 'plot.png')
        plt.close(figure)
        (tmp_path / name / 'same.txt').write_text('same')
    (tmp_path / 'a' / 'extra.csv').write_text('1')
    assert figures.output_differences(str(tmp_path / 'a'), str(tmp_path / 'a')) == []
    assert figures.output_differences(str(tmp_path / 'a'), str(tmp_path / 'b')) == [
        f'extra.csv is only in {tmp_path / "a"}', 'plot.png differs']


@pytest.mark.integration
def test_a_packed_workflow_draws_the_same_figure(finished, tmp_path):
    library, path, out = finished
    packed = pack_workflow(path, str(tmp_path / 'pack'))
    assert {'figures/fig/z_against_q.py', 'figures/fig/style.py'} <= set(packed['files'])
    step = next(s for s in _read_json(packed['workflow'])['steps'] if s['id'] == 'fig')
    assert step['function'] == 'figures/fig/z_against_q.py:draw'
    assert step['files'] == ['figures/fig/style.py']
    again = str(tmp_path / 'out')
    run_calibration_workflow(packed['workflow'], output_dir=again)
    script = os.path.join(os.path.dirname(packed['workflow']), 'figures', 'fig', 'z_against_q.py')
    alone = str(tmp_path / 'alone')
    assert figures.output_differences(os.path.join(again, 'fig', 'figure'),
                                      _run_alone(script, again, alone)) == []


@pytest.mark.integration
def test_a_figure_is_redrawn_when_it_is_out_of_date_and_not_otherwise(library, tmp_path):
    path, out = _with_figure(library), str(tmp_path / 'out')
    run_calibration_workflow(path, output_dir=out)
    events = []

    def run(**keys):
        events.clear()
        run_calibration_workflow(path, output_dir=out, progress=events.append, **keys)
        return [(e['event'], e['step']) for e in events if 'step' in e and e['step'] == 'fig']

    # nothing it reads changed: only the step asked for runs
    assert run(only='rest')[0] == ('step_started', 'fig')    # rest, which it reads, ran again
    first = _read_json(os.path.join(out, 'fig', 'step_result.json'))['started']
    raw = _read_json(path)
    raw['steps'].append(dict(raw['steps'][2], id='other_scan'))
    _write_json(path, raw)
    assert run(only='other_scan') == [('step_skipped', 'fig')]
    assert _read_json(os.path.join(out, 'fig', 'step_result.json'))['started'] == first

    # a file it lists changed: stale, and redrawn by any run
    style = os.path.join(os.path.dirname(path), 'figures', 'style.py')
    with open(style, 'w') as f:
        f.write("COLOUR = 'tab:blue'\n")
    assert workflow_status(path, out)['steps']['fig']['stale']
    node = next(n for n in provenance_graph(out)['nodes'] if n['id'] == 'fig')
    assert node['stale'] and {e['key']: e['status'] for e in node['inputs']}['file:style.py'] == \
        'stale'
    # an undrawn figure does not hold back the calibrated parameters
    write_calibrated_parameters_to(path, out, os.path.join(out, 'calibrated.json'))
    assert run(only='other_scan')[0] == ('step_started', 'fig')
    assert not workflow_status(path, out)['steps']['fig']['stale']

    # a step it reads is out of date (fit_a ran again after scan): it waits, and stays stale
    assert run(only='fit_a') == [('step_skipped', 'fig')]
    assert workflow_status(path, out)['steps']['fig']['stale']
    with pytest.raises(WorkflowError, match='needs the result of "scan"'):
        run(only='fig')

    # a function that raises fails its step once the run's results are written
    run(from_step='rest')
    with open(os.path.join(os.path.dirname(path), 'figures', 'z_against_q.py'), 'a') as f:
        f.write('\n\ndef bad(curves, out_dir):\n    raise RuntimeError("no data to draw")\n')
    raw['steps'].append({'id': 'bad', 'kind': 'figure',
                         'function': 'figures/z_against_q.py:bad',
                         'inputs': {'curves': {'sweep': 'scan'}}})
    _write_json(path, raw)
    os.remove(os.path.join(out, 'workflow_result.json'))
    with pytest.raises(WorkflowStepError, match='its function raised RuntimeError: no data'):
        run(only='other_scan')
    assert _read_json(os.path.join(out, 'workflow_result.json'))['steps']['bad'] is None
    assert ('step_failed', 'bad') in [(e['event'], e.get('step')) for e in events]
