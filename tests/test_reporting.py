"""Variable mappings and methods LaTeX (libcuflynx.reporting, cuflynx-variable-mapping,
cuflynx-methods-latex).

The unit tier uses resources/Goodwin.cellml (any CellML, not a generated model) and small
Myokit models built in code; the integration tier runs the calibration-workflow fixture
library (libcuflynx.external_testing.workflow_library) and, when pdflatex is installed,
compiles what it writes.
"""
import csv
import os
import shutil
import subprocess

import myokit
import pytest

from libcuflynx.reporting import (calibration_methods_latex, check_variable_mapping,
                                  default_variable_mapping, module_sections,
                                  read_variable_mapping, update_variable_mapping_file,
                                  workflow_methods_latex, write_variable_mapping)
from libcuflynx.reporting import flowchart
from libcuflynx.reporting.latex_names import escape, symbol
from libcuflynx.reporting.model_latex import format_number, format_units
from libcuflynx.reporting.variable_mapping import VariableMappingError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(REPO_ROOT, 'resources')
GOODWIN = os.path.join(RESOURCES, 'Goodwin.cellml')

needs_latex = pytest.mark.skipif(shutil.which('pdflatex') is None, reason='pdflatex not installed')


def _compile(tex_path):
    """Compile twice (for the figure reference); returns the log."""
    for _ in range(2):
        done = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                               os.path.basename(tex_path)], cwd=os.path.dirname(tex_path),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=180)
        assert done.returncode == 0, done.stdout[-3000:]
    with open(os.path.splitext(tex_path)[0] + '.log', errors='replace') as f:
        return f.read()


# --------------------------------------------------------------------------------------------
# the default rule
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('name, extra, expected', [
    ('V', (), 'V'),
    ('g_leak_Na', (), r'g_{\mathrm{leak},\mathrm{Na}}'),
    ('rho_M', (), r'\rho_{M}'),
    ('Delta_C', (), r'\Delta_{C}'),
    ('Alpha', (), r'\alpha'),          # no \Alpha in LaTeX
    ('Nai', (), r'\mathrm{Nai}'),
    ('V0', (), 'V_{0}'),
    ('q_us_0', (r'\mathrm{pvn}',), r'q_{\mathrm{us},0,\mathrm{pvn}}'),
    ('tau_w', (), r'\tau_{w}'),
])
def test_the_default_symbol_rule(name, extra, expected):
    assert symbol(name, extra) == expected


@pytest.mark.unit
def test_escaping_and_numbers_and_units():
    assert escape('a_b%c') == r'a\_b\%c'
    assert format_number(0.00012345) == r'0.0001234'
    assert format_number(1.5e-9) == r'1.5\times10^{-9}'
    assert format_number(0) == '0'
    assert format_units('m^4*s^2/g (0.001)') == \
        r'$\mathrm{m}^{4}\,\mathrm{s}^{2}/\mathrm{g}\ (\times 0.001)$'
    assert format_units('dimensionless') == 'dimensionless'


# --------------------------------------------------------------------------------------------
# the mapping file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_mapping_has_every_variable_by_its_canonical_name():
    rows = default_variable_mapping(GOODWIN)
    by_name = {r['variable_name']: r for r in rows}
    assert rows[0]['variable_name'] == 'environment/time' and rows[0]['latex'] == 't'
    assert by_name['Yi/alpha_i']['latex'] == r'\alpha_{i,\mathrm{Yi}}'
    assert by_name['Xi/Xi']['kind'] == 'state'
    assert by_name['Xi/ai']['kind'] == 'constant'
    assert set(rows[0]) >= {'variable_name', 'latex', 'kind', 'component', 'units'}


def _generated_style_model():
    """A model laid out the way the generator lays one out: hoisted parameters in
    `parameters`, modules as `<vessel>_module`, a global, and a piecewise equation."""
    return myokit.parse_model('''
[[model]]
mod_A_module.x = 1

[environment]
time = 0 bind time

[parameters]
p_mod_A = 2
k_mod_B = 0.5

[parameters_global]
R = 8.314

[mod_A_module]
dot(x) = -parameters.p_mod_A * x + parameters_global.R * 0
y = piecewise(x > 0.5, x, 0)

[mod_B_module]
z = parameters.k_mod_B * mod_A_module.y
''')


@pytest.mark.unit
def test_generated_names_split_into_symbol_and_module():
    rows = {r['variable_name']: r for r in default_variable_mapping(_generated_style_model())}
    # generated alone as `mod`: the prefix is dropped from the module subscript
    assert rows['parameters/p_mod_A']['latex'] == r'p_{\mathrm{A}}'
    assert rows['parameters/k_mod_B']['latex'] == r'k_{\mathrm{B}}'
    assert rows['parameters_global/R']['latex'] == 'R'
    assert rows['mod_A/x']['latex'] == r'x_{\mathrm{A}}'
    assert rows['mod_B/z']['kind'] == 'algebraic'
    assert rows['parameters/p_mod_A']['kind'] == 'parameter'


@pytest.mark.unit
def test_refreshing_a_mapping_keeps_edits_and_follows_the_model(tmp_path):
    path = str(tmp_path / 'Goodwin_variable_mapping.csv')
    rows = update_variable_mapping_file(GOODWIN, path)
    assert not any(r['edited'] for r in rows)
    # the user edits one symbol, and a variable the model does not have is in the file
    edited = read_variable_mapping(path)
    edited['Xi/Xi']['latex'] = 'X_i'
    edited['Gone/old'] = {'variable_name': 'Gone/old', 'latex': 'o'}
    write_variable_mapping(list(edited.values()), path)
    rows = update_variable_mapping_file(GOODWIN, path)
    by_name = {r['variable_name']: r for r in rows}
    assert by_name['Xi/Xi']['latex'] == 'X_i' and by_name['Xi/Xi']['edited']
    assert 'Gone/old' not in read_variable_mapping(path)
    with open(path) as f:
        assert next(csv.reader(f))[:2] == ['variable_name', 'latex']


@pytest.mark.unit
def test_problems_a_reader_would_notice_are_reported(tmp_path):
    rows = [{'variable_name': 'a/x', 'latex': 'x'}, {'variable_name': 'b/x', 'latex': 'x'},
            {'variable_name': 'c/y', 'latex': ''}]
    problems = check_variable_mapping(rows)
    assert problems == {'empty': ['c/y'], 'duplicates': {'x': ['a/x', 'b/x']}}
    with pytest.raises(VariableMappingError, match='twice'):
        write_variable_mapping(rows + [{'variable_name': 'a/x', 'latex': 'z'}],
                               str(tmp_path / 'm.csv'))
    with pytest.raises(VariableMappingError, match='only CellML'):
        default_variable_mapping(str(tmp_path / 'model.py'))


# --------------------------------------------------------------------------------------------
# equations
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_each_module_is_a_section_in_the_mapped_symbols():
    model = _generated_style_model()
    text = module_sections(model, {'parameters/p_mod_A': {'latex': r'\kappa'}},
                           module_info={'mod_A': {'module_type': 'lin_a', 'version': 'v1',
                                                  'parameterisation': 'fit_a'}})
    assert text.count(r'\subsection{') == 2
    assert r'\subsection{Module A}' in text and r'\texttt{lin\_a}' in text
    assert r'\frac{d{x_{\mathrm{A}}}}{dt} &= -\kappa\cdot x_{\mathrm{A}}' in text
    # a conditional is a cases block, not a function call
    assert r'\begin{cases}' in text and r'\text{otherwise}' in text
    assert 'piecewise' not in text
    # B's equation uses A's output under A's symbol, and B's parameter is in B's table
    assert r'z_{\mathrm{B}} &= k_{\mathrm{B}}\cdot y_{\mathrm{A}}' in text
    assert r'\texttt{parameters/k\_mod\_B}' in text
    # an operator is never glued to the operand after it
    assert r'\cdotx' not in text and r'\cdoty' not in text


# --------------------------------------------------------------------------------------------
# flow charts
# --------------------------------------------------------------------------------------------

def _step(step_id, depends=(), fixed=(), priors=()):
    return {'id': step_id,
            'model': {'module_type': 'm', 'version': 'v1', 'parameterisation': step_id},
            'n_parameters': 1, 'n_data_items': 2, 'settings': {'param_id_method': 'CMA-ES'},
            'depends_on': list(depends), 'fixed_from': list(fixed), 'priors_from': list(priors)}


@pytest.mark.unit
def test_a_workflow_flow_chart_follows_its_dependencies():
    target = {'id': None,
              'model': {'module_type': 'soma', 'version': 'v', 'parameterisation': 'r'},
              'n_merged': 3}
    nodes, edges = flowchart.workflow_graph([
        _step('a'), _step('b'), _step('c', depends=['a', 'b'], fixed=['a'], priors=['b']),
        target])
    by_edge = {(e['from'], e['to']): e for e in edges}
    assert by_edge[('a', 'c')]['style'] == 'fixed' and by_edge[('a', 'c')]['label'] == 'fixed'
    assert by_edge[('b', 'c')]['style'] == 'prior'
    assert ('c', '__target__') in by_edge and ('a', '__target__') not in by_edge
    picture = flowchart.tikz(nodes, edges)
    # a and b side by side on the first row, c below them, the target last
    assert '(na) at (0.00,0.00)' in picture and '(nb) at (5.20,0.00)' in picture
    assert '(nc) at (2.60,-2.60)' in picture
    assert flowchart.width_cm(nodes, edges) == pytest.approx(5.2 + 4.6)
    assert 'CMA-ES' in picture


@pytest.mark.unit
def test_a_flow_chart_merges_only_calibrations_and_labels_what_each_step_takes():
    target = {'id': None,
              'model': {'module_type': 'soma', 'version': 'v', 'parameterisation': 'r'},
              'n_merged': 1}
    screen = dict(_step('screen'), kind='sensitivity', settings={'sa_options': {}})
    reduce = {'id': 'reduce', 'kind': 'reduction', 'model': None, 'n_parameters': 0,
              'n_data_items': 0, 'settings': {}, 'depends_on': ['screen'], 'fixed_from': [],
              'priors_from': [], 'sensitivity_from': 'screen', 'reduction': {'top_n': 2}}
    fit = dict(_step('fit', depends=['reduce']), params_for_id_from='reduce')
    ia = dict(_step('ia', depends=['fit']), kind='identifiability', result_from='fit',
              settings={'ia_options': {'method': 'Laplace'}})
    nodes, edges = flowchart.workflow_graph([screen, reduce, fit, ia, target])
    by_edge = {(e['from'], e['to']): e for e in edges}
    assert [by_edge[k]['label'] for k in (('screen', 'reduce'), ('reduce', 'fit'),
                                          ('fit', 'ia'))] == ['indices', 'parameters',
                                                              'at its fit']
    # fit's values reach the target although ia analyses them; the analyses add none
    assert [k for k in by_edge if k[1] == '__target__'] == [('fit', '__target__')]
    kinds = {n['id']: n['kind'] for n in nodes}
    assert kinds['fit'] == 'step' and kinds['screen'] == kinds['ia'] == 'analysis'
    lines = {n['id']: n['lines'] for n in nodes}
    assert lines['reduce'] == ['keeps the top 2 by Sobol ST']
    assert lines['screen'][-1] == 'Sobol sensitivity analysis'
    assert lines['ia'][-1] == 'Laplace identifiability analysis'


@pytest.mark.integration
def test_methods_of_a_workflow_with_analyses_describe_each_kind(tmp_path):
    from libcuflynx.calibration_workflow import run_calibration_workflow
    from libcuflynx.external_testing.workflow_library import build_workflow_library

    library = build_workflow_library(str(tmp_path / 'library'))
    run_dir = str(tmp_path / 'run')
    text = workflow_methods_latex(library['chain_analysed'])
    assert ('The workflow has 5 steps (Figure~\\ref{fig:calibration-flowchart}), run in the '
            'order shown: 2 calibrations, 1 sensitivity analysis, 1 reduction and 1 '
            'identifiability analysis.') in text
    assert ('This step ranks the parameters of step screen by the max over the observables '
            'of their total Sobol index, and keeps the top 1.') in text
    assert 'It calibrates the parameters kept by step reduce.' in text
    assert 'analyses the identifiability of parameterisation' in text

    run_calibration_workflow(library['chain_analysed'], output_dir=run_dir)
    text = workflow_methods_latex(library['chain_analysed'], run_dir=run_dir)
    assert 'It kept \\texttt{mod\\_\\_C/q} (' in text
    assert 'The standard deviations of the Laplace approximation: mod\\_\\_C/q: ' in text
    # only the calibrations' values are the target's; q is fit's
    assert r'Calibrated in step \texttt{fit} (Section~\ref{sec:step:fit})' in text
    if shutil.which('pdflatex'):
        tex = str(tmp_path / 'methods.tex')
        with open(tex, 'w') as f:
            f.write(workflow_methods_latex(library['chain_analysed'], run_dir=run_dir))
        assert 'Undefined control sequence' not in _compile(tex)


@pytest.mark.unit
def test_a_single_calibration_flow_chart_has_its_stages():
    nodes, edges = flowchart.single_calibration_graph(3, 4, 2, {}, do_uq=True, do_ia=True)
    ids = [n['id'] for n in nodes]
    assert ids == ['model', 'data', 'params', 'fit', 'uq', 'ia', 'out']
    assert {'from': 'uq', 'to': 'out'} in [{k: e[k] for k in ('from', 'to')} for e in edges]


# --------------------------------------------------------------------------------------------
# methods documents
# --------------------------------------------------------------------------------------------

def _goodwin_methods(**kwargs):
    return calibration_methods_latex(
        GOODWIN, os.path.join(RESOURCES, 'Goodwin_obs_data.json'),
        os.path.join(RESOURCES, 'Goodwin_params_for_id.json'),
        settings={'param_id_method': 'genetic_algorithm',
                  'optimiser_options': {'num_calls_to_function': 500,
                                        'cost_type': 'gaussian_MLE'},
                  'solver_info': {'solver': 'CVODE_myokit'}, 'dt': 0.1}, **kwargs)


@pytest.mark.unit
def test_methods_of_one_calibration():
    text = _goodwin_methods()
    assert text.startswith(r'\documentclass') and text.rstrip().endswith(r'\end{document}')
    assert r'\section{Model}' in text and r'\section{Calibration}' in text
    assert r'\begin{tikzpicture}' in text and r'\label{fig:calibration-flowchart}' in text
    assert 'Genetic algorithm' in text and 'gaussian\\_MLE' in text
    assert r'\paragraph{Observables.}' in text and r'\paragraph{Parameters.}' in text
    body = _goodwin_methods(standalone=False)
    assert not body.startswith(r'\documentclass') and r'\begin{document}' not in body


@pytest.mark.unit
@needs_latex
def test_methods_of_one_calibration_compile(tmp_path):
    path = str(tmp_path / 'goodwin.tex')
    with open(path, 'w') as f:
        f.write(_goodwin_methods(do_uq=True, do_ia=True))
    log = _compile(path)
    assert 'Undefined control sequence' not in log
    assert os.path.isfile(str(tmp_path / 'goodwin.pdf'))


@pytest.mark.integration
def test_methods_of_a_workflow_report_each_step_and_its_results(tmp_path):
    from libcuflynx.calibration_workflow import run_calibration_workflow
    from libcuflynx.external_testing.workflow_library import build_workflow_library

    library = build_workflow_library(str(tmp_path / 'library'))
    run_dir = str(tmp_path / 'run')

    # before any run: the plan, without results
    text = workflow_methods_latex(library['chain_rest'])
    assert r'\subsection{Step fit\_a}' in text and r'\subsection{Step rest}' in text
    assert r'\paragraph{Result.}' not in text
    assert 'were fixed in this step' in text

    run_calibration_workflow(library['chain_rest'], output_dir=run_dir)

    # the mapping goes beside the target parameterisation, and its edits reach the document
    from libcuflynx.reporting.methods import update_workflow_variable_mapping
    path, rows = update_workflow_variable_mapping(library['chain_rest'], run_dir=run_dir)
    assert path == os.path.join(library['chain/rest'], 'rest_variable_mapping.csv')
    mapping = read_variable_mapping(path)
    assert mapping['parameters/p_mod__A']['latex'] == r'p_{\mathrm{A}}'
    mapping['parameters/p_mod__A']['latex'] = r'\pi_{A}'
    write_variable_mapping(list(mapping.values()), path)

    text = workflow_methods_latex(library['chain_rest'], run_dir=run_dir)
    assert r'\pi_{A}' in text and r'p_{\mathrm{A}}' not in text
    assert r'\paragraph{Result.}' in text
    # the value rest was given, in its symbol, and where it came from
    assert r'$\pi_{A}$$\,=\,$$2$ (from fit\_a)' in text
    # the modules carry their module type from the supermodule config
    assert r'\texttt{lin\_a}, version \texttt{v1}' in text
    # the flow chart: fit_a feeds rest as a fixed value, rest is merged into the target
    assert r'-- (nrest) node[midway, fill=white, inner sep=1pt, font=\scriptsize] {fixed}' \
        in text
    assert 'Merged: chain' in text
    # every parameter with its source: k cites lin_c's references, p and q name their step
    assert r'\section{Parameters and sources}' in text
    assert r'\cite{doe2020gain}; Table 1: the gain of C' in text
    assert r'Calibrated in step \texttt{fit\_a} (Section~\ref{sec:step:fit-a})' in text
    assert r'Calibrated in step \texttt{rest} (Section~\ref{sec:step:rest})' in text
    assert r'\bibliography{' in text and 'lin_c_v1_references' in text

    chart = workflow_methods_latex(library['chain_rest'], flowchart_only=True,
                                   standalone=False)
    assert r'\begin{tikzpicture}' in chart and r'\section{' not in chart

    if shutil.which('pdflatex'):
        # the document as the command writes it: the cited entries in methods.bib beside it,
        # compiled with BibTeX when latexmk is there (open methods.pdf to read it)
        tex, bib = str(tmp_path / 'methods.tex'), str(tmp_path / 'methods.bib')
        with open(tex, 'w') as f:
            f.write(workflow_methods_latex(library['chain_rest'], run_dir=run_dir, bib_out=bib))
        with open(bib) as f:
            assert f.read().startswith('@article{doe2020gain,')
        if shutil.which('latexmk'):
            done = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode',
                                   '-halt-on-error', 'methods.tex'], cwd=str(tmp_path),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  timeout=300)
            assert done.returncode == 0, done.stdout[-3000:]
            with open(str(tmp_path / 'methods.bbl')) as f:
                assert 'A gain of one' in f.read()
            with open(str(tmp_path / 'methods.log'), errors='replace') as f:
                log = f.read()
        else:
            log = _compile(tex)
        assert 'Undefined control sequence' not in log


@pytest.mark.integration
def test_the_commands_write_the_mapping_and_the_methods(tmp_path):
    from libcuflynx.external_testing.workflow_library import build_workflow_library
    from libcuflynx.scripts import methods_latex_script, variable_mapping_script

    mapping = str(tmp_path / 'goodwin_map.csv')
    assert variable_mapping_script.main([GOODWIN, '-o', mapping]) == 0
    assert 'Xi/Xi' in read_variable_mapping(mapping)

    out = str(tmp_path / 'goodwin.tex')
    assert methods_latex_script.main([
        '--model', GOODWIN, '--obs-data', os.path.join(RESOURCES, 'Goodwin_obs_data.json'),
        '--params-for-id', os.path.join(RESOURCES, 'Goodwin_params_for_id.json'),
        '--mapping', mapping, '-o', out, '--body-only']) == 0
    with open(out) as f:
        assert r'\section{Model}' in f.read()

    library = build_workflow_library(str(tmp_path / 'library'))
    assert variable_mapping_script.main(['--workflow', library['twin_split']]) == 0
    assert os.path.isfile(os.path.join(library['twin/split'], 'split_variable_mapping.csv'))
    out = str(tmp_path / 'twin.tex')
    assert methods_latex_script.main(['--workflow', library['twin_split'], '-o', out]) == 0
    with open(out) as f:
        text = f.read()
    # two independent steps: side by side, both merged into the target
    assert '(nfitxa) at (0.00,0.00)' in text and '(nfitxb) at (5.20,0.00)' in text
    assert text.count(r'-- (nxxtargetxx)') == 2


# --------------------------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_source_cites_the_keys_of_its_bibliography(tmp_path):
    from libcuflynx.reporting.bibliography import cite, cited_keys, read_bib, write_bib
    bib = tmp_path / 'refs.bib'
    bib.write_text('@comment{ignored, not an entry}\n'
                   '@article{martin2023modelling,\n  title = {A {Nested} title},\n  year = {2023}\n}\n'
                   '@book{boron2017medical, title = {Medical physiology}}\n')
    entries = read_bib([str(bib)])
    assert sorted(entries) == ['boron2017medical', 'martin2023modelling']
    assert entries['martin2023modelling'].endswith('year = {2023}\n}')
    text = 'martin2023modelling; Methods (p. 16): w_inf k = 10 mV, cf. boron2017medical'
    assert cite(text, entries) == (r'\cite{martin2023modelling}; Methods (p. 16): w\_inf k = '
                                   r'10 mV, cf. \cite{boron2017medical}')
    assert cited_keys(text, entries) == ['martin2023modelling', 'boron2017medical']
    # a word that is not a key stays text
    assert cite('definitional; user_defined', entries) == r'definitional; user\_defined'
    out = write_bib({k: entries[k] for k in ['martin2023modelling']}, str(tmp_path / 'o.bib'))
    assert read_bib([out]).keys() == {'martin2023modelling'}
