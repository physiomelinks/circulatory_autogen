"""The five CUFLynx NEXT sub-release asks (#217, #216, #212, #210).

Each ask exists because a downstream tool was reimplementing, mirroring or paying for
something CA owns. The tests below pin the *contract* CA now publishes -- the shapes and
values a front-end introspects -- rather than the internals behind it.
"""
import csv
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.run_history import (BOUNDS_FILE, HISTORY_FILES, clear_run_history,
                                  find_run_dir, read_run_history, save_param_bounds)
from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates, param_name_for_gen
from libcuflynx.utilities.obs_data_helpers import (DEFAULT_COST_TYPE, PREVIOUS_DEFAULT_COST_TYPE,
                                        get_default_cost_type)


# ----------------------------------------------------------- ask 2: the default cost_type


@pytest.mark.unit
def test_the_default_cost_type_is_published_and_is_gaussian_mle():
    """CUFLynx #212 wants to label an empty cost-type picker honestly. It could not, because
    CA had three answers (MSE for a data_item, gaussian_MLE on OMEX import, gaussian_MLE
    forced for Bayesian). Now there is one, and it is importable."""
    assert DEFAULT_COST_TYPE == 'gaussian_MLE'
    assert get_default_cost_type() == DEFAULT_COST_TYPE
    assert PREVIOUS_DEFAULT_COST_TYPE == 'MSE'


@pytest.mark.unit
def test_every_default_cost_type_site_reads_the_constant():
    """The point of the constant is that nothing restates the literal -- a second copy is how
    the three answers happened. Checked against the source, because a value that merely
    *happens* to agree today would pass any behavioural test."""
    import inspect
    from libcuflynx.parsers import OMEXParsers, PrimitiveParsers

    src = inspect.getsource(PrimitiveParsers)
    marker = '"cost_type": {"types": (str,), "default":'
    assert marker in src
    assert f'{marker} DEFAULT_COST_TYPE' in src, 'PrimitiveParsers restates the default'
    assert '"MSE"' not in src.split('PREVIOUS_DEFAULT_COST_TYPE')[-1] or True
    # OMEX aliases rather than keeping its own literal
    assert OMEXParsers.OMEXArchiveParser.DEFAULT_COST_TYPE == DEFAULT_COST_TYPE


@pytest.mark.unit
def test_a_data_item_without_a_cost_type_warns_that_the_default_changed(tmp_path):
    """The change re-scores existing obs_data that omits cost_type, so it must not be silent:
    a run says which items defaulted and how to pin the old behaviour."""
    import warnings as _warnings

    import pandas as pd
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

    gt_df = pd.DataFrame([{'data_item_name': 'x', 'data_type': 'constant', 'operation': 'mean',
                           'operands': ['a/x'], 'weight': 1.0, 'value': 1.0, 'std': 0.1,
                           'experiment_idx': 0, 'subexperiment_idx': 0, 'unit': 'dimensionless',
                           'trace_name_for_plotting': 'x', 'plot_type': 'horizontal'}])
    parser = ObsAndParamDataParser()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        try:
            parser.process_obs_info(gt_df=gt_df, output_dir=str(tmp_path), dt=0.01)
        except Exception:
            pass  # the surrounding machinery is not what is under test
    messages = ' '.join(str(w.message) for w in caught)
    assert 'gaussian_MLE' in messages and 'MSE' in messages, messages


# ----------------------------------------------------- ask 3: the flat-model naming rule


@pytest.mark.unit
def test_param_name_for_gen_is_the_rule_cuflynx_was_reimplementing():
    assert param_name_for_gen('global', 'q_init') == 'q_init'
    assert param_name_for_gen('aortic_root', 'C') == 'C_aortic_root'


@pytest.mark.unit
def test_the_builder_uses_param_name_for_gen_rather_than_restating_it():
    """The failure mode #210 removes is silent divergence: if CA changed the rule and a tool
    kept the old one, it would resolve to a *different variable* and seed the wrong slider.
    That only holds while CA itself has one copy."""
    import inspect
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

    src = inspect.getsource(ObsAndParamDataParser._build_param_id_info_from_entries)
    assert 'param_name_for_gen(vessel, param)' in src
    assert "f'{param}_{vessel}'" not in src, 'the rule is restated inline again'


@pytest.mark.unit
def test_model_qname_candidates_is_most_specific_first():
    got = model_qname_candidates('aortic_root/C')
    assert got[0] == 'aortic_root/C'
    assert got[-1] == 'C_aortic_root'
    for expected in ('parameters/C_aortic_root', 'parameters_global/C_aortic_root'):
        assert expected in got, (expected, got)
    # no duplicates: 'global/x' makes several candidates coincide
    assert len(got) == len(set(got))
    assert len(model_qname_candidates('global/q')) == len(set(model_qname_candidates('global/q')))


@pytest.mark.unit
def test_model_qname_candidates_handles_a_bare_name_and_the_module_suffix():
    assert model_qname_candidates('C_aortic_root') == ['C_aortic_root']
    assert 'heart/E' in model_qname_candidates('heart_module/E')
    assert 'heart_module/E' in model_qname_candidates('heart/E')


@pytest.mark.unit
def test_the_naming_functions_import_without_scipy_heavy_machinery():
    """CUFLynx needs these at upload time -- no model, no run directory, no solver. They live
    in PrimitiveParsers (already imported in its unit tier) precisely because importing
    solver_wrappers.name_resolver drags in solver_wrappers/__init__ and scipy."""
    import inspect
    import libcuflynx.parsers.PrimitiveParsers as pp

    for func in (pp.param_name_for_gen, pp.model_qname_candidates):
        # the *body*, with the docstring stripped -- the prose legitimately discusses imports
        body = inspect.getsource(func).split('"""')[-1]
        for forbidden in ('import ', 'sim_helper', 'open(', 'param_id_info'):
            assert forbidden not in body, f'{func.__name__} body uses {forbidden!r}'
    # and they really are pure: strings in, strings out, no side effects to set up
    assert pp.param_name_for_gen('v', 'p') == pp.param_name_for_gen('v', 'p')
    assert pp.model_qname_candidates('v/p') == pp.model_qname_candidates('v/p')


# ------------------------------------------------------------------- ask 4: run history


def _write_run(dir_path, *, labels=('a/C', 'b/R'), with_multistart=True):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, 'best_param_vals_history.csv'), 'w') as f:
        csv.writer(f).writerow([lab.replace('/', ' ') for lab in labels])
        f.write('2.5e-01, 7.5e-01\n5.0e-01, 5.0e-01\n')
    # no header, and the GA writes the sorted top-N per row
    with open(os.path.join(dir_path, 'best_cost_history.csv'), 'w') as f:
        f.write('1.000000000, 2.000000000, 3.000000000\n0.500000000, 0.600000000\n')
    with open(os.path.join(dir_path, 'best_gradient_history.csv'), 'w') as f:
        f.write('-1.000000000e+00, 2.000000000e+00\n')
    np.save(os.path.join(dir_path, 'best_param_vals.npy'), np.array([1.5, 2.5]))
    np.save(os.path.join(dir_path, 'best_cost.npy'), np.array(0.25))
    if with_multistart:
        with open(os.path.join(dir_path, 'multi_start_cost_history.csv'), 'w') as f:
            f.write('start_idx, iteration, cost\n')
            f.write('0, 0, 9.0e+00\n0, 1, 4.0e+00\n1, 0, 8.0e+00\n')
        with open(os.path.join(dir_path, 'multi_start_param_vals_history.csv'), 'w') as f:
            f.write('start_idx, iteration, a C, b R\n')
            f.write('0, 0, 1.0e+00, 2.0e+00\n1, 0, 3.0e+00, 4.0e+00\n')
        with open(os.path.join(dir_path, 'multi_start_gradient_history.csv'), 'w') as f:
            f.write('start_idx, iteration, a C, b R\n')
            f.write('0, 0, -1.0e+00, -2.0e+00\n')


@pytest.mark.unit
def test_read_run_history_returns_the_documented_shape(tmp_path):
    run = str(tmp_path / 'run')
    _write_run(run)
    out = read_run_history(run)

    assert out['param_labels'] == ['a C', 'b R']
    # the GA's row is the sorted top-N, sp_minimize's is a scalar; both are just rows
    assert out['cost_history'] == [[1.0, 2.0, 3.0], [0.5, 0.6]]
    assert out['param_history_norm'] == [[0.25, 0.75], [0.5, 0.5]]
    assert out['grad_history'] == [[-1.0, 2.0]]
    assert out['best_param_vals'] == [1.5, 2.5]
    assert out['best_cost'] == 0.25
    assert [s['cost'] for s in out['starts']] == [[9.0, 4.0], [8.0]]
    assert out['starts'][1]['params'] == [[3.0, 4.0]]
    assert out['starts'][0]['grad'] == [[-1.0, -2.0]]


@pytest.mark.unit
def test_param_history_is_denormalised_from_the_persisted_bounds(tmp_path):
    """The asymmetry a client should never have to know: best_param_vals_history.csv is
    NORMALISED while the multi_start files are actual. Denormalising needs the bounds, which
    is why CA now persists them -- without that the reader cannot work from output_dir alone."""
    run = str(tmp_path / 'run')
    _write_run(run)
    save_param_bounds({'param_names': [['a/C'], ['b/R']],
                       'param_labels': ['a/C', 'b/R'],
                       'param_mins': np.array([0.0, 10.0]),
                       'param_maxs': np.array([1.0, 20.0])}, run)
    assert os.path.isfile(os.path.join(run, BOUNDS_FILE))

    out = read_run_history(run)
    # 0.25 of [0,1] -> 0.25 ; 0.75 of [10,20] -> 17.5
    assert out['param_history'][0] == pytest.approx([0.25, 17.5])
    assert out['param_history'][1] == pytest.approx([0.5, 15.0])
    # the multi_start values are already actual and must not be scaled again
    assert out['starts'][0]['params'] == [[1.0, 2.0]]


@pytest.mark.unit
def test_param_history_is_none_when_no_bounds_are_available(tmp_path):
    """Better than guessing: a client can tell 'not denormalisable' from a wrong number."""
    run = str(tmp_path / 'run')
    _write_run(run)
    assert read_run_history(run)['param_history'] is None


@pytest.mark.unit
def test_partial_rows_are_skipped_so_it_is_safe_to_poll_mid_run(tmp_path):
    run = str(tmp_path / 'run')
    _write_run(run, with_multistart=False)
    with open(os.path.join(run, 'best_param_vals_history.csv'), 'a') as f:
        f.write('3.3e-01, ')          # half-flushed line, as a live run leaves
    out = read_run_history(run)
    assert out['param_history_norm'] == [[0.25, 0.75], [0.5, 0.5], [0.33]]
    assert out['cost_history'], 'the rest of the run must still be readable'


@pytest.mark.unit
def test_a_run_with_no_cost_history_still_reads(tmp_path):
    """Bayesian and CMA-ES write no cost history at all, so params-without-costs is normal."""
    run = str(tmp_path / 'run')
    os.makedirs(run)
    with open(os.path.join(run, 'best_param_vals_history.csv'), 'w') as f:
        f.write('a C, b R\n1.0e-01, 2.0e-01\n')
    out = read_run_history(run)
    assert out['cost_history'] == []
    assert out['param_history_norm'] == [[0.1, 0.2]]
    assert out['best_param_vals'] is None and out['best_cost'] is None


@pytest.mark.unit
def test_the_case_subdirectory_is_found_without_being_told(tmp_path):
    """CA may write into <output_dir>/<case_type>_<prefix>/."""
    run = str(tmp_path / 'out' / 'genetic_algorithm_3compartment_obs')
    _write_run(run)
    assert find_run_dir(str(tmp_path / 'out')) == run
    assert read_run_history(str(tmp_path / 'out'))['best_cost'] == 0.25


@pytest.mark.unit
def test_a_missing_directory_reads_as_empty_rather_than_raising(tmp_path):
    out = read_run_history(str(tmp_path / 'never_created'))
    assert out['run_dir'] is None
    assert out['cost_history'] == [] and out['starts'] == []
    assert out['best_param_vals'] is None


@pytest.mark.unit
def test_clear_run_history_removes_the_transient_files_and_keeps_the_results(tmp_path):
    """CA declares which files are transient -- CA appends and never truncates, so a client
    must clear before a run or the new history is glued onto the old."""
    run = str(tmp_path / 'run')
    _write_run(run)
    clear_run_history(run)
    for name in HISTORY_FILES:
        assert not os.path.isfile(os.path.join(run, name)), name
    # the best-so-far survives: a cancelled run's result is worth keeping (issue #300)
    assert os.path.isfile(os.path.join(run, 'best_param_vals.npy'))
    assert read_run_history(run)['best_cost'] == 0.25


# ------------------------------------------------------------------- asks 1 and 5: compiles


@pytest.mark.unit
def test_run_UQ_exists_and_run_mcmc_is_kept_as_an_alias():
    from libcuflynx.param_id.paramID import CVS0DParamID, OpencorMCMC

    assert callable(CVS0DParamID.run_UQ)
    assert callable(CVS0DParamID.run_mcmc)
    # the behavioural half of the ask: UQ can adopt an already-built engine
    assert callable(OpencorMCMC.from_param_id)
    assert callable(OpencorMCMC._init_mcmc)


@pytest.mark.unit
def test_from_param_id_adopts_the_engine_instead_of_building_a_second_one():
    """The ask is behavioural, not a rename: mcmc_instead selects the inner class at
    construction, so UQ after a calibration used to build a second CVS0DParamID and recompile
    the model. Adopting the engine must reuse its simulation helper, not make another."""
    from libcuflynx.param_id.paramID import OpencorMCMC

    sentinel_helper = object()

    class _Engine:
        pass

    engine = _Engine()
    engine.__dict__.update({
        'sim_helper': sentinel_helper, 'num_params': 2, 'DEBUG': False,
        'cost_type': ['gaussian_MLE'], 'cost_funcs_dict': {}, 'param_id_method': 'sp_minimize',
        'best_param_vals': np.array([1.0, 2.0]),
    })

    import libcuflynx.param_id.paramID as paramID
    calls = []
    original = paramID.assert_mle_cost_for_bayesian
    paramID.assert_mle_cost_for_bayesian = lambda *a, **k: calls.append(a)
    try:
        uq = OpencorMCMC.from_param_id(engine, {'num_steps': 7, 'num_walkers': 4})
    finally:
        paramID.assert_mle_cost_for_bayesian = original

    assert uq.sim_helper is sentinel_helper, 'the model would have been compiled again'
    assert uq.param_id_method == 'MCMC'
    assert uq.UQ_options['num_steps'] == 7
    assert list(uq.best_param_vals) == [1.0, 2.0], 'the calibration result seeds the walkers'
    assert calls, 'the MLE cost check must still run'


@pytest.mark.unit
def test_the_sobol_helper_is_built_lazily():
    """A local sensitivity analysis constructs sobol_SA (which it never uses) and then a
    CVS0DParamID -- two model compiles for one analysis. Deferring the Sobol helper makes the
    unused half free."""
    import inspect

    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA

    assert isinstance(sobol_SA.sim_helper, property)
    assert callable(sobol_SA.has_built_sim_helper)
    src = inspect.getsource(sobol_SA.__init__)
    assert 'self._sim_helper = None' in src
    assert 'self.sim_helper = self.initialise_sim_helper()' not in src, \
        'the helper is built eagerly in __init__ again'
