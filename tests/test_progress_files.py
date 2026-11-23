"""progress.json and the files beside it (libcuflynx.utilities.progress): the writer, the reports
of the MPI ranks, and what a Sobol run, an emulator training and a profile likelihood put in
them. The runs through a real model are in test_sensitivity_analysis.py and
test_emulator_training.py."""
import json
import os
import warnings

import numpy as np
import pytest

from libcuflynx.schemas import PROGRESS_SCHEMA, load_schema
from libcuflynx.schemas.validation import validate
from libcuflynx.utilities import progress as progress_module
from libcuflynx.utilities.progress import (KINDS, PHASE_STATES, PROGRESS_FILE, STATES,
                                           ProgressFile, RankReports)


def read_progress(directory):
    with open(os.path.join(str(directory), PROGRESS_FILE)) as f:
        document = json.load(f)
    validate(PROGRESS_SCHEMA, document, source=PROGRESS_FILE, what='progress file')
    return document


# ---- the writer ------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_schema_lists_the_kinds_and_states_the_writer_uses():
    schema = load_schema(PROGRESS_SCHEMA)
    assert schema['properties']['kind']['enum'] == list(KINDS)
    assert schema['properties']['state']['enum'] == list(STATES)
    assert schema['$defs']['phase']['properties']['state']['enum'] == list(PHASE_STATES)
    assert schema['properties']['schema_version']['const'] == progress_module.SCHEMA_VERSION


@pytest.mark.unit
def test_a_run_goes_from_running_to_done_with_its_phases(tmp_path):
    with ProgressFile(str(tmp_path), 'sobol', min_interval=0) as progress:
        first = read_progress(tmp_path)
        assert first['state'] == 'running' and first['phases'] == []
        progress.phase('sampling', 4)
        assert read_progress(tmp_path)['phases'] == [
            {'name': 'sampling', 'state': 'queued', 'done': 0, 'total': 4, 'failed': 0}]
        progress.advance('sampling', done=3, failed=1)
        assert read_progress(tmp_path)['phases'][0] == {
            'name': 'sampling', 'state': 'running', 'done': 3, 'total': 4, 'failed': 1}
        progress.advance('sampling')
        assert read_progress(tmp_path)['phases'][0]['state'] == 'done'
    last = read_progress(tmp_path)
    assert last['state'] == 'done' and last['kind'] == 'sobol' and last['schema_version'] == 1
    assert last['started'] <= last['updated']
    assert os.listdir(str(tmp_path)) == [PROGRESS_FILE], 'no temporary file is left'


@pytest.mark.unit
def test_an_error_leaves_the_run_and_its_running_phase_failed(tmp_path):
    with pytest.raises(RuntimeError, match='the model broke'):
        with ProgressFile(str(tmp_path), 'emulation', min_interval=0) as progress:
            progress.phase('sampling', 10)
            progress.advance('sampling', done=2)
            progress.phase('later', 3)
            raise RuntimeError('the model broke')
    document = read_progress(tmp_path)
    assert document['state'] == 'failed'
    assert [phase['state'] for phase in document['phases']] == ['failed', 'queued']


@pytest.mark.unit
def test_unforced_updates_are_written_at_most_once_per_interval(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(progress_module.time, 'monotonic', lambda: clock[0])
    progress = ProgressFile(str(tmp_path), 'sobol', min_interval=1.0)
    progress.phase('sampling', 100)
    assert progress.advance('sampling') is True, 'the first item starts the phase: forced'
    assert progress.advance('sampling') is False
    assert read_progress(tmp_path)['phases'][0]['done'] == 1
    clock[0] += 0.9
    assert progress.advance('sampling') is False
    clock[0] += 0.2
    assert progress.advance('sampling') is True
    assert read_progress(tmp_path)['phases'][0]['done'] == 4
    assert progress.advance('sampling', force=True) is True
    progress.finish()
    assert read_progress(tmp_path)['phases'][0]['done'] == 5


@pytest.mark.unit
def test_two_phases_can_share_a_name(tmp_path):
    progress = ProgressFile(str(tmp_path), 'emulation', min_interval=0)
    progress.phase('sampling_1', 8, name='sampling', stage=1, method='sobol')
    progress.phase('sampling_2', 4, name='sampling', stage=2, method='gradient_weighted')
    progress.advance('sampling_2', done=4)
    phases = read_progress(tmp_path)['phases']
    assert [(p['name'], p['stage'], p['done']) for p in phases] == [
        ('sampling', 1, 0), ('sampling', 2, 4)]


class _Rank:
    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return 2


@pytest.mark.unit
def test_only_rank_0_writes_and_no_directory_writes_nothing(tmp_path):
    worker = ProgressFile(str(tmp_path), 'sobol', comm=_Rank(1), min_interval=0)
    worker.phase('sampling', 2)
    worker.advance('sampling', done=2)
    worker.finish()
    assert os.listdir(str(tmp_path)) == []
    nowhere = ProgressFile(None, 'sobol', min_interval=0)
    nowhere.phase('sampling', 2)
    nowhere.advance('sampling')
    nowhere.finish()
    ProgressFile(str(tmp_path), 'sobol', comm=_Rank(0), min_interval=0)
    assert os.listdir(str(tmp_path)) == [PROGRESS_FILE]


@pytest.mark.unit
def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match='progress kind'):
        ProgressFile(str(tmp_path), 'sobl')


@pytest.mark.unit
def test_a_file_that_cannot_be_written_warns_once_and_the_run_goes_on(tmp_path, monkeypatch):
    progress = ProgressFile(str(tmp_path), 'sobol', min_interval=0)
    progress.phase('sampling', 3)

    def full_disk(path, document):
        raise OSError('No space left on device')

    monkeypatch.setattr(progress_module, 'write_json_atomic', full_disk)
    with pytest.warns(UserWarning, match='No space left on device'):
        assert progress.advance('sampling') is False
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        progress.advance('sampling')
        progress.finish()
    assert progress.document['phases'][0]['done'] == 2


# ---- the ranks' reports ----------------------------------------------------------------------


class _Wire:
    """The messages between the fake ranks of one test."""

    def __init__(self, size):
        self.size, self.queue = size, []


class _Request:
    def Test(self):
        return True

    def wait(self):
        return None


class _FakeComm:
    def __init__(self, wire, rank):
        self.wire, self.rank = wire, rank

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.wire.size

    def isend(self, obj, dest=None, tag=None):
        assert dest == 0
        self.wire.queue.append((tag, obj))
        return _Request()

    def iprobe(self, tag=None):
        return any(t == tag for t, _ in self.wire.queue)

    def recv(self, tag=None):
        for i, (t, obj) in enumerate(self.wire.queue):
            if t == tag:
                del self.wire.queue[i]
                return obj
        raise AssertionError('recv with nothing waiting would block')


@pytest.mark.unit
def test_one_rank_reports_its_own_items_and_sends_nothing():
    seen = []
    reports = RankReports(None, lambda indices, ok: seen.append((indices.tolist(), ok.tolist())))
    reports.add(0)
    reports.add(1, ok=False)
    reports.close()
    assert seen == [([0], [True]), ([1], [False])]


@pytest.mark.unit
def test_rank_0_sees_every_ranks_items_once_all_have_closed(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(progress_module.time, 'monotonic', lambda: clock[0])
    wire, status = _Wire(3), np.zeros(9, dtype=int)

    def on_report(indices, ok):
        assert np.all(status[indices] == 0), 'an item is reported once'
        status[indices] = np.where(ok, 1, 2)

    root = RankReports(_FakeComm(wire, 0), on_report)
    workers = [RankReports(_FakeComm(wire, rank), None) for rank in (1, 2)]

    root.add(0)
    workers[0].add(3)
    workers[0].add(4, ok=False)
    assert wire.queue == [], 'a worker sends about once per interval, not per item'
    clock[0] += 1.5
    workers[0].add(5)
    assert len(wire.queue) == 1
    root.add(1)                         # rank 0 receives between its own items
    assert status.tolist() == [1, 1, 0, 1, 2, 1, 0, 0, 0]
    workers[1].add(6)
    workers[1].add(7)
    workers[1].add(8, ok=False)
    root.add(2)
    for worker in workers:
        worker.close()
    root.close()                        # returns: both workers' last messages were waiting
    assert status.tolist() == [1, 1, 1, 1, 2, 1, 1, 1, 2]
    assert wire.queue == []


# ---- Sobol -----------------------------------------------------------------------------------


def _bare_sobol(tmp_path, num_samples=16):
    """A sobol_SA with no model: three parameters and two outputs that are functions of them."""
    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA
    manager = object.__new__(sobol_SA)
    manager.num_params, manager.rank, manager.verbose = 3, 0, False
    manager.output_dir = str(tmp_path)
    manager.SA_info = {'param_names': ['a', 'b', 'c'], 'param_mins': [0.0, 0.0, 0.0],
                       'param_maxs': [1.0, 1.0, 1.0], 'num_samples': num_samples,
                       'sample_type': 'saltelli'}
    manager._param_labels = lambda: ['a', 'b', 'c']
    manager._output_columns = lambda n: (['sum', 'only_a'][:n], [('data_item', 0)] * n)
    samples = np.asarray(manager.generate_samples())
    outputs = np.column_stack([samples[:, 0] + 2 * samples[:, 1] + 0 * samples[:, 2],
                               samples[:, 0]])
    return manager, samples, outputs


@pytest.mark.unit
def test_sobol_convergence_has_a_row_per_prefix_and_ends_at_the_runs_own_indices(tmp_path):
    from libcuflynx.sensitivity_analysis.sobolSA import SOBOL_CONVERGENCE_FILE
    manager, samples, outputs = _bare_sobol(tmp_path, num_samples=64)
    S1, ST, _ = manager.sobol_index(outputs)
    path = manager.save_sobol_convergence(S1, ST)
    assert os.path.basename(path) == SOBOL_CONVERGENCE_FILE
    with open(path) as f:
        convergence = json.load(f)
    assert convergence['n'] == [8, 16, 32, 64]
    assert sorted(convergence['outputs']) == ['only_a', 'sum']
    for i, output in enumerate(['sum', 'only_a']):
        for index, full in (('S1', S1), ('ST', ST)):
            table = convergence['outputs'][output][index]
            assert sorted(table) == ['a', 'b', 'c']
            for j, param in enumerate(['a', 'b', 'c']):
                rows = table[param]
                assert len(rows) == 4 and all(len(row) == 2 for row in rows)
                assert rows[-1][0] == pytest.approx(full[i, j]), 'the last row is the run'
                assert rows[-1][1] == pytest.approx(manager.sobol_confidence[index][i, j])
    # only_a depends on a alone, and the estimates say so at every prefix from 16 up
    only_a = convergence['outputs']['only_a']['ST']
    assert all(row[0] == pytest.approx(1.0, abs=0.2) for row in only_a['a'][1:])
    assert all(abs(row[0]) < 0.05 for row in only_a['c'])
    # a prefix is the analysis of its own rows: the same call gives the same numbers
    from SALib.analyze import sobol
    from libcuflynx.sensitivity_analysis.sobolSA import CONVERGENCE_SEED
    block = len(samples) // 64
    alone = sobol.analyze(manager.problem, outputs[:16 * block, 0], seed=CONVERGENCE_SEED)
    assert convergence['outputs']['sum']['S1']['b'][1] == pytest.approx(
        [alone['S1'][1], alone['S1_conf'][1]])


@pytest.mark.unit
def test_sobol_convergence_of_a_small_run_is_the_run_alone(tmp_path):
    manager, _, outputs = _bare_sobol(tmp_path, num_samples=6)
    S1, ST, _ = manager.sobol_index(outputs)
    assert manager.sobol_convergence(S1, ST)['n'] == [6]


@pytest.mark.unit
def test_the_sobol_design_and_an_empty_status_are_written_before_any_run(tmp_path):
    from libcuflynx.sensitivity_analysis.sobolSA import (SAMPLE_NOT_RUN, SOBOL_DESIGN_FILE,
                                                        SOBOL_DESIGN_PARAMS_FILE,
                                                        SOBOL_SAMPLE_STATUS_FILE)
    manager, samples, _ = _bare_sobol(tmp_path, num_samples=4)
    manager.progress = ProgressFile(str(tmp_path), 'sobol')
    manager._start_sample_status(samples)
    assert np.load(os.path.join(str(tmp_path), SOBOL_DESIGN_FILE)) == pytest.approx(samples)
    status = np.load(os.path.join(str(tmp_path), SOBOL_SAMPLE_STATUS_FILE))
    assert status.dtype == np.int8 and status.shape == (len(samples),)
    assert np.all(status == SAMPLE_NOT_RUN)
    with open(os.path.join(str(tmp_path), SOBOL_DESIGN_PARAMS_FILE)) as f:
        assert json.load(f) == {'names': ['a', 'b', 'c'], 'bounds': [[0.0, 1.0]] * 3,
                                'num_samples': 4, 'sample_type': 'saltelli',
                                'rows': len(samples)}


# ---- profile likelihood ----------------------------------------------------------------------


@pytest.mark.unit
def test_a_profile_likelihood_has_a_phase_per_parameter(tmp_path, monkeypatch):
    from test_profile_likelihood import _FakeOptimiser, _profile_for, _stub_param_id
    pid = _stub_param_id(tmp_path)
    seen = []

    def optimum(held):
        seen.append(read_progress(os.path.join(str(tmp_path), 'profile_likelihood')))
        idx, value = held_entry(held)
        if idx == 2 and value < 5.0:
            return [np.nan, np.nan]  # a failed point: its cost is not finite
        return [value, value + 100.0]

    def held_entry(held):
        # the one held entry, whether the object holds one entry or a list of them
        if hasattr(held, 'indices'):
            return held.indices[0], held.values[0]
        return held.idx, held.value

    monkeypatch.setattr(pid, 'make_optimiser', _FakeOptimiser(optimum=optimum), raising=False)
    profile = _profile_for(pid)
    profile.run()
    if profile.rank != 0:
        return

    document = read_progress(os.path.join(str(tmp_path), 'profile_likelihood'))
    assert document['kind'] == 'profile_likelihood' and document['state'] == 'done'
    assert [(p['name'], p['param_index'], p['state'], p['done'], p['total'], p['failed'])
            for p in document['phases']] == [
        ('g/A+g/B', 0, 'done', 3, 3, 0), ('m/C+m/D', 1, 'done', 3, 3, 0),
        ('p/E', 2, 'done', 3, 3, 1)]
    # while it runs: every phase is listed from the start, and each point is written as it ends
    assert len(seen) == 9 and all(d['state'] == 'running' for d in seen)
    assert [sum(p['done'] for p in d['phases']) for d in seen] == list(range(9))
    assert [p['state'] for p in seen[4]['phases']] == ['done', 'running', 'queued']
    # done is written after the results
    assert os.path.exists(os.path.join(str(tmp_path), 'profile_likelihood', 'summary.json'))


# ---- emulator training -----------------------------------------------------------------------


class _Metric:
    name = 'r2'


class _Result:
    def __init__(self, index, score):
        self.id, self.test_metrics = index, {_Metric(): (score, 0.01)}


class GoodModel:
    pass


class BrokenModel:
    pass


class LastModel:
    pass


@pytest.mark.unit
def test_the_training_phase_follows_each_model_autoemulate_compares(tmp_path, monkeypatch):
    """A stand-in for AutoEmulate with its two hooks: compare() over self.models, and
    add_result() for each model that fitted (BrokenModel never gets one)."""
    from libcuflynx.emulators.emulator_trainer import ModelProgress, reporting_autoemulate
    seen = []

    class FakeAutoEmulate:
        def __init__(self):
            self.models, self.results = [GoodModel, BrokenModel, LastModel], []
            self.compare()

        def compare(self):
            for index, model in enumerate(self.models):
                seen.append(read_progress(tmp_path)['phases'][0])
                if model is not BrokenModel:
                    self.add_result(_Result(index, 0.9 - 0.1 * index))

        def add_result(self, result):
            self.results.append(result)

    progress = ProgressFile(str(tmp_path), 'emulation', min_interval=5)
    model_progress = ModelProgress(progress)
    waiting = read_progress(tmp_path)['phases'][0]
    assert (waiting['state'], waiting['total'], waiting['models']) == ('running', 0, [])
    assert reporting_autoemulate(FakeAutoEmulate, None) is FakeAutoEmulate

    emulation = reporting_autoemulate(FakeAutoEmulate, model_progress)()
    assert len(emulation.results) == 2, 'the comparison itself is unchanged'
    states = [[model['state'] for model in phase['models']] for phase in seen]
    assert states == [['running', 'queued', 'queued'], ['done', 'running', 'queued'],
                      ['done', 'running', 'queued']]
    model_progress.close('LastModel', 0.7, 'r2')
    progress.finish()

    phase = read_progress(tmp_path)['phases'][0]
    assert (phase['name'], phase['state'], phase['done'], phase['total'], phase['failed'],
            phase['metric']) == ('training', 'done', 3, 3, 1, 'r2')
    good, broken, last = phase['models']
    assert (good['name'], good['state'], good['score']) == ('GoodModel', 'done', 0.9)
    assert (broken['name'], broken['state'], broken['score'], broken['duration_s']) == (
        'BrokenModel', 'failed', None, None)
    assert (last['name'], last['state'], last['score']) == ('LastModel', 'done',
                                                            pytest.approx(0.7))
    assert good['duration_s'] >= 0 and last['duration_s'] >= 0


@pytest.mark.unit
def test_a_fit_that_reports_no_model_is_one_entry(tmp_path):
    from libcuflynx.emulators.emulator_trainer import ModelProgress
    progress = ProgressFile(str(tmp_path), 'emulation', min_interval=0)
    ModelProgress(progress).close('two_phase_GaussianProcess', np.float64(0.93), 'r2')
    phase = read_progress(tmp_path)['phases'][0]
    assert (phase['done'], phase['total'], phase['failed'], phase['state']) == (1, 1, 0, 'done')
    assert phase['models'][0]['name'] == 'two_phase_GaussianProcess'
    assert phase['models'][0]['score'] == pytest.approx(0.93)
