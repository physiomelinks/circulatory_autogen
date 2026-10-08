"""The CI shards in tests/ci_shards.json, and the --ci-shard selection in conftest.py.

A shard that names a test which no longer exists, or a workflow matrix that leaves a group out,
would stop tests running in CI without anything failing. These checks are what fails instead.
"""
import ast
import os
import re

import pytest
import yaml

from conftest import ci_shard_entry_matches, load_ci_shards, select_ci_shard

pytestmark = pytest.mark.unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'tests.yml')
SHARDS = load_ci_shards()


def _test_functions(path):
    with open(os.path.join(ROOT, path)) as f:
        tree = ast.parse(f.read())
    return {node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize('suite', sorted(SHARDS))
def test_every_entry_names_an_existing_test(suite):
    for group, entries in SHARDS[suite].items():
        assert entries, f'{suite}:{group} is empty'
        for entry in entries:
            path, _, function = entry.partition('::')
            assert os.path.isfile(os.path.join(ROOT, path)), f'{suite}:{group}: no file {path}'
            if function:
                assert function in _test_functions(path), \
                    f'{suite}:{group}: {path} defines no {function}'


@pytest.mark.parametrize('suite', sorted(SHARDS))
def test_no_test_is_in_two_groups(suite):
    groups = SHARDS[suite]
    assert 'rest' not in groups, "'rest' is every test no group names; it cannot be a group"
    owner = {}
    for group, entries in groups.items():
        for entry in entries:
            path = entry.partition('::')[0]
            for other in (entry, path):
                claimed = owner.get(other)
                assert claimed in (None, group), f'{entry} is in {claimed} and {group}'
            owner[entry] = group
    whole_files = {e for e in owner if '::' not in e}
    for entry, group in owner.items():
        path = entry.partition('::')[0]
        if '::' in entry and path in whole_files:
            assert owner[path] == group, f'{entry} ({group}) is inside {path} ({owner[path]})'


def _sharded_jobs():
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)['jobs']
    found = []
    for name, job in jobs.items():
        commands = '\n'.join(step.get('run', '') for step in job.get('steps', []))
        match = re.search(r'--ci-shard "([\w-]+):\$\{\{ matrix\.shard \}\}"', commands)
        if match:
            found.append((name, match.group(1), job['strategy']['matrix']['shard']))
    return found


def test_the_workflow_shards_jobs():
    assert {suite for _, suite, _ in _sharded_jobs()} == set(SHARDS)


@pytest.mark.parametrize('job, suite, matrix', _sharded_jobs(), ids=lambda v: str(v))
def test_each_sharded_job_runs_every_group_once(job, suite, matrix):
    groups = [g for shard in matrix for g in shard.split(',')]
    assert sorted(groups) == sorted(list(SHARDS[suite]) + ['rest']), \
        f'{job} runs {groups}; suite {suite} is {sorted(SHARDS[suite])} plus rest'


# --- select_ci_shard -----------------------------------------------------------------------

class _Item:
    def __init__(self, nodeid):
        self.nodeid = nodeid


class _Config:
    def __init__(self, spec):
        self.spec = spec
        self.deselected = []
        config = self

        class _Hook:
            @staticmethod
            def pytest_deselected(items):
                config.deselected += items
        self.hook = _Hook()

    def getoption(self, name):
        assert name == '--ci-shard'
        return self.spec


EXAMPLE = {'s': {'a': ['tests/test_a.py'], 'b': ['tests/test_b.py::test_one']}}
NODEIDS = ['tests/test_a.py::test_x', 'tests/test_b.py::test_one[1]', 'tests/test_b.py::test_one',
           'tests/test_b.py::test_one_more', 'tests/test_c.py::test_y']


def _select(spec):
    config, items = _Config(spec), [_Item(n) for n in NODEIDS]
    select_ci_shard(config, items, shards=EXAMPLE)
    assert len(items) + len(config.deselected) == len(NODEIDS)
    return [i.nodeid for i in items]


def test_a_group_keeps_its_tests_and_rest_keeps_the_others():
    assert _select('s:a') == ['tests/test_a.py::test_x']
    assert _select('s:b') == ['tests/test_b.py::test_one[1]', 'tests/test_b.py::test_one']
    assert _select('s:rest') == ['tests/test_b.py::test_one_more', 'tests/test_c.py::test_y']
    assert _select('s:a,b,rest') == NODEIDS


def test_without_the_option_nothing_is_deselected():
    assert _select(None) == NODEIDS


@pytest.mark.parametrize('spec', ['s', 'nope:a', 's:nope', 's:'])
def test_an_unknown_shard_is_a_usage_error(spec):
    with pytest.raises(pytest.UsageError):
        _select(spec)


def test_a_function_entry_does_not_match_a_longer_name():
    assert ci_shard_entry_matches('tests/t.py::test_one', 'tests/t.py::test_one[x-1]')
    assert not ci_shard_entry_matches('tests/t.py::test_one', 'tests/t.py::test_one_more')
    assert ci_shard_entry_matches('tests/t.py', 'tests/t.py::Cls::test_m')
    assert not ci_shard_entry_matches('tests/t.py', 'tests/t.py.bak::test_m')
