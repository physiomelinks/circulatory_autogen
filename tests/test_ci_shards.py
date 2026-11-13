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


def _github_yaml_files():
    found = []
    for folder, _, names in os.walk(os.path.join(ROOT, '.github')):
        found += [os.path.join(folder, n) for n in names if n.endswith(('.yml', '.yaml'))]
    return sorted(found)


def test_no_workflow_uses_ubuntu_latest():
    """The PyPI libcellml wheel needs libxml2.so.2, which ubuntu-latest (26.04) does not have."""
    assert _github_yaml_files()
    for path in _github_yaml_files():
        with open(path) as f:
            for number, line in enumerate(f, 1):
                code = line.split('#')[0]
                assert 'ubuntu-latest' not in code, f'{path}:{number} uses ubuntu-latest'


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


# --- which files the param-id jobs run ------------------------------------------------------

@pytest.mark.parametrize('job', ['test-param-id-mpi', 'test-param-id-serial'])
def test_the_param_id_jobs_run_the_profile_likelihood_tests(job):
    """Its integration tests are `mpi`-marked, so the `test` jobs (`not mpi`) never run them."""
    with open(WORKFLOW) as f:
        steps = yaml.safe_load(f)['jobs'][job]['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert 'tests/test_profile_likelihood.py' in commands


# --- every mpi-marked test runs somewhere ---------------------------------------------------

def _pytest_commands():
    """(job, pytest command) for every pytest run in tests.yml, continuation lines joined."""
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)['jobs']
    found = []
    for name, job in jobs.items():
        if job.get('if') is False:
            continue
        for step in job.get('steps', []):
            script = re.sub(r'[\\`]\n\s*', ' ', step.get('run', ''))
            found += [(name, line) for line in script.splitlines() if 'pytest' in line]
    return found


def _selects_mpi(command):
    """Whether a pytest command keeps `mpi`-marked tests: no -m, or a -m that does not say not mpi."""
    match = re.search(r'-m\s+"([^"]*)"', command)
    return match is None or not re.search(r'\bnot\s+mpi\b', match.group(1))


def _mpi_test_files():
    files = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'tests'))):
        if name.startswith('test_') and name.endswith('.py'):
            with open(os.path.join(ROOT, 'tests', name)) as f:
                if re.search(r'pytest\.mark\.mpi\b', f.read()):
                    files.append(f'tests/{name}')
    return files


def _covers(command, path):
    if path in command:
        return True
    ignored = re.findall(r'--ignore=(\S+)', command)
    return bool(re.search(r'\stests/?(\s|$)', command)) and path not in ignored


def test_every_mpi_test_runs_in_some_ci_job():
    """The `test` jobs deselect `mpi`; a file of `mpi` tests no job names would never run.

    That is how ~40 tests (prediction features, emulator training, the working-directory and
    plot checks) went unrun until test-mpi-rest.
    """
    files = _mpi_test_files()
    assert 'tests/test_param_id.py' in files
    commands = [(job, c) for job, c in _pytest_commands() if _selects_mpi(c)]
    unrun = [f for f in files if not any(_covers(c, f) for _, c in commands)]
    assert not unrun, f'mpi-marked tests that no CI job runs: {unrun}'


def test_the_main_jobs_deselect_mpi_and_mpi_rest_selects_it():
    by_job = {}
    for job, command in _pytest_commands():
        by_job.setdefault(job, []).append(command)
    assert not any(_selects_mpi(c) for c in by_job['test'])
    assert all(_selects_mpi(c) for c in by_job['test-mpi-rest'])


def test_every_ci_job_with_pytest_uploads_or_checks_its_junit():
    """Skip counts can only be audited from a junit file someone can download."""
    with open(WORKFLOW) as f:
        jobs = yaml.safe_load(f)['jobs']
    for job in ('test', 'test-mpi-rest', 'test-min-deps', 'test-windows'):
        uses = [step.get('uses', '') for step in jobs[job]['steps']]
        assert any(u.startswith('actions/upload-artifact') for u in uses), job


def test_the_min_deps_pins_are_the_pyproject_floors():
    """test-min-deps tests the oldest versions a user can get; a raised floor must move it."""
    from _pyproject import load_pyproject
    floors = {}
    for spec in load_pyproject()['project']['dependencies'] + \
            load_pyproject()['project']['optional-dependencies']['casadi']:
        match = re.match(r'([A-Za-z0-9_.-]+)>=([0-9.]+)', spec)
        if match:
            floors[match.group(1).lower()] = match.group(2)
    with open(os.path.join(ROOT, 'tests', 'min-deps-constraints.txt')) as f:
        pins = dict(line.strip().split('==') for line in f if '==' in line)
    assert pins == {name: floors[name] for name in pins}
    assert set(pins) == {'casadi', 'myokit', 'libcellml'}
