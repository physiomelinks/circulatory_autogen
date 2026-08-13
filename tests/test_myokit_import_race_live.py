"""The same race as ``test_myokit_import_race``, but run for real against real Myokit.

The mocked tests lock the retry *contract*. This one checks the retry actually clears the
*failure*, by doing what CI does: several processes importing Myokit at the same moment into a
HOME where its config directory does not yet exist.

It proves its own premise before it asserts anything. First it runs the plain ``import myokit``
that CA used to do and looks for the FileExistsError; if the race does not fire on this machine
(fast disk, few cores, a Myokit that has stopped racing) there is nothing to regress and the
test skips rather than passing vacuously.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

#: Enough processes to lose the race, few enough to stay quick.
N_PROCS = 6

#: Rounds to give the unfixed import before accepting that it is not going to lose here.
BARE_ATTEMPTS = 5

#: Rounds the fixed import has to survive, once the unfixed one has been shown to lose.
FIXED_ROUNDS = 3


def _interpreter():
    """A Python that can be respawned. OpenCOR's embedded shell leaves sys.executable empty."""
    return sys.executable or shutil.which('python3') or shutil.which('python')


_CHILD = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, {src!r})

    # mpiexec binds its ranks to a core, and a child inherits that mask -- which would leave
    # these processes taking turns on one CPU, where nothing races. Give them the machine back.
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass

    # Line the processes up. They start staggered -- by seconds, on a loaded machine -- and the
    # race lives only in the microseconds between Myokit's os.path.exists and its os.makedirs,
    # so a wall-clock offset is not enough: each one announces itself and then waits for the
    # rest, and they go together however long the slowest took to boot.
    gate, n_procs = sys.argv[2], int(sys.argv[3])
    open(os.path.join(gate, str(os.getpid())), 'w').close()
    deadline = time.time() + 120
    while len(os.listdir(gate)) < n_procs and time.time() < deadline:
        time.sleep(0.001)

    try:
        if sys.argv[1] == 'bare':
            import myokit  # what CA did before the fix
        else:
            from solver_wrappers.myokit_helper import (
                _import_myokit_tolerating_first_run_race as imp)
            imp()
        print(json.dumps({{'ok': True, 'err': ''}}))
    except BaseException as exc:
        print(json.dumps({{'ok': False, 'err': '{{}}: {{}}'.format(type(exc).__name__, exc)}}))
    """
)


def _import_concurrently(mode, home, n_procs=N_PROCS):
    """Import Myokit in ``n_procs`` processes at once, into ``home``. Returns their errors."""
    env = dict(os.environ, HOME=home, PYTHONPATH=SRC_DIR)
    # The suite itself runs under mpiexec, and a process started inside that job inherits the
    # launcher's variables and tries to join it -- which fails, because it was never launched.
    # These children are deliberately plain processes, so drop the job's fingerprints.
    for key in list(env):
        if key.startswith(('OMPI_', 'PMIX_', 'PMI_', 'ORTE_', 'OPAL_')):
            del env[key]
    # Keep the user site dir reachable: HOME is being redirected, and on some installs that is
    # where Myokit lives.
    env.setdefault('PYTHONUSERBASE', os.path.join(os.path.expanduser('~'), '.local'))
    env.pop('MYOKIT_DIR_USER', None)

    script = _CHILD.format(src=SRC_DIR)
    gate = os.path.join(home, 'gate')
    os.makedirs(gate, exist_ok=True)
    procs = [
        subprocess.Popen(
            [_interpreter(), '-c', script, mode, gate, str(n_procs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        for _ in range(n_procs)
    ]

    errors = []
    for proc in procs:
        out, err = proc.communicate(timeout=180)
        line = next((ln for ln in out.splitlines() if ln.startswith('{')), None)
        if line is None:
            # Died without reporting -- treat the stderr as the error, so a crash cannot be
            # mistaken for a clean import.
            errors.append(err.strip().splitlines()[-1] if err.strip() else 'no output')
            continue
        result = json.loads(line)
        if not result['ok']:
            errors.append(result['err'])
    return errors


def test_the_retry_clears_a_real_concurrent_first_import(tmp_path):
    """Concurrent first import: plain ``import myokit`` loses, CA's retrying import does not."""
    if _interpreter() is None:
        pytest.skip('no respawnable interpreter, so concurrent imports cannot be staged')

    # Pre-flight, uncontended: both imports must work at all in the child before a failure
    # under contention can be read as the race rather than as a broken child environment.
    preflight = tmp_path / 'home_preflight'
    preflight.mkdir()
    for mode in ('bare', 'fixed'):
        broken = _import_concurrently(mode, str(preflight), n_procs=1)
        if broken:
            pytest.skip(f'the child cannot import Myokit at all ({mode}: {broken[0]})')

    # Losing is a matter of scheduling -- on this machine a round of six loses about half the
    # time -- so give it several rounds before believing the race is not there to catch.
    raced, bare_errors = [], []
    for attempt in range(BARE_ATTEMPTS):
        bare_home = tmp_path / f'home_bare_{attempt}'
        bare_home.mkdir()
        bare_errors = _import_concurrently('bare', str(bare_home))
        raced = [e for e in bare_errors if 'FileExistsError' in e]
        if raced:
            break

    if not raced:
        pytest.skip(
            f'the first-import race did not fire in {BARE_ATTEMPTS} rounds of {N_PROCS} '
            f'(last round: {bare_errors or "no errors"}), so there is nothing to regress here'
        )

    # Now the same conditions, several times over, through CA's import. Surviving one round
    # could be the same luck that makes the bug look like flakiness; surviving every round is
    # the claim being made.
    for round_ in range(FIXED_ROUNDS):
        fixed_home = tmp_path / f'home_fixed_{round_}'
        fixed_home.mkdir()
        fixed_errors = _import_concurrently('fixed', str(fixed_home))
        assert fixed_errors == [], (
            f'{len(raced)}/{N_PROCS} processes lost the race with a plain import, and CA\'s '
            f'retrying import was meant to survive it -- but in round {round_ + 1} it failed '
            f'with: {fixed_errors}'
        )
