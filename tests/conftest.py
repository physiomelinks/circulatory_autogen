"""
Pytest configuration file for circulatory_autogen tests.

This file sets up the test environment to work with OpenCOR's Python shell.
It provides fixtures for test data, configuration, and deterministic randomness.
"""
import os
import sys
import yaml
import tempfile
import shutil
import pytest
import numpy as np
import random
import fcntl
import io
import contextlib
import time
from mpi4py import MPI

# Ensure src is on sys.path before importing project modules
_TEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
_SRC_DIR = os.path.join(_TEST_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from scripts.script_generate_with_new_architecture import generate_with_new_architecture

# Store pytest config for hooks that need plugin access (xdist reports lack config)
_PYTEST_CONFIG = None
_AUTOGEN_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", ".pytest_autogen_results")
_SESSION_START = None
_PARAM_ID_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", ".pytest_param_id_results")
_AUTOGEN_CONFIGS = [
    {"file_prefix": "3compartment", "input_param_file": "3compartment_parameters.csv", "model_type": "cellml_only", "solver": "CVODE"},
    {"file_prefix": "simple_physiological", "input_param_file": "simple_physiological_parameters.csv", "model_type": "cellml_only", "solver": "CVODE"},
    {"file_prefix": "test_fft", "input_param_file": "test_fft_parameters.csv", "model_type": "cellml_only", "solver": "CVODE"},
]
_LOCK_FILE = os.path.realpath(os.path.join(_TEST_ROOT, ".pytest_param_id_lock"))
_PARAM_ID_TRIGGERS = ("test_param_id", "compare_optimisers", "test_sensitivity_analysis")


def _mpi_rank_size():
    """Safely return (rank, size) even if MPI misbehaves."""
    try:
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        return 0, 1


def _silence_non_root_output():
    """
    Avoid duplicate pytest banners/progress on non-root ranks.
    Terminalreporter is unregistered in pytest_configure; keep stdout/stderr
    intact so worker ranks can emit per-test diagnostics when needed.
    """
    rank, size = _mpi_rank_size()
    if size > 1 and rank != 0:
        return


_silence_non_root_output()


def _ansi(code):
    return f"\033[{code}m"


_BLUE = _ansi("34")
_GREEN = _ansi("32")
_RED = _ansi("31")
_RESET = _ansi("0")


def _load_base_inputs(user_inputs_dir):
    """Load base user inputs and strip overrides so defaults are used."""
    user_inputs_path = os.path.join(user_inputs_dir, 'user_inputs.yaml')

    if not os.path.exists(user_inputs_path):
        pytest.skip(f"User inputs file not found: {user_inputs_path}")

    with open(user_inputs_path, 'r') as file:
        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

    # Remove user_input entries so they aren't passed to the generation script,
    # this ensures the default dirs are used
    for key in ['user_inputs_path_override', 'resources_dir', 'generated_models_dir', 'param_id_output_dir']:
        if key in inp_data_dict:
            del inp_data_dict[key]

    return inp_data_dict


def _get_assigned_autogen_rank(request):
    for key, val in getattr(request.node, "user_properties", []):
        if key == "autogen_rank":
            return val
    return 0


def pytest_configure(config):
    """
    Configure pytest to work with OpenCOR Python environment.
    This ensures the src directory is in the Python path for imports.
    """
    global _PYTEST_CONFIG
    _PYTEST_CONFIG = config

    # Silence all terminal output on non-root MPI ranks to avoid repeated pytest
    # banners/progress. Failures still bubble up via process exit codes.
    rank, size = _mpi_rank_size()
    if size > 1 and rank != 0:
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if tr:
            config.pluginmanager.unregister(tr)
        # Prevent later re-registration and suppress header/footer hooks
        config.pluginmanager.set_blocked("terminalreporter")

    # Add src directory to path if not already there
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    src_dir = os.path.join(root_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Ensure pytest-xdist group marker is registered (also in pyproject for strict markers)
    config.addinivalue_line("markers", "xdist_group(name): serialize a group of tests under pytest-xdist")
    config.addinivalue_line("markers", "autogen_rank(idx): rank assigned to run an autogeneration test")


def pytest_sessionstart(session):
    """Record session start time for aggregated summaries."""
    global _SESSION_START
    _SESSION_START = time.time()
    # Reset results file on rank 0
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0 and os.path.exists(_AUTOGEN_RESULTS_FILE):
        try:
            os.remove(_AUTOGEN_RESULTS_FILE)
        except OSError:
            pass
    if rank == 0 and os.path.exists(_PARAM_ID_RESULTS_FILE):
        try:
            os.remove(_PARAM_ID_RESULTS_FILE)
        except OSError:
            pass


def _has_serial_marker(item):
    """Return True if item should be serialized across ranks/workers."""
    nodeid = item.nodeid
    return any(trigger in nodeid for trigger in _PARAM_ID_TRIGGERS)


def _is_param_id_related_nodeid(nodeid: str) -> bool:
    return any(trigger in nodeid for trigger in _PARAM_ID_TRIGGERS)


def pytest_runtest_setup(item):
    """
    Serialize param_id/comparison/SA tests so only one runs at a time across workers/ranks.
    Uses a filesystem lock to coordinate even when xdist or MPI are involved.
    """
    if not _has_serial_marker(item):
        return
    # Under MPI we need all ranks to enter the test concurrently; skip the lock when size>1
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        return
    fh = open(_LOCK_FILE, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    item._param_id_lock_fh = fh
    if comm.Get_rank() == 0:
        print(f"[LOCK] Acquired for {item.nodeid}")


def pytest_runtest_teardown(item, nextitem):
    """Release the serialization lock."""
    fh = getattr(item, "_param_id_lock_fh", None)
    # If MPI size > 1 we skipped the lock entirely
    if fh:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print(f"[LOCK] Released for {item.nodeid}")
        item._param_id_lock_fh = None

def pytest_runtest_logreport(report):
    """
    Ensure captured stdout from test_compare_optimisers is shown even when tests pass.
    Some runners/plugins keep capture enabled; this surfaces the comparison output.
    """
    if report.when != "call":
        return

    # Collect autogen outcomes for cross-rank aggregation
    if "autogen_task" in getattr(report, "keywords", {}):
        assigned_rank = 0
        for name, value in getattr(report, "user_properties", []):
            if name == "autogen_rank":
                assigned_rank = value
                break
        # capture minimal failure info
        msg = None
        if report.failed:
            # prefer longrepr text if available
            try:
                msg = str(report.longrepr)
            except Exception:
                msg = None
        # Persist to shared file for rank0 summary without MPI gather.
        # Do not emit immediate PASSED/FAILED lines here: we want only the
        # "[AUTOGEN] Starting ..." line before the captured test output.
        try:
            with open(_AUTOGEN_RESULTS_FILE, "a") as f:
                if msg:
                    f.write(f"{report.nodeid}|{report.outcome}|{assigned_rank}|call|{msg}\n")
                else:
                    f.write(f"{report.nodeid}|{report.outcome}|{assigned_rank}\n")
        except Exception:
            pass

    # Collect param_id/comparison/sensitivity outcomes for summary aggregation.
    # These tests run under MPI; to avoid duplicate spam, only rank 0 records.
    try:
        nodeid = getattr(report, "nodeid", "")
        if nodeid and _is_param_id_related_nodeid(nodeid) and "autogen_task" not in getattr(report, "keywords", {}):
            rank, size = _mpi_rank_size()
            if rank == 0:
                status = report.outcome
                # capture a small failure hint if available
                msg = None
                if getattr(report, "failed", False):
                    try:
                        msg = str(getattr(report, "longrepr", "")).strip()
                    except Exception:
                        msg = None
                try:
                    with open(_PARAM_ID_RESULTS_FILE, "a") as f:
                        if msg:
                            f.write(f"{nodeid}|{status}|{rank}|call|{msg}\n")
                        else:
                            f.write(f"{nodeid}|{status}|{rank}\n")
                except OSError:
                    pass
    except Exception:
        # Never fail the run due to reporting
        pass

    if report.passed and "test_compare_optimisers" in report.nodeid:
        # report in xdist may not carry config; use stored config instead
        global _PYTEST_CONFIG
        if _PYTEST_CONFIG is None:
            return
        tr = _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")
        if not tr:
            return

        # Debug marker to confirm hook execution
        tr.write_line("\n=== Comparison hook executed ===", yellow=True)

        # Prefer captured stdout if available
        cap = getattr(report, "capstdout", "")
        if cap:
            tr.write_line("\n=== Comparison output (captured) ===", yellow=True)
            tr.write(cap)

        # Always attempt to read the persisted comparison output file if present
        try:
            # temp_output_dir is unique per test instance; collect from report.user_properties if set
            out_file = None
            for name, value in getattr(report, "user_properties", []):
                if name == "comparison_output_file":
                    out_file = value
                    break
            if out_file and os.path.exists(out_file):
                with open(out_file, "r") as f:
                    contents = f.read()
                if contents:
                    tr.write_line("\n=== Comparison output (file) ===", yellow=True)
                    tr.write(contents)
        except Exception:
            # Do not fail the test if reading the file fails
            pass


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Attach per-phase reports to the item so fixtures can access the final outcome.
    Used by autogen_output_buffer to print a colored PASSED/FAILED status after
    emitting captured output (while keeping the log clean at test start).
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

@pytest.fixture(scope="session")
def project_root():
    """Fixture that returns the project root directory."""
    return os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture(scope="session")
def opencor_python_path():
    """
    Fixture that attempts to read the OpenCOR Python shell path.
    Returns None if the path cannot be determined.
    """
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    opencor_path_file = os.path.join(root_dir, 'user_run_files', 'opencor_pythonshell_path.sh')
    
    if os.path.exists(opencor_path_file):
        try:
            # Read the shell script and extract the path
            with open(opencor_path_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#') and 'opencor_pythonshell_path=' in line:
                        # Extract the path value
                        path = line.split('opencor_pythonshell_path=', 1)[1].strip()
                        # Remove quotes if present
                        path = path.strip('"\'')
                        if os.path.exists(path):
                            return path
        except Exception as e:
            pytest.skip(f"Could not read OpenCOR Python path: {e}")
    
    return None


@pytest.fixture(scope="session")
def user_inputs_dir(project_root):
    """Fixture that returns the user inputs directory."""
    return os.path.join(project_root, 'user_run_files')


@pytest.fixture(scope="session")
def resources_dir(project_root):
    """Fixture that returns the resources directory."""
    return os.path.join(project_root, 'resources')


@pytest.fixture(scope="function")
def base_user_inputs(user_inputs_dir):
    """
    Fixture that loads and returns base user inputs configuration.
    Removes path overrides to use default directories.
    """
    return _load_base_inputs(user_inputs_dir)


@pytest.fixture(scope="function")
def temp_output_dir():
    """
    Fixture that creates a temporary directory for test outputs.
    Automatically cleans up after the test.
    """
    temp_dir = tempfile.mkdtemp(prefix='circulatory_autogen_test_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def set_random_seed():
    """
    Fixture that sets random seeds for deterministic tests.
    Applied automatically to all tests.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    yield
    # Reset seeds after test (though not strictly necessary)


@pytest.fixture(scope="function")
def test_model_configs():
    """
    Fixture that returns test model configurations for parametrized tests.
    Returns a list of tuples: (file_prefix, input_param_file, model_type, solver)
    """
    return [
        # CellML models
        ('ports_test', 'ports_test_parameters.csv', 'cellml_only', 'CVODE'),
        ('3compartment', '3compartment_parameters.csv', 'cellml_only', 'CVODE'),
        ('simple_physiological', 'simple_physiological_parameters.csv', 'cellml_only', 'CVODE'),
        ('parasympathetic_model', 'parasympathetic_model_parameters.csv', 'cellml_only', 'CVODE'),
        ('test_fft', 'test_fft_parameters.csv', 'cellml_only', 'CVODE'),
        ('neonatal', 'neonatal_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test2_closed_loop', 'generic_junction_test_closed_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('generic_junction_test2_open_loop', 'generic_junction_test_open_loop_parameters.csv', 'cellml_only', 'CVODE'),
        ('SN_simple', 'SN_simple_parameters.csv', 'cellml_only', 'CVODE'),
        ('physiological', 'physiological_parameters.csv', 'cellml_only', 'CVODE'),
        ('control_phys', 'control_phys_parameters.csv', 'cellml_only', 'CVODE'),
        # CPP models
        ('aortic_bif_1d', 'aortic_bif_1d_parameters.csv', 'cpp', 'RK4'),
    ]


def pytest_collection_modifyitems(items):
    """
    Ensure autogeneration tests run before param_id tests, which in turn run before others.
    This is useful when running the full suite so that generated assets exist before
    parameter ID tests execute.
    """
    import os
    autogen_items = [item for item in items if "test_autogeneration" in item.nodeid]
    os.environ["AUTOGEN_TOTAL"] = str(len(autogen_items))
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assigned_counts = {}

    # Guard: if using more MPI ranks than autogen tests, nothing useful can be
    # scheduled on many ranks and output ordering/progress becomes confusing.
    # Exit early with a clear message rather than silently wasting ranks.
    if autogen_items and size > len(autogen_items):
        msg = (
            f"Requested {size} MPI rank(s) (-n {size}) but only {len(autogen_items)} "
            f"autogeneration test(s) were collected. Please rerun with -n <= {len(autogen_items)}."
        )
        # Exit on all ranks so mpiexec terminates cleanly.
        pytest.exit(msg, returncode=2)

    def sort_key(item):
        nodeid = item.nodeid
        # Highest priority: autogeneration tests
        if "test_autogeneration" in nodeid:
            return (0, nodeid)
        # Next: param_id tests
        if "test_param_id" in nodeid:
            return (1, nodeid)
        # Next: optimiser comparison tests
        if "compare_optimisers" in nodeid:
            return (1, nodeid)
        # Ensure sensitivity analysis runs last
        if "test_sensitivity_analysis" in nodeid:
            return (3, nodeid)
        # Then everything else
        return (2, nodeid)

    items.sort(key=sort_key)

    # Mark param_id and comparison tests to run serially as one group under xdist
    for item in items:
        nodeid = item.nodeid
        if "test_param_id" in nodeid or "compare_optimisers" in nodeid:
            item.add_marker(pytest.mark.xdist_group("param_id_serial"))
            item.add_marker(pytest.mark.need_autogen)
        if "test_sensitivity_analysis" in nodeid:
            item.add_marker(pytest.mark.xdist_group("param_id_serial"))
            item.add_marker(pytest.mark.need_autogen)

    # Mark autogeneration tests so they record completion
    deselected = []
    for idx, item in enumerate(autogen_items):
        assigned_rank = idx % max(size, 1)
        assigned_counts[assigned_rank] = assigned_counts.get(assigned_rank, 0) + 1
        item.add_marker(pytest.mark.autogen_task)
        item.add_marker(pytest.mark.autogen_rank(assigned_rank))
        # Store for runtime lookup
        item.user_properties.append(("autogen_rank", assigned_rank))
        if size > 1 and assigned_rank != rank:
            deselected.append(item)

    if deselected:
        # Drop rank-mismatched autogen tests from this rank to avoid SKIPPED noise
        items[:] = [i for i in items if i not in deselected]

    if rank == 0 and autogen_items:
        summary = ", ".join(
            f"rank {r}: {assigned_counts.get(r, 0)}"
            for r in range(size)
        )
        print(f"[AUTOGEN] Distribution -> {summary}")


@pytest.fixture(scope="session")
def autogen_status_file(project_root):
    """Path to the shared autogeneration completion tracker file."""
    return os.path.join(project_root, ".pytest_autogen_status")


@pytest.fixture(scope="function", autouse=True)
def track_autogen_completion(request, autogen_status_file):
    """
    For autogeneration tests, append a line to the status file on completion.
    Param/compare tests can wait on this file to ensure all autogen tests finished.
    """
    yield
    if "autogen_task" in request.node.keywords:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        assigned_rank = _get_assigned_autogen_rank(request)
        if rank != assigned_rank:
            return
        # Append a line to indicate completion
        with open(autogen_status_file, "a") as f:
            f.write("done\n")


@pytest.fixture(scope="function", autouse=True)
def wait_for_autogen_if_needed(request, autogen_status_file):
    """
    Block param/compare tests until all autogeneration tests have completed.
    """
    if "need_autogen" not in request.node.keywords:
        return
    import time
    import os

    try:
        total = int(os.environ.get("AUTOGEN_TOTAL", "0"))
    except ValueError:
        total = 0

    if total == 0:
        # No autogen tests collected; nothing to wait for
        return

    waited = 0.0
    while True:
        if os.path.exists(autogen_status_file):
            try:
                with open(autogen_status_file, "r") as f:
                    count = sum(1 for _ in f)
                if count >= total:
                    return
            except OSError:
                pass
        time.sleep(0.1)
        waited += 0.1
        # Optional: log every ~10 seconds to show waiting
        rank, size = _mpi_rank_size()
        if size > 1 and rank != 0:
            continue
        if abs(waited - round(waited, 1)) < 1e-6 and int(waited) % 10 == 0 and waited > 0:
            print(f"Waiting for autogeneration to finish... ({waited:.1f}s)")


@pytest.fixture(scope="function", autouse=True)
def autogen_rank_gate(request):
    """
    Ensure each autogen test runs only on its assigned rank.
    """
    if "autogen_task" not in request.node.keywords:
        return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assigned_rank = _get_assigned_autogen_rank(request)

    # Announce start so concurrency across ranks is visible immediately.
    # Write to the original stdout to avoid pytest capture/xdist buffering, so the
    # "Starting" line appears before any later PASSED/FAILED reporting hooks.
    try:
        sys.__stdout__.write(f"\n{_BLUE}[AUTOGEN] Starting {request.node.nodeid} on rank {rank}{_RESET}\n")
        sys.__stdout__.flush()
    except Exception:
        # Fall back to regular print if sys.__stdout__ isn't usable
        print(f"\n{_BLUE}[AUTOGEN] Starting {request.node.nodeid} on rank {rank}{_RESET}", flush=True)

    if rank != assigned_rank:
        pytest.fail(f"Autogeneration test assigned to rank {assigned_rank} reached rank {rank}")
    # Assigned rank proceeds without extra barriers to allow parallel autogen


@pytest.fixture(scope="function", autouse=True)
def autogen_output_buffer(request):
    """
    Buffer stdout/stderr for autogen tests and emit once after completion,
    emitted on the executing rank. Avoid cross-rank gathers because autogen
    tests only run on their assigned rank.
    """
    if "autogen_task" not in request.node.keywords:
        # Still yield to satisfy pytest's expectation for generator fixtures.
        yield
        return
    comm = MPI.COMM_WORLD
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield
    captured = buf.getvalue()
    # Emit captured output first (no extra headers before it), then a short
    # completion line. This keeps the log clean: only the "[AUTOGEN] Starting ..."
    # line precedes the simulation/generation output.
    if captured:
        try:
            sys.__stdout__.write("\n" + captured.rstrip() + "\n")
            sys.__stdout__.flush()
        except Exception:
            print("\n" + captured.rstrip() + "\n", flush=True)
    # Print completion status (colored) after the captured output.
    rep = getattr(request.node, "rep_call", None)
    outcome = getattr(rep, "outcome", None) if rep is not None else None
    status = (outcome or "unknown").upper()
    color = _GREEN if outcome == "passed" else (_RED if outcome == "failed" else _RESET)
    try:
        sys.__stdout__.write(f"{color}[AUTOGEN] Completed {request.node.nodeid} {status} on rank {comm.Get_rank()}{_RESET}\n")
        sys.__stdout__.flush()
    except Exception:
        print(f"[AUTOGEN] Completed {request.node.nodeid} {status} on rank {comm.Get_rank()}", flush=True)


def pytest_report_teststatus(report, config):
    """
    Suppress the verbose word for autogen passes on rank 0 so pytest only shows
    the progress percent (no extra 'PASSED' from the root rank).
    """
    if report.when == "call" and "autogen_task" in getattr(report, "keywords", {}):
        try:
            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0
        if rank == 0 and report.outcome == "passed":
            return report.outcome, "", ""
    if report.when == "call" and _is_param_id_related_nodeid(getattr(report, "nodeid", "")):
        try:
            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0
        if rank == 0:
            nodeid = getattr(report, "nodeid", "")
            if report.outcome == "passed":
                return report.outcome, "P", f"PASSED[PARAM_ID] {nodeid} PASSED on rank {rank}"
            if report.outcome == "failed":
                # Keep status line single-line; full traceback is already in pytest output.
                full_msg = ""
                try:
                    full_msg = str(getattr(report, "longrepr", "")).strip()
                except Exception:
                    full_msg = ""
                first_line = full_msg.splitlines()[0] if full_msg else ""
                details = f" :: {first_line}" if first_line else ""
                return report.outcome, "F", f"FAILED[PARAM_ID] {nodeid} FAILED on rank {rank}{details}"
    if report.when == "call" and report.outcome == "failed":
        nodeid = getattr(report, "nodeid", "")
        return report.outcome, "F", f"FAILED[TEST] {nodeid}"
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Aggregate autogen results across ranks and print a global summary after
    pytest's own summary. Also print a concise summary for param_id/comparison/SA
    tests so it's obvious which ones passed/failed.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        return

    duration = time.time() - (_SESSION_START or time.time())

    # --- AUTOGEN summary (existing behavior, but don't return early if absent) ---
    autogen_results = []
    if os.path.exists(_AUTOGEN_RESULTS_FILE):
        try:
            with open(_AUTOGEN_RESULTS_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) == 3:
                        nodeid, status, r = parts
                        autogen_results.append({"nodeid": nodeid, "status": status, "rank": int(r)})
                    elif len(parts) >= 5:
                        # extended format with failure info (nodeid|status|rank|phase|message)
                        nodeid, status, r, phase, msg = parts[0], parts[1], parts[2], parts[3], "|".join(parts[4:])
                        autogen_results.append({"nodeid": nodeid, "status": status, "rank": int(r), "phase": phase, "message": msg})
        except OSError:
            pass

    if autogen_results:
        autogen_results.sort(key=lambda r: r["nodeid"])
        passed = sum(1 for r in autogen_results if r["status"] == "passed")
        failed = [r for r in autogen_results if r["status"] != "passed"]
        total = len(autogen_results)

        # Align pytest stats: add dummy passed for unmatched, and dummy failed for failures (no duplicates)
        stats_pass = terminalreporter.stats.setdefault("passed", [])
        stats_failed = terminalreporter.stats.setdefault("failed", [])
        existing_fail_nodeids = set(getattr(r, "nodeid", None) for r in stats_failed if hasattr(r, "nodeid"))
        existing_passes = len(stats_pass)
        additional_needed = max(0, total - len(failed) - existing_passes)
        if additional_needed:
            class _DummyReport:
                def __init__(self, nodeid):
                    self.nodeid = nodeid
            stats_pass.extend(_DummyReport(f"[autogen-aggregate-pass-{i}]") for i in range(additional_needed))
            terminalreporter._numcollected = max(getattr(terminalreporter, "_numcollected", 0), total)

        for fail in failed:
            if fail["nodeid"] in existing_fail_nodeids:
                continue
            class _DummyFailReport:
                def __init__(self, nodeid, longrepr=None):
                    self.nodeid = nodeid
                    self.longrepr = longrepr
                    self.when = "call"
                    self.outcome = "failed"
                    self.failed = True
                    self.passed = False
                    self.skipped = False
                def _get_verbose_word_with_markup(self, *args, **kwargs):
                    # minimal implementation expected by pytest terminal summary
                    return ("FAILED", {"red": True})
            longrepr = None
            if "message" in fail:
                longrepr = fail["message"]
            stats_failed.append(_DummyFailReport(fail["nodeid"], longrepr=longrepr))
            terminalreporter._numcollected = max(getattr(terminalreporter, "_numcollected", 0), total)

        terminalreporter.write_line("")
        for res in autogen_results:
            line = f"[AUTOGEN] {res['nodeid']} {res['status'].upper()} on rank {res['rank']}"
            if "phase" in res:
                line += f" ({res['phase']})"
            terminalreporter.write_line(line)
            if res.get("message"):
                terminalreporter.write_line(f"[AUTOGEN OUTPUT] {res['message']}")

        terminalreporter.write_line(
            f"[AUTOGEN] Summary: {passed}/{total} passed, {len(failed)} failed in {duration:.2f}s"
        )
        footer_line = f"{passed} passed, {len(failed)} failed in {duration:.2f}s across ranks"
        terminalreporter.write_sep("=", footer_line)

    # --- PARAM_ID summary ---
    param_results = []
    if os.path.exists(_PARAM_ID_RESULTS_FILE):
        try:
            with open(_PARAM_ID_RESULTS_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) == 3:
                        nodeid, status, r = parts
                        param_results.append({"nodeid": nodeid, "status": status, "rank": int(r)})
                    elif len(parts) >= 5:
                        nodeid, status, r, phase, msg = parts[0], parts[1], parts[2], parts[3], "|".join(parts[4:])
                        param_results.append({"nodeid": nodeid, "status": status, "rank": int(r), "phase": phase, "message": msg})
        except OSError:
            pass

    if param_results:
        param_results.sort(key=lambda r: r["nodeid"])
        passed = sum(1 for r in param_results if r["status"] == "passed")
        failed = [r for r in param_results if r["status"] != "passed"]
        total = len(param_results)

        terminalreporter.write_line("")
        for res in param_results:
            line = f"[PARAM_ID] {res['nodeid']} {res['status'].upper()} on rank {res['rank']}"
            if "phase" in res:
                line += f" ({res['phase']})"
            terminalreporter.write_line(line)
            if res.get("message"):
                # Keep it compact; full traceback is already in pytest output.
                first_line = res["message"].splitlines()[0] if res["message"].splitlines() else res["message"]
                terminalreporter.write_line(f"[PARAM_ID OUTPUT] {first_line[:300]}")

        terminalreporter.write_line(
            f"[PARAM_ID] Summary: {passed}/{total} passed, {len(failed)} failed in {duration:.2f}s"
        )
        footer_line = f"{passed} passed, {len(failed)} failed in {duration:.2f}s (param_id/comparison/SA)"
        terminalreporter.write_sep("=", footer_line)



def _needs_param_id_autogen(session_items):
    triggers = ("test_param_id", "compare_optimisers", "test_sensitivity_analysis")
    return any(any(trigger in item.nodeid for trigger in triggers) for item in session_items)


@pytest.fixture(scope="session", autouse=True)
def mpi_banner_and_prepare_autogen(request, user_inputs_dir):
    """
    Announce MPI rank usage once and pre-run minimal autogeneration for
    param_id/comparison/sensitivity tests on rank 0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"[MPI] Running tests with {size} rank(s)")
    comm.Barrier()

    if not _needs_param_id_autogen(request.session.items):
        return

    if rank == 0:
        base_inputs = _load_base_inputs(user_inputs_dir)
        for cfg in _AUTOGEN_CONFIGS:
            autogen_config = base_inputs.copy()
            autogen_config.update(cfg)
            generate_with_new_architecture(False, autogen_config)
    comm.Barrier()


@pytest.fixture(scope="function")
def minimal_param_id_config(base_user_inputs, resources_dir, temp_output_dir):
    """
    Fixture that returns a minimal parameter identification configuration.
    """
    config = base_user_inputs.copy()
    config.update({
        'model_type': 'cellml_only',
        'solver': 'CVODE',
        'param_id_method': 'genetic_algorithm',
        'pre_time': 1,
        'sim_time': 1,
        'dt': 0.01,
        'DEBUG': True,
        'do_mcmc': False,
        'plot_predictions': False,
        'solver_info': {
            'MaximumStep': 0.001,
            'MaximumNumberOfSteps': 5000,
        },
        'param_id_output_dir': temp_output_dir,
    })
    return config
