"""The released CUFLynx executable, driven against *this* checkout of libcuflynx.

CUFLynx ships a frozen desktop app with a libcuflynx bundled inside it. That bundled copy
is whatever was current when the app was released, so nothing in either repo's CI notices
when a change here breaks the app people actually downloaded -- CUFLynx tests its bundle,
and CA tests its source, and the pairing of the two is tested nowhere.

This closes that gap from CA's side: take the *released* binary, point it at the working
tree, and drive the five things a user does with it -- moving a slider, training an
emulator, sensitivity, calibration, and UQ evaluated on the emulator. A break here means a
merged PR would leave the current download unable to run this engine.

**The override is the whole premise, so it is asserted first.** Pointing Settings at a
checkout only works because ``ca_imports`` puts that directory's ``src`` at the front of
``sys.path`` and ``libcuflynx.*`` then resolves there rather than to the frozen copy. If
that ever stopped working the app would quietly fall back to its own bundled engine and
every test below would pass while testing nothing at all -- the failure mode this file
exists to prevent, arriving disguised as a green run. ``test_the_app_runs_this_checkout``
is not a nicety; it is what makes the rest evidence.

Skipped unless ``CUFLYNX_EXECUTABLE`` names the binary. In CI the workflow downloads and
caches it and sets ``CUFLYNX_EXECUTABLE_REQUIRED``, which turns the skip into a failure --
a suite that silently skips its only end-to-end coverage is indistinguishable from one
that passes it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# tests/executable/this_file.py -> parents[2] is the repo root. This test lives one level
# deeper than the rest of the suite on purpose: `tests/conftest.py` imports yaml, numpy,
# mpi4py and libcuflynx at module scope, and the CI job that runs this one deliberately
# installs nothing but pytest -- the whole point being that the app carries its own
# environment. Collected from `tests/`, that conftest is loaded first and the job dies
# with ModuleNotFoundError before a single test runs. The job passes --confcutdir so
# nothing above this directory is loaded.
ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "tests" / "test_inputs"
SRC = ROOT / "src"
MODEL = INPUTS / "Lotka_Volterra_forced.cellml"
OBS = INPUTS / "Lotka_Volterra_forced_obs_data.json"
PARAMS = INPUTS / "Lotka_Volterra_forced_params_for_id.csv"

BINARY = os.environ.get("CUFLYNX_EXECUTABLE", "").strip()
REQUIRED = os.environ.get("CUFLYNX_EXECUTABLE_REQUIRED", "").strip() not in ("", "0")
PORT = int(os.environ.get("CUFLYNX_EXECUTABLE_PORT", "8791"))
BASE = f"http://127.0.0.1:{PORT}"

# Lotka-Volterra rather than a circulatory model on purpose: two states, no stiffness and
# no compile worth speaking of, so the whole file is minutes. The point here is the
# *pairing* of a released binary with this source -- an expensive model would test the
# solver, which CA's own suite already does far more thoroughly.
if not BINARY:
    if REQUIRED:  # pragma: no cover - CI misconfiguration
        raise RuntimeError(
            "CUFLYNX_EXECUTABLE_REQUIRED is set but CUFLYNX_EXECUTABLE is empty: the "
            "executable end-to-end tests would have skipped silently, which reads as a "
            "pass. Check the download/cache step in the workflow."
        )
    pytestmark = pytest.mark.skip(
        reason="set CUFLYNX_EXECUTABLE to a released CUFLynx binary to run these"
    )


def _req(method: str, path: str, data=None, timeout: float = 60):
    """One JSON call, with the server's own explanation kept on failure.

    urllib raises HTTPError with the body unread, so an unhandled 422 reaches CI as a bare
    status line -- and these endpoints answer with the sentence that says what is wrong
    ("no emulator has been trained for this study", say). Losing that turns a two-second
    diagnosis into a rerun with extra logging.
    """
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise AssertionError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None


def _upload(path: str, field: str, file: Path, extra: dict) -> dict:
    boundary = "----cacuflynxsmoke"
    parts: list[bytes] = []
    for k, v in extra.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="{field}"; filename="{file.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(file.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        BASE + path,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _poll(kind: str, job_id: str, timeout: float) -> dict:
    """Wait for an analysis job, returning its final payload.

    Deadline rather than an iteration count: these runs are minutes and the interesting
    failure is a hang, which a bounded loop of sleeps reports as "still running" forever.
    """
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _req("GET", f"/api/{kind}/{job_id}/status")
        if last.get("state") in ("done", "error", "cancelled"):
            return last
        time.sleep(2)
    lines = "\n".join((last.get("lines") or [])[-25:])
    raise AssertionError(f"{kind} did not finish within {timeout:.0f}s. Tail:\n{lines}")


def _fail_with_log(kind: str, payload: dict) -> str:
    return (
        f"{kind} ended in state {payload.get('state')!r}. Tail:\n"
        + "\n".join((payload.get("lines") or [])[-25:])
    )


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    """The released binary, serving, pointed at this working tree.

    ``CIRCULATORY_AUTOGEN_SRC`` rather than a POST to /api/config after startup: the
    directory has to be in place *before* the app first imports libcuflynx, or the bundled
    copy is already in ``sys.modules`` and the checkout can never win. A stale settings
    file would otherwise decide which engine gets tested, so the config dir is a throwaway
    too.
    """
    config_dir = tmp_path_factory.mktemp("cuflynx_config")
    binary = Path(BINARY).resolve()
    assert binary.is_file(), f"CUFLYNX_EXECUTABLE does not exist: {binary}"
    assert os.access(binary, os.X_OK), f"not executable (chmod +x?): {binary}"
    for fixture in (MODEL, OBS, PARAMS):
        assert fixture.is_file(), f"fixture missing: {fixture}"

    env = dict(os.environ)
    env["CIRCULATORY_AUTOGEN_SRC"] = str(ROOT)
    env["CUFLYNX_CONFIG_DIR"] = str(config_dir)

    proc = subprocess.Popen(
        [str(binary), "--port", str(PORT), "--browser"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"the app exited during startup (code {proc.returncode})"
                )
            try:
                urllib.request.urlopen(BASE + "/api/health", timeout=3).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            raise AssertionError("the app never became healthy")

        conf = _req("POST", "/api/config", {
            "ca_dir": str(ROOT),
            # Pin the backend: analyses inherit the engine's solver, and leaving it at a
            # CA default of CVODE_opencor -- which is not installed and never will be
            # here -- makes the run depend on what was persisted.
            "generated_model_format": "cellml",
            "solver": "CVODE_myokit",
            "solver_info": {"dt": 0.01},
        })
        assert conf.get("ca_exists"), f"checkout not accepted as a CA dir: {conf!r}"
        yield conf
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged app
            proc.kill()


@pytest.fixture(scope="session")
def study(app):
    """Model, obs_data and params_for_id uploaded once, as a user would."""
    up = _upload("/api/models/upload", "file", MODEL, {})
    model_id = up["model_id"]
    _upload("/api/obs_data/upload", "file", OBS, {"model_id": model_id})
    _upload("/api/params_for_id/upload", "file", PARAMS, {"model_id": model_id})
    return model_id


@pytest.fixture(scope="session")
def bundled_app(tmp_path_factory):
    """The released binary with **no** CA configured, so it runs the engine it ships.

    The opposite arrangement to ``app``, and deliberately a second process: which engine is
    in use is decided before the first libcuflynx import, so it cannot be changed by a POST
    afterwards. Its own port, so the two can coexist in one session.

    No ``CIRCULATORY_AUTOGEN_SRC``, and a throwaway config dir -- a ``ca_dir`` saved by any
    previous run on this machine would silently put a checkout back in front of the bundle
    and this would test the wrong engine while looking like it passed.
    """
    config_dir = tmp_path_factory.mktemp("bundled_config")
    binary = Path(BINARY).resolve()
    assert binary.is_file(), f"CUFLYNX_EXECUTABLE does not exist: {binary}"

    port = PORT + 1
    base = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if k != "CIRCULATORY_AUTOGEN_SRC"}
    env["CUFLYNX_CONFIG_DIR"] = str(config_dir)

    proc = subprocess.Popen(
        [str(binary), "--port", str(port), "--browser"],
        cwd=str(tmp_path_factory.mktemp("bundled_cwd")), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"the app exited during startup (code {proc.returncode})")
            try:
                urllib.request.urlopen(base + "/api/health", timeout=3).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            raise AssertionError("the app never became healthy")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged app
            proc.kill()


def test_the_bundle_carries_a_working_engine(bundled_app):
    """With nothing configured, the app must still have an engine -- its own.

    ``ca_dir`` empty *and* ``ca_exists`` true is the pair that says so: an empty directory
    with a present engine can only mean the bundled package. Either alone is ambiguous --
    ``ca_exists`` is true for a configured checkout too, and an empty ``ca_dir`` on a source
    run just means nobody chose one.
    """
    body = json.loads(
        urllib.request.urlopen(bundled_app + "/api/config", timeout=60).read().decode())
    assert body.get("packaged") is True, (
        "not the released executable, so there is no bundle to check")
    assert body.get("ca_dir") == "", (
        f"a CA directory is configured ({body.get('ca_dir')!r}), so this is not testing the "
        f"bundled engine -- a settings file from an earlier run is the usual cause")
    assert body.get("ca_exists") is True, (
        "nothing configured and no engine found: the bundle is missing libcuflynx")


def test_the_bundled_engine_reports_its_version(tmp_path_factory):
    """Which libcuflynx is frozen in, asked of the bundle itself.

    CUFLynx's own CI checks the declared floor against the version pip *resolved* into a
    venv -- not against what PyInstaller actually collected, which is a different question
    and the one a user's download depends on. Until now the only thing that ever checked the
    bundle was somebody reading the dist-info by hand.

    Asked through the app's runner mode rather than over HTTP: ``/api/config`` reports
    ``ca_src`` as empty when no CA directory is configured, which is exactly the arrangement
    that puts the bundled engine in charge -- so the endpoint cannot answer this. Running a
    probe inside the bundle can, and it is the same mechanism the checkout test above uses.
    """
    work = tmp_path_factory.mktemp("bundled_version")
    probe = work / "probe.py"
    probe.write_text(
        "import importlib.metadata as md\n"
        "try:\n"
        "    print('BUNDLED:', md.version('libcuflynx'))\n"
        "except Exception as exc:\n"
        # A bundle whose engine has no metadata is itself the finding: nothing downstream
        # can then tell which version shipped.
        "    print('NOMETA:', exc)\n",
        encoding="utf-8",
    )
    cfg = work / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    out = subprocess.run(
        [BINARY, "--_cuflynx-run-analysis", str(probe), str(cfg)],
        capture_output=True, text=True, timeout=300,
    )
    combined = out.stdout + out.stderr
    assert "BUNDLED:" in combined, (
        f"the bundle could not report a libcuflynx version:\n{combined[-2000:]}")

    version = combined.split("BUNDLED:", 1)[1].split()[0]
    assert tuple(int(p) for p in version.split(".")[:3]) >= (0, 7, 0), (
        f"the bundle carries libcuflynx {version}, older than the vocabulary change in "
        f"0.7.0 that this app's obs_info reads depend on")


def test_the_app_runs_this_checkout_not_its_own_bundle(app, tmp_path_factory):
    """Everything else here is meaningless if this fails.

    The binary carries its own libcuflynx. If the checkout does not take precedence, the
    tests below exercise the *released* engine and pass no matter what this PR does to the
    source -- a green run that proves nothing. Ask the bundle which file it actually
    imports rather than trusting that pointing at a directory was enough.

    Asked through the app's runner mode, which is how the app runs its own analyses, so
    this is the same import machinery the tests below depend on and not a simulation of it.
    """
    info = _req("GET", "/api/config")
    assert info.get("packaged") is True, (
        "the app does not report itself as packaged, so this is not the released "
        "executable and the pairing is not being tested"
    )

    work = tmp_path_factory.mktemp("ca_override")
    probe = work / "probe.py"
    probe.write_text(
        "import sys, importlib\n"
        # Whether the bundled copy is already in sys.modules decides whether this check
        # can conclude anything at all: once it is imported, a sys.path entry cannot
        # displace it and the answer below would be the bundle's no matter what. Report
        # it, so the guard can say "I cannot tell" instead of quietly passing.
        "print('PREIMPORTED:', 'libcuflynx' in sys.modules)\n"
        # The app's own mechanism: ca_imports.ensure_ca_path() puts the configured
        # directory's src at the front before the first CA import.
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "m = importlib.import_module('libcuflynx.parsers.PrimitiveParsers')\n"
        "print('RESOLVED:', m.__file__)\n",
        encoding="utf-8",
    )
    cfg = work / "cfg.json"  # runner mode expects a config path as argv[2]
    cfg.write_text("{}", encoding="utf-8")
    out = subprocess.run(
        [BINARY, "--_cuflynx-run-analysis", str(probe), str(cfg)],
        capture_output=True, text=True, timeout=300,
    )
    combined = out.stdout + out.stderr
    assert "PREIMPORTED: False" in combined, (
        "the bundle imported libcuflynx before the probe ran, so nothing this check does "
        "could displace it and its answer would be meaningless. The result below is not "
        f"evidence either way -- treat this as inconclusive, not as a pass.\n{combined[-2000:]}"
    )
    assert f"RESOLVED: {SRC}" in combined, (
        "the frozen app resolved libcuflynx to its own bundled copy rather than to this "
        f"checkout, so these tests would prove nothing about this source.\n{combined[-2000:]}"
    )


def test_param_sliding_runs_the_model(study):
    """What dragging a slider does: change a constant, re-simulate, get a trace back.

    The live tier, not the analysis tier -- it runs *in* the app's own process against
    this checkout's solver_wrappers, and it is the one interaction every user performs.
    """
    base = _req("POST", "/api/simulate",
                {"model_id": study, "params": {}, "sim_time": 5.0}, timeout=300)
    assert "detail" not in base, f"baseline simulation failed: {base.get('detail')}"
    assert len(base["time"]) > 1 and base["outputs"], f"no usable trace: {base.keys()}"

    moved = _req("POST", "/api/simulate",
                 {"model_id": study,
                  "params": {"Lotka_Volterra_module/alpha": 2.5},
                  "sim_time": 5.0}, timeout=300)
    assert "detail" not in moved, f"simulation after the slider move failed: {moved.get('detail')}"

    key = next(iter(base["outputs"]))
    assert moved["outputs"][key] != base["outputs"][key], (
        f"moving alpha changed nothing in {key}: the parameter never reached the solver, "
        f"which is the failure a slider shows as a dead control"
    )


@pytest.fixture(scope="session")
def emulator(study):
    """Train a surrogate -- and hand it to whatever needs one trained.

    A fixture rather than test ordering: the UQ-on-the-emulator test genuinely depends on
    this having happened, and depending on declaration order makes that invisible and
    breaks under -k or -p xdist.
    """
    started = _req("POST", "/api/emulator/train", {
        "model_id": study,
        # Far below CA's default of 128. This asks whether the emulator can be trained and
        # then evaluated by the other analyses, not whether it is accurate; a real
        # training run would dominate the job.
        "settings": {"num_train_samples": 16, "sample_type": "sobol",
                     "random_seed": 0, "n_iter": 2, "n_splits": 2, "min_r2": -1e9},
    })
    done = _poll("emulator", started["job_id"], 900)
    assert done["state"] == "done", _fail_with_log("emulator training", done)
    return done


def test_emulator_training(emulator):
    """It has to leave a bundle behind, not merely exit cleanly."""
    meta = emulator.get("metadata") or {}
    assert meta, (
        "training finished with no metadata, so nothing was written for the other "
        "analyses to load"
    )


def test_sensitivity_analysis(study):
    """Local sensitivity about the current point -- the cheapest real analysis run."""
    started = _req("POST", "/api/sensitivity/run", {
        "model_id": study,
        "settings": {"method": "local", "gradient_method": "FD", "nominal": "current",
                     "rel_step": 0.05, "dt": 0.01, "num_cores": 1},
    })
    done = _poll("sensitivity", started["job_id"], 900)
    assert done["state"] == "done", _fail_with_log("sensitivity", done)


def test_calibration(study):
    """A short genetic-algorithm identification; DEBUG keeps the population small."""
    started = _req("POST", "/api/calibration/run", {
        "model_id": study,
        "settings": {"param_id_method": "genetic_algorithm",
                     "num_calls_to_function": 30, "DEBUG": True, "dt": 0.01},
    })
    done = _poll("calibration", started["job_id"], 900)
    assert done["state"] == "done", _fail_with_log("calibration", done)
    best = done.get("best_params") or {}
    assert best and all(isinstance(v, (int, float)) for v in best.values()), (
        f"calibration produced no usable best_params: {best!r}"
    )


def test_uq_evaluated_on_the_emulator(study, emulator):
    """UQ with ``use_emulator`` on -- the combination nothing else covers.

    Worth its own test rather than folding into the UQ run: sampling against a surrogate
    goes through a different path in CA from sampling against the solver, and it is the
    one the -full bundle exists to make possible.
    """
    started = _req("POST", "/api/uq/run", {
        "model_id": study,
        "settings": {"method": "mcmc", "library": "emcee",
                     # Enough to move the sampler through its real code path and no more.
                     "num_steps": 60, "burn_in": 0.5, "dt": 0.01,
                     "use_emulator": True},
    })
    done = _poll("uq", started["job_id"], 900)
    assert done["state"] == "done", _fail_with_log("UQ on the emulator", done)
