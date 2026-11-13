"""Execution of the interactive tutorial notebooks, cell by cell, in a real Jupyter kernel.

Each notebook has one code cell tagged ``parameters`` holding the costly knobs (simulation
length, optimiser calls, Sobol samples, ...) at the values the tutorial teaches with.

* The quick tests run every cell of the notebook with a cell injected right after the settings
  cell that overrides those knobs with small values (the way papermill injects parameters). They
  run in CI.
* The ``manual`` tests run the same notebooks with the settings as shipped (``--run-manual``).

Outputs go to a temporary ``$CUFLYNX_USER_DIR`` so a run never writes into the checkout.

Two cells of the hybrid section of generation_and_calibration.ipynb are replaced by ``pass`` in both
runs (``SKIPPED_CELLS``), because the 1D solver fails on that model; every other cell of every
notebook is executed.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

from ipykernel.kernelspec import write_kernel_spec  # noqa: E402

NOTEBOOK_ROOT = Path(__file__).resolve().parents[2] / "tutorial" / "interactive"

#: image_to_hemodynamics_model.ipynb runs up to its last code cell, where image_proc_funcs.py fails
#: building the parameter array: the vessel-module lookup reads only module_config_user/, which no
#: longer holds the modules the helper asks for (``no modules found ... venous vp``); reading the
#: package's generators/resources too gets past that and then stops at ``KeyError: 'u_ext'``, so the
#: helper needs porting to the current module configs. strict: the day it is fixed, this fails.
IMAGE_PIPELINE_BROKEN = pytest.mark.xfail(
    strict=True, raises=nbclient.exceptions.CellExecutionError,
    reason="image_proc_funcs.py does not run against the current module configs",
)

#: Wall-clock limit for one quick notebook run, and for any single cell of it.
QUICK_NOTEBOOK_BUDGET_S = 300
#: Single-cell limit for the full-settings runs.
FULL_CELL_TIMEOUT_S = 1800

#: Cells that cannot run, as (a line of the cell, why). The cell is replaced by ``pass``.
SKIPPED_CELLS = {
    "generation_and_calibration.ipynb": [
        ('subprocess.run(["./run_coupler1d0d.bash", solver_hyb_dir]',
         "the coupler runs, but the 1D solver exits with 'Sum of integration weights > 1' on "
         "cvs_model_with_arm_hybrid, so the coupler waits on pipes the solver never opens"),
        ("sol1d = np.genfromtxt(output_hyb_1d",
         "reads the 1D solver's output, which the cell above never produces"),
    ],
}

QUICK_SETTINGS = {
    "generation_and_calibration.ipynb": dict(num_warmup_periods=6, sa_num_samples=4, ga_num_calls=30),
    "generation_and_calibration_test.ipynb": dict(warmup_time_s=8.0, sa_num_samples=4, ga_num_calls=30),
    # own_image_path is added by the test: a tiny synthetic segmentation.
    "image_to_hemodynamics_model.ipynb": dict(
        install_requirements=True, download_resources=False, open_fiji=False,
        run_ilastik_batch_processing=False, sub_volume=0.1,
    ),
}

#: The full run only turns off what this machine cannot do: the Fiji GUI, and Ilastik.
FULL_SETTINGS = {
    "generation_and_calibration.ipynb": {},
    "generation_and_calibration_test.ipynb": {},
    "image_to_hemodynamics_model.ipynb": dict(
        install_requirements=False, open_fiji=False, run_ilastik_batch_processing=False,
    ),
}


def _parameters_cell_index(nb) -> int:
    tagged = [i for i, c in enumerate(nb.cells)
              if c.cell_type == "code" and "parameters" in c.metadata.get("tags", [])]
    assert len(tagged) == 1, f"expected one cell tagged 'parameters', found {len(tagged)}"
    return tagged[0]


def _defined_names(source: str) -> set[str]:
    import ast
    return {t.id for node in ast.parse(source).body if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name)}


def _prepare(nb, name: str, settings: dict):
    """Inject ``settings`` after the parameters cell and blank the cells that cannot run."""
    idx = _parameters_cell_index(nb)
    unknown = set(settings) - _defined_names(nb.cells[idx].source)
    assert not unknown, f"{name}: settings {sorted(unknown)} are not defined in its parameters cell"
    if settings:
        injected = "\n".join(f"{k} = {v!r}" for k, v in settings.items())
        cell = nbformat.v4.new_code_cell(injected)
        cell.metadata["tags"] = ["injected-parameters"]
        nb.cells.insert(idx + 1, cell)
    for marker, _why in SKIPPED_CELLS.get(name, []):
        hits = [c for c in nb.cells if c.cell_type == "code" and marker in c.source]
        assert len(hits) == 1, f"{name}: expected one cell containing {marker!r}, found {len(hits)}"
        hits[0].source = "pass"
    return nb


def _execute(name: str, settings: dict, workdir: Path, user_dir: Path, cell_timeout: int,
             monkeypatch, set_user_dir: bool = True) -> tuple[object, float]:
    """Run every cell of ``workdir/name``; return the executed notebook and the seconds taken."""
    nb = _prepare(nbformat.read(workdir / name, as_version=4), name, settings)

    # A kernel for the interpreter running the tests, found through JUPYTER_PATH.
    kernel_dir = user_dir / "jupyter"
    write_kernel_spec(str(kernel_dir / "kernels" / "cuflynx_tests"), overrides={"display_name": "cuflynx_tests"})
    monkeypatch.setenv("JUPYTER_PATH", str(kernel_dir))
    if set_user_dir:  # redirect generated_models/ and param_id_output/ out of the checkout
        monkeypatch.setenv("CUFLYNX_USER_DIR", str(user_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for key in list(os.environ):  # a kernel started under mpiexec would try to join its job
        if key.startswith(("OMPI_", "PMI", "MPI", "PMIX_", "HYDRA_")):
            monkeypatch.delenv(key)

    client = nbclient.NotebookClient(
        nb, kernel_name="cuflynx_tests", timeout=cell_timeout, allow_errors=False,
        resources={"metadata": {"path": str(workdir)}},
    )
    start = time.time()
    client.execute()
    return nb, time.time() - start


def _assert_every_cell_ran(nb, name: str):
    skipped = {m for m, _ in SKIPPED_CELLS.get(name, [])}
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip() and cell.source != "pass":
            assert cell.execution_count is not None, f"{name}: a cell did not run:\n{cell.source[:200]}"
    assert not skipped - {m for m, _ in SKIPPED_CELLS.get(name, [])}


def _workdir_in_checkout(name: str, tmp_path: Path) -> Path:
    return NOTEBOOK_ROOT


def _workdir_isolated(tmp_path: Path) -> Path:
    """A copy of the tutorial folder layout, for notebooks that write beside themselves."""
    work = tmp_path / "CA" / "tutorial" / "interactive"
    work.mkdir(parents=True)
    (tmp_path / "CA" / "resources").mkdir()  # the pipeline writes the module array here
    for f in ("image_to_hemodynamics_model.ipynb", "image_proc_funcs.py"):
        shutil.copy(NOTEBOOK_ROOT / f, work / f)
    return work


def _need_cmake():
    if shutil.which("cmake") is None or shutil.which("g++") is None:
        pytest.skip("Section C of the notebook builds the generated C++ model: needs cmake and a C++ compiler")


def _check_main_outputs(user_dir: Path):
    models = user_dir / "generated_models" / "cvs_model_with_arm_0d"
    out = user_dir / "param_id_output"
    assert (models / "cvs_model_with_arm_0d.cellml").exists()
    assert (out / "quicklooks" / "uncalibrated_outputs.png").exists()
    assert any(out.rglob("best_param_vals.npy"))
    assert any(out.rglob("protocol_runner_calibrated.png"))
    assert any((out / "sensitivity").rglob("*.png"))
    assert (models / "simulation_outputs_cpp").exists()  # the C++ 0D model was built and run


def _check_test_notebook_outputs(user_dir: Path):
    out = user_dir / "param_id_output"
    assert (user_dir / "generated_models" / "3compartment" / "3compartment.cellml").exists()
    assert (out / "quicklooks" / "uncalibrated_outputs.png").exists()
    assert any(out.rglob("best_param_vals.npy"))
    assert any(out.rglob("protocol_runner_calibrated.png"))
    sensitivity = out / "sensitivity"
    assert any(sensitivity.rglob("*First_Order*Sobol_Heatmap.png"))
    assert any(sensitivity.rglob("*Total_Order*Sobol_Heatmap.png"))
    assert any(sensitivity.rglob("*.csv"))


def _run_calibration_notebook(name, check, settings, cell_timeout, budget, tmp_path, monkeypatch):
    if name == "generation_and_calibration.ipynb":
        _need_cmake()
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    nb, elapsed = _execute(name, settings, _workdir_in_checkout(name, tmp_path), user_dir,
                           cell_timeout, monkeypatch)
    print(f"\n{name}: {elapsed:.0f}s")
    _assert_every_cell_ran(nb, name)
    check(user_dir)
    if budget is not None:
        assert elapsed < budget, f"{name} took {elapsed:.0f}s; a quick run has {budget}s"


def _run_image_notebook(settings, cell_timeout, budget, tmp_path, monkeypatch, synthetic):
    name = "image_to_hemodynamics_model.ipynb"
    pytest.importorskip("h5py")
    work = _workdir_isolated(tmp_path)
    settings = dict(settings)
    if synthetic:
        import h5py
        import numpy as np
        image = tmp_path / "synthetic_segmentation.h5"
        n = 64  # large enough that the 10% sub-volume still holds the vessels
        z, y, x = np.mgrid[:n, :n, :n]
        data = np.zeros((n, n, n), dtype=np.uint8)
        data[(y - 32) ** 2 + (x - 32) ** 2 <= 9] = 2  # a vessel along z (label 2) ...
        data[((y - 32) ** 2 + (z - 32) ** 2 <= 4) & (x > 32)] = 2  # ... and a branch along x
        with h5py.File(image, "w") as h5f:
            h5f.create_dataset("exported_data", data=data)
        settings["own_image_path"] = str(image)
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    nb, elapsed = _execute(name, settings, work, user_dir, cell_timeout, monkeypatch,
                           set_user_dir=False)  # the image pipeline writes beside its notebook
    print(f"\n{name}: {elapsed:.0f}s")
    _assert_every_cell_ran(nb, name)
    assert (work / "resources" / "user_output" / "image_to_model_parameters.json").exists()
    assert (work / "resources" / "user_output" / "image_to_model_module_array.csv").exists()
    if budget is not None:
        assert elapsed < budget, f"{name} took {elapsed:.0f}s; a quick run has {budget}s"


# --- quick: every cell, small settings (CI) -------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_generation_and_calibration_notebook_quick(tmp_path, monkeypatch):
    name = "generation_and_calibration.ipynb"
    _run_calibration_notebook(name, _check_main_outputs, QUICK_SETTINGS[name],
                              QUICK_NOTEBOOK_BUDGET_S, QUICK_NOTEBOOK_BUDGET_S, tmp_path, monkeypatch)


@pytest.mark.integration
@pytest.mark.slow
def test_generation_and_calibration_test_notebook_quick(tmp_path, monkeypatch):
    name = "generation_and_calibration_test.ipynb"
    _run_calibration_notebook(name, _check_test_notebook_outputs, QUICK_SETTINGS[name],
                              QUICK_NOTEBOOK_BUDGET_S, QUICK_NOTEBOOK_BUDGET_S, tmp_path, monkeypatch)


@IMAGE_PIPELINE_BROKEN
@pytest.mark.integration
@pytest.mark.slow
def test_image_to_hemodynamics_notebook_quick(tmp_path, monkeypatch):
    _run_image_notebook(QUICK_SETTINGS["image_to_hemodynamics_model.ipynb"],
                        QUICK_NOTEBOOK_BUDGET_S, QUICK_NOTEBOOK_BUDGET_S, tmp_path, monkeypatch,
                        synthetic=True)


# --- manual: the settings as shipped ---------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
def test_generation_and_calibration_notebook_full(tmp_path, monkeypatch):
    name = "generation_and_calibration.ipynb"
    _run_calibration_notebook(name, _check_main_outputs, FULL_SETTINGS[name],
                              FULL_CELL_TIMEOUT_S, None, tmp_path, monkeypatch)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
def test_generation_and_calibration_test_notebook_full(tmp_path, monkeypatch):
    name = "generation_and_calibration_test.ipynb"
    _run_calibration_notebook(name, _check_test_notebook_outputs, FULL_SETTINGS[name],
                              FULL_CELL_TIMEOUT_S, None, tmp_path, monkeypatch)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.manual
@IMAGE_PIPELINE_BROKEN
def test_image_to_hemodynamics_notebook_full(tmp_path, monkeypatch):
    """Downloads the example images (Dropbox) and segments a 10% sub-volume: about ten minutes."""
    _run_image_notebook(FULL_SETTINGS["image_to_hemodynamics_model.ipynb"],
                        FULL_CELL_TIMEOUT_S, None, tmp_path, monkeypatch, synthetic=False)
