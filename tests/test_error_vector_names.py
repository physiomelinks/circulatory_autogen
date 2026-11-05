"""The saved error vectors identify their own observables (issue #341).

``percent_error_vec.npy`` / ``std_error_vec.npy`` are positional: entry i belongs
to ``data_items[i]``. Every consumer therefore re-derives the labels by reading
obs_data.json in the same order, so reordering that file silently relabels every
bar -- a wrong plot that looks entirely right. Now that these artefacts are an
interface rather than an implementation detail, the names travel with them.

No model/MPI needed: only the artefact writing is under test.
"""
import os

import numpy as np
import pytest

from libcuflynx.param_id.plot_outputs import ParamIDPlotOutputs


class _Client:
    def __init__(self, out, names):
        self.output_dir = str(out)
        self.obs_info = {"item_names_for_plotting": list(names), "num_obs": len(names)}


def _plotter(out, names):
    plotter = ParamIDPlotOutputs.__new__(ParamIDPlotOutputs)
    plotter.client = _Client(out, names)
    return plotter


def test_the_names_are_saved_beside_the_error_vectors(tmp_path):
    plotter = _plotter(tmp_path, ["flow", "pressure"])
    plotter.save_error_vectors(np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    names = np.load(os.path.join(tmp_path, "error_vec_names.npy"), allow_pickle=True)
    assert list(names) == ["flow", "pressure"]


def test_the_names_are_in_the_error_vectors_order(tmp_path):
    """The whole point: position i in the vectors is position i in the names, so a
    consumer never has to guess the ordering from another file."""
    plotter = _plotter(tmp_path, ["a", "b", "c"])
    percent = np.array([10.0, 20.0, 30.0])
    plotter.save_error_vectors(percent, np.zeros(3))

    names = np.load(os.path.join(tmp_path, "error_vec_names.npy"), allow_pickle=True)
    assert len(names) == len(percent)
    assert dict(zip(names, percent)) == {"a": 10.0, "b": 20.0, "c": 30.0}


def test_the_saved_names_are_raw_not_mathtext(tmp_path):
    """Data for whoever reads it. A consumer wanting mathtext can add the '$'; one
    wanting the plain name could not reliably strip them."""
    plotter = _plotter(tmp_path, ["q_{lv}"])
    plotter.save_error_vectors(np.array([1.0]), np.array([1.0]))

    names = np.load(os.path.join(tmp_path, "error_vec_names.npy"), allow_pickle=True)
    assert names[0] == "q_{lv}"
    assert not names[0].startswith("$")


def test_the_bar_plots_still_get_mathtext(tmp_path):
    """The plotting path is unchanged -- it wraps the same labels."""
    plotter = _plotter(tmp_path, ["q_{lv}", "p_{ao}"])
    assert list(plotter._observable_names_for_error_plots()) == ["$q_{lv}$", "$p_{ao}$"]


def test_the_existing_vectors_are_still_written(tmp_path):
    """Additive: nothing that already read these files may break."""
    plotter = _plotter(tmp_path, ["a"])
    plotter.save_error_vectors(np.array([5.0]), np.array([6.0]))

    assert np.load(os.path.join(tmp_path, "percent_error_vec.npy")) == pytest.approx([5.0])
    assert np.load(os.path.join(tmp_path, "std_error_vec.npy")) == pytest.approx([6.0])


def test_no_observables_writes_an_empty_name_vector(tmp_path):
    plotter = _plotter(tmp_path, [])
    plotter.save_error_vectors(np.array([]), np.array([]))

    names = np.load(os.path.join(tmp_path, "error_vec_names.npy"), allow_pickle=True)
    assert len(names) == 0
