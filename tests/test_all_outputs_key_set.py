"""The all-outputs npz must have the same key set every run.

Myokit merges connected variables on import and keeps one qname for the pair, so the model's
time variable is named by the importer -- `volume_sum_module.t` for the 3compartment model,
because `environment.time` is connected to it and does not survive separately. This project
publishes the time under one stable name.

The log sometimes also carried that importer-specific qname and sometimes did not, so the npz
gained a duplicate of the time series in roughly one run of five and anything diffing two runs
saw a change that was not one.
"""
import numpy as np
import pytest

from libcuflynx.solver_wrappers.myokit_helper import SimulationHelper


class _Log(dict):
    """A DataLog stand-in: a dict that also knows which of its keys is the time."""

    def __init__(self, mapping, time_key):
        super().__init__(mapping)
        self._time_key = time_key

    def time_key(self):
        return self._time_key


def _collect(log):
    helper = SimulationHelper.__new__(SimulationHelper)
    helper.last_log = log
    return SimulationHelper._collect_all_results_dict_from_log(helper)


@pytest.mark.unit
def test_the_importers_time_qname_is_not_published_alongside_environment_time():
    """Two keys for one series is what made the file's key set vary between runs."""
    t = np.linspace(0.0, 1.0, 5)
    results = _collect(_Log({"volume_sum_module.t": t, "heart.v": t * 2},
                            time_key="volume_sum_module.t"))

    assert "volume_sum_module.t" not in results
    assert "heart.v" in results
    np.testing.assert_array_equal(results["environment.time"], t)


@pytest.mark.unit
def test_the_key_set_does_not_depend_on_what_the_importer_named_the_time():
    """The same model logged under either spelling must produce the same keys.

    This is the invariant the varying npz broke: two runs of one study differed only in
    whether the importer's time qname came through, and the file's key set followed.
    """
    t = np.linspace(0.0, 1.0, 5)
    importer_named = _collect(_Log({"volume_sum_module.t": t, "heart.v": t},
                                   time_key="volume_sum_module.t"))
    already_stable = _collect(_Log({"environment.time": t, "heart.v": t},
                                   time_key="environment.time"))

    assert set(importer_named) == set(already_stable) == {"environment.time", "heart.v"}


@pytest.mark.unit
def test_a_model_whose_time_is_already_environment_time_is_unchanged():
    """The common case must not lose its time series to the exclusion above."""
    t = np.linspace(0.0, 1.0, 5)
    results = _collect(_Log({"environment.time": t, "heart.v": t}, time_key="environment.time"))

    assert set(results) == {"environment.time", "heart.v"}
    np.testing.assert_array_equal(results["environment.time"], t)
