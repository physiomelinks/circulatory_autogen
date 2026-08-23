"""Sensitivity analysis has to hand its protocol to the simulation helper.

A ``params_to_change`` entry may name a protocol trace by string -- that is how an
AP-clamp or voltage-clamp study drives a variable from a measured waveform -- and
resolving that name against ``protocol_info['protocol_traces']`` is the helper's
job. ``param_id`` and ``ProtocolRunner`` have always called ``set_protocol_info``;
sensitivity analysis was the one path that did not, so those studies failed on the
first sample with

    params_to_change entry is a string trace key, but protocol_traces not found
    in protocol_info

after the whole Sobol design had already been generated.

A unit test rather than an integration one: the bug is a missing call, and
reproducing it through a real model means generating one and running
``num_samples*(2M+2)`` simulations to reach the failure.
"""
import pytest

from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA


class RecordingHelper:
    """Stands in for a simulation helper, remembering what it was told."""

    emulates_features = False

    def __init__(self):
        self.protocol_info = None
        self.times = None

    def set_protocol_info(self, protocol_info):
        self.protocol_info = protocol_info

    def update_times(self, dt, start, sim_time, pre_time):
        self.times = (dt, start, sim_time, pre_time)


@pytest.fixture
def agent(monkeypatch):
    """A ``sobol_SA`` with just the attributes the ``sim_helper`` property reads.

    Built with ``__new__`` deliberately: ``__init__`` parses obs_data and builds a
    model, none of which this behaviour depends on.
    """
    sa = sobol_SA.__new__(sobol_SA)
    sa._sim_helper = None
    sa.dt = 0.01
    sa.sim_time = 2.0
    sa.pre_time = 1.0
    sa.protocol_info = {
        "pre_times": [1.0],
        "sim_times": [[2.0]],
        "params_to_change": {"soma/V_set": [["a_measured_waveform"]]},
        "protocol_traces": {"a_measured_waveform": {"t": [0.0], "v": [1.0]}},
    }

    helper = RecordingHelper()
    monkeypatch.setattr(sobol_SA, "initialise_sim_helper", lambda self: helper)
    return sa, helper


@pytest.mark.unit
def test_sim_helper_is_given_the_protocol(agent):
    sa, helper = agent

    assert sa.sim_helper is helper
    assert helper.protocol_info is sa.protocol_info


@pytest.mark.unit
def test_the_traces_reach_the_helper(agent):
    """The specific thing that was missing: without protocol_traces the helper
    cannot resolve a string params_to_change entry, and raises on the first
    sample."""
    sa, helper = agent

    _ = sa.sim_helper

    assert "protocol_traces" in helper.protocol_info
    assert "a_measured_waveform" in helper.protocol_info["protocol_traces"]


@pytest.mark.unit
def test_times_are_still_applied(agent):
    """The protocol is set alongside the existing update_times, not instead of it."""
    sa, helper = agent

    _ = sa.sim_helper

    assert helper.times == (0.01, 0.0, 2.0, 1.0)


@pytest.mark.unit
def test_the_helper_is_built_once(agent):
    sa, helper = agent

    assert sa.sim_helper is sa.sim_helper is helper
