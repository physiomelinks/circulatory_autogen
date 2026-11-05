"""The ``time`` result shared by every SimulationHelper backend.

``ProtocolExecutor.run_protocol`` sets up an experiment's first sub-experiment with
``update_times(dt, 0.0, sim_time, pre_times[e])`` and each later one with
``update_times(dt, current_time, sim_time, 0.0)``, where ``current_time`` includes the warm-up.
So ``tSim`` is always on one clock that starts at the beginning of the warm-up, and the time a
user sees is ``tSim`` minus *the experiment's* warm-up -- not the current sub-experiment's, which
is 0 after the first. That is the same axis as the executor's ``t_by_exp``.
"""


class ExperimentTimeMixin:
    """``get_time()`` and the ``'time'`` result as cumulative time after the experiment's warm-up.

    A helper calls ``_track_experiment_start(start_time, pre_time)`` from its ``update_times``;
    a call with ``start_time == 0`` starts a new experiment and records its warm-up.
    """

    experiment_pre_time = 0.0

    def _track_experiment_start(self, start_time, pre_time):
        if start_time == 0:
            self.experiment_pre_time = float(pre_time or 0.0)

    def get_time(self, include_pre_time=False):
        """The output time points of the current sub-experiment.

        Args:
            include_pre_time: If True, return them on the solver's clock, which starts at the
                beginning of the experiment's warm-up; otherwise relative to the end of it.
        """
        if include_pre_time:
            return self.tSim
        return self.tSim - self.experiment_pre_time
