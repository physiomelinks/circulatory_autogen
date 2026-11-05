"""The ``time`` result means one thing on every backend (issue #557).

``get_results(['time'])`` and ``get_time()`` are the cumulative time after the experiment's
warm-up, the same axis as the executor's ``t_by_exp``, in every sub-experiment.
"""
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
from libcuflynx.solver_wrappers import get_simulation_helper

# dx/dt = u
_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="probe">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="comp">
    <variable name="t" units="second"/>
    <variable name="u" units="dimensionless" initial_value="1"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><times/><ci>u</ci><cn cellml:units="per_second">1</cn></apply></apply>
    </math>
  </component>
</model>"""

_EXTERNAL_MODEL = '''
    import numpy as np

    class Probe:
        parameters = {"comp/u": 1.0}
        output_names = ["comp/x"]

        def init_solver(self, config):
            self.update_times(config['dt'], config['start_time'],
                              config['sim_time'], config['pre_time'])

        def update_times(self, dt, start_time, sim_time, pre_time):
            self.t = start_time + np.arange(int(pre_time / dt) + int(sim_time / dt) + 1) * dt

        def set_param_vals(self, param_dict):
            pass

        def run(self):
            return True

        def get_results(self):
            return {"comp/x": self.t.copy()}

    SIM_HELPER = Probe
'''

_DT = 0.1
_PROTOCOL = {"pre_times": [0.5], "sim_times": [[1.0, 1.0]], "params_to_change": {}}

# backend -> (model_type, solver, how the model file is made)
_BACKENDS = {
    "myokit": ("cellml", "CVODE_myokit", "cellml"),
    "opencor": ("cellml", "CVODE_opencor", "cellml"),
    "python": ("python", "solve_ivp", "python"),
    "casadi": ("casadi_python", "casadi_integrator", "casadi"),
    "aadc": ("aadc_python", "aadc_semi_implicit", "aadc"),
    "external": ("external_python", "external", "external"),
}
_SOLVER_INFO = {
    "myokit": {"MaximumStep": 0.01},
    "opencor": {"MaximumStep": 0.01},
    "python": {"method": "RK45"},
    "casadi": {"method": "cvodes"},
}


def _model_file(kind, tmp_path):
    cellml = tmp_path / "probe.cellml"
    cellml.write_text(_CELLML)
    if kind == "cellml":
        return str(cellml)
    if kind == "external":
        path = tmp_path / "probe_external.py"
        path.write_text(textwrap.dedent(_EXTERNAL_MODEL))
        return str(path)
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    return PythonGenerator(str(cellml), output_dir=str(tmp_path / kind), module_name="probe",
                           casadi_compat=(kind == "casadi"),
                           aadc_compat=(kind == "aadc")).generate()


@pytest.mark.parametrize("backend", [
    "myokit",
    pytest.param("opencor", marks=pytest.mark.need_opencor),
    "python",
    "casadi",
    "aadc",
    "external",
])
def test_time_result_matches_t_by_exp(backend, tmp_path):
    if backend == "aadc":
        pytest.importorskip("aadc")
    model_type, solver, kind = _BACKENDS[backend]
    helper = get_simulation_helper(
        model_path=_model_file(kind, tmp_path), model_type=model_type, solver=solver,
        dt=_DT, sim_time=1.0, pre_time=0.5,
        solver_info=_SOLVER_INFO.get(backend, {}))

    ok, _, time_by_sub, t_by_exp = ProtocolExecutor(helper).run_protocol(
        _PROTOCOL, result_variables=["time"], extra_result_variables=["time"])
    assert ok

    t = np.asarray(t_by_exp[0], dtype=float)
    assert t[0] == pytest.approx(0.0) and t[-1] == pytest.approx(2.0)
    n = int(1.0 / _DT)
    for sub in (0, 1):
        expected = t[sub * n:(sub + 1) * n + 1]
        got = np.ravel(np.asarray(time_by_sub[(0, sub)][0], dtype=float))
        np.testing.assert_allclose(got, expected, atol=1e-9, err_msg=f"sub-experiment {sub}")
    np.testing.assert_allclose(np.asarray(helper.get_time(), dtype=float), t[n:], atol=1e-9)


@pytest.mark.unit
def test_emulator_get_time_is_after_the_experiment_warm_up():
    """The emulator integrates nothing, so only its time bookkeeping is checked."""
    from libcuflynx.solver_wrappers.emulator_solver_helper import SimulationHelper

    helper = object.__new__(SimulationHelper)
    helper.update_times(_DT, 0.0, 1.0, 0.5)
    np.testing.assert_allclose(helper.get_time(), np.linspace(0.0, 1.0, 11))
    helper.update_times(_DT, 1.5, 1.0, 0.0)
    np.testing.assert_allclose(helper.get_time(), np.linspace(1.0, 2.0, 11))
