"""casadi_python reports and integrates on CVODE_myokit's output grid.

Myokit runs pre_time in full, then logs int(sim_time/dt) + 1 points spanning exactly sim_time.
When dt does not divide sim_time (or pre_time) those steps are slightly longer than dt.
casadi_python uses the same grid, so results from the two backends line up sample by sample.

    dx/dt = -k*x,  x(0) = 1,  so x(t) = exp(-k t) at every reported time.
"""
import numpy as np
import pytest

from libcuflynx.solver_wrappers import get_simulation_helper

_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" name="decay">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="m">
    <variable name="t" units="second"/>
    <variable name="x" units="dimensionless" initial_value="1"/>
    <variable name="k" units="per_second" initial_value="0.5"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/>
        <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><minus/><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
      </apply>
    </math>
  </component>
</model>
"""
_K = 0.5

# (dt, pre_time, sim_time): dividing, sim_time not divided, both not divided, pre_time < dt.
GRIDS = [(0.01, 0.0, 1.0), (0.0073, 0.0, 1.0), (0.007, 0.5, 1.0), (0.01, 0.004, 0.5)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    pytest.importorskip("casadi")
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    out = tmp_path_factory.mktemp("decay")
    cellml = out / "decay.cellml"
    cellml.write_text(_CELLML)
    casadi_py = PythonGenerator(str(cellml), output_dir=str(out), module_name="decay_casadi",
                                casadi_compat=True).generate()
    return {"cellml": str(cellml), "casadi": casadi_py}


def _times(helper):
    return np.asarray(helper.get_time()).ravel()


@pytest.mark.unit
@pytest.mark.parametrize("dt, pre_time, sim_time", GRIDS)
def test_casadi_reports_myokits_times(models, dt, pre_time, sim_time):
    myokit = get_simulation_helper(model_path=models["cellml"], solver="CVODE_myokit",
                                   model_type="cellml", dt=dt, sim_time=sim_time,
                                   pre_time=pre_time)
    casadi = get_simulation_helper(model_path=models["casadi"], solver="casadi_integrator",
                                   model_type="casadi_python", dt=dt, sim_time=sim_time,
                                   pre_time=pre_time, solver_info={"method": "bdf"})
    myokit.run()
    casadi.run()
    expected = _times(myokit)
    assert len(expected) == int(sim_time / dt) + 1
    np.testing.assert_allclose(_times(casadi), expected, rtol=0, atol=1e-12)
    assert len(np.asarray(casadi.get_results([["m/x"]], flatten=True)[0]).ravel()) == len(expected)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["cvodes", "bdf", "rk"])
@pytest.mark.parametrize("dt, pre_time, sim_time", GRIDS)
def test_casadi_values_are_at_the_reported_times(models, method, dt, pre_time, sim_time):
    """Each part (pre_time, sim_time) is stepped at its own step, so x is exp(-k t) at each
    reported time: a step of dt where the grid's step is longer would put x up to a dt off."""
    casadi = get_simulation_helper(model_path=models["casadi"], solver="casadi_integrator",
                                   model_type="casadi_python", dt=dt, sim_time=sim_time,
                                   pre_time=pre_time,
                                   solver_info={"method": method, "jit": False})
    casadi.run()
    t = pre_time + _times(casadi)  # get_time() counts from the end of pre_time
    x = np.asarray(casadi.get_results([["m/x"]], flatten=True)[0]).ravel()
    np.testing.assert_allclose(x, np.exp(-_K * t), rtol=1e-5)
