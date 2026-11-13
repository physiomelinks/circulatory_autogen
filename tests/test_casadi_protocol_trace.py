"""A constant driven by a protocol trace: casadi_python + cvodes against CVODE_myokit.

    dx/dt = -k*x + u(t),  x(0) = 1,  k calibrated,  u from protocol_traces

The trace has a ramp, a hold, a step (a repeated time) and a hold past its last point, and the
experiment has a pre_time, through which the trace's first value is held. Myokit interpolates
the trace linearly and holds its end values; casadi_python restarts CVODES at each trace point
with the input linear in between. The two must give the same trajectory, the same observables,
and the same gradient of the cost in k: CasADi AD (CVODES forward sensitivities) against Myokit
FSA. casadi_python used to refuse a trace outright.
"""
import json
import os

import numpy as np
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID

_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" name="traced">
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <component name="m">
    <variable name="t" units="second"/>
    <variable name="x" units="dimensionless" initial_value="1"/>
    <variable name="k" units="per_second" initial_value="0.5"/>
    <variable name="u" units="per_second" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML" xmlns:cellml="http://www.cellml.org/cellml/2.0#">
      <apply><eq/>
        <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><plus/>
          <apply><minus/><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
          <ci>u</ci>
        </apply>
      </apply>
    </math>
  </component>
</model>
"""

_PRE, _T, _DT = 1.0, 6.0, 0.01
_TRACE = {"t": [0.0, 1.0, 2.0, 2.0, 4.0, 4.5], "values": [0.2, 1.0, 1.0, 1.6, 0.4, 0.4]}


def _item(name, operation, value):
    return {"data_item_name": name, "data_type": "constant", "operation": operation,
            "operands": ["m/x"], "unit": "dimensionless", "weight": 1.0, "value": value,
            "std": 0.05, "experiment_idx": 0, "subexperiment_idx": 0}


def _obs_data():
    return {"protocol_info": {"pre_times": [_PRE], "sim_times": [[_T]],
                              "params_to_change": {"m/u": [["u_trace"]]},
                              "protocol_traces": {"u_trace": _TRACE}},
            "data_items": [_item("x_mean", "mean", 1.3), _item("x_max", "max", 2.1),
                           _item("x_min", "min", 0.6)],
            "prediction_items": []}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    pytest.importorskip("casadi")
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    out = tmp_path_factory.mktemp("traced")
    cellml = out / "traced.cellml"
    cellml.write_text(_CELLML)
    obs = out / "traced_obs_data.json"
    obs.write_text(json.dumps(_obs_data()))
    params = out / "traced_params_for_id.json"
    params.write_text(json.dumps({"version": 1, "params": [
        {"name": "k", "targets": ["m/k"], "min": 0.1, "max": 2.0}]}))
    casadi_py = PythonGenerator(str(cellml), output_dir=str(out), module_name="traced_casadi",
                                casadi_compat=True).generate()
    return {"cellml": str(cellml), "casadi": casadi_py, "obs": str(obs), "params": str(params),
            "out": str(out)}


def _engine(models, backend):
    if backend == "myokit":
        model_path, model_type = models["cellml"], "cellml"
        solver_info = {"solver": "CVODE_myokit", "MaximumStep": 0.01, "rtol": 1e-9,
                       "atol": 1e-9}
    else:
        model_path, model_type = models["casadi"], "casadi_python"
        solver_info = {"solver": "casadi_integrator", "method": "cvodes",
                       "max_num_steps": 100000}
    out_dir = os.path.join(models["out"], backend)
    os.makedirs(out_dir, exist_ok=True)
    outer = CVS0DParamID(
        model_path=model_path, model_type=model_type, param_id_method="sp_minimize",
        file_name_prefix="traced", params_for_id_path=models["params"],
        param_id_obs_path=models["obs"], sim_time=_T, pre_time=_PRE, dt=_DT,
        solver_info=solver_info, do_ad=True, DEBUG=True, param_id_output_dir=out_dir)
    engine = outer.param_id
    engine.output_dir = out_dir
    return engine


def _trajectory(models, backend, k):
    from libcuflynx.solver_wrappers import get_simulation_helper
    if backend == "myokit":
        h = get_simulation_helper(model_path=models["cellml"], model_type="cellml",
                                  solver="CVODE_myokit", dt=_DT, sim_time=_T, pre_time=_PRE,
                                  solver_info={"MaximumStep": 0.01, "rtol": 1e-9,
                                               "atol": 1e-9})
    else:
        h = get_simulation_helper(model_path=models["casadi"], model_type="casadi_python",
                                  solver="casadi_integrator", dt=_DT, sim_time=_T,
                                  pre_time=_PRE, solver_info={"method": "cvodes"})
    h.set_protocol_info(_obs_data()["protocol_info"])
    h.reset_states()
    h.update_times(_DT, 0.0, _T, _PRE)
    h.set_param_vals(["m/k", "m/u"], [k, "u_trace"])
    assert h.run()
    x = np.asarray(h.get_results([["m/x"]], flatten=True)[0], dtype=float).ravel()
    h.close_simulation()
    return x


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize("k", [0.5, 1.3])
def test_trace_driven_trajectory_matches_myokit(models, k):
    myokit, casadi = _trajectory(models, "myokit", k), _trajectory(models, "casadi", k)
    assert myokit.shape == casadi.shape
    # the trace moves x by order one, so this cannot pass on a constant input
    assert np.ptp(myokit) > 0.5
    assert np.max(np.abs(casadi - myokit)) < 1e-6


@pytest.mark.integration
@pytest.mark.parametrize("k", [0.5, 1.3])
def test_trace_driven_ad_gradient_matches_myokit_fsa(models, k):
    theta = np.array([k])
    fsa = _engine(models, "myokit")
    ad = _engine(models, "casadi")
    cost_fsa = float(fsa.get_cost(theta))
    cost_ad = float(ad.get_cost(theta))
    assert cost_ad == pytest.approx(cost_fsa, rel=1e-6)
    g_fsa = float(np.asarray(fsa.get_gradient(theta)).ravel()[0])
    g_ad = float(np.asarray(ad.get_gradient(theta)).ravel()[0])
    assert abs(g_fsa) > 1.0, f"a flat cost would make this vacuous ({g_fsa})"
    assert g_ad == pytest.approx(g_fsa, rel=1e-4)
