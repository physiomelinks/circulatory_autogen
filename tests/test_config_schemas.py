"""Module configs and module arrays, and PhLynx's multi_port semantics.

Module configs and arrays use PhLynx's key names (#163); the keys libcuflynx used before
(``vessel_type``, ``BC_type``, ``module_file``, ``inp_vessels``, ``out_vessels``), the older
array name ``<prefix>_vessel_array.*`` and a one-output heart are refused with the
``cuflynx-upgrade`` command that converts them (utilities/config_schemas.py).

multi_port values are case-insensitive. A whole-port "Sum" on a port other than a
volume_port makes the port's single variable the sum of the neighbours' variables (read as
the list form ``["sum"]``); on a volume_port it keeps the legacy sum_blood_volume sum.
"Multiply" on a port, at either end of a connection, sets each neighbour's variable to
``multiply_factor`` times this module's.

Fixture modules (in an ``external_modules_dir``):

* ``flow_src`` -- exit flow_port [v]: a prescribed flow (sinusoid).
* ``flow_sink`` -- entrance flow_port [v]: a prescribed flow drawn out (sinusoid).
* ``collector`` -- entrance flow_port [v_in_sum], multi_port "Sum"; ``v_seen = v_in_sum``.
* ``distributor`` -- exit flow_port [v_out_sum], multi_port "Sum"; ``v_seen = v_out_sum``
  (the microvasculature_network Nout pattern).
* ``tank`` -- exit volume_port [q]; ``dq/dt = rate``.
* ``vol_sum`` -- entrance volume_port [q_sum], multi_port "sum" (the legacy volume sum).
* ``gain`` -- exit gain_port [x] (a sinusoid), multi_port "Multiply".
* ``reader`` -- entrance gain_port [y]; ``y_seen = y``. ``reader_multi`` has a "True" port.
* ``gain_sum`` -- entrance gain_port [y_sum], multi_port "Sum"; ``y_seen = y_sum``.
* ``gain_vp`` / ``reader_vp`` -- ``gain`` (factor 3) and ``reader`` on vessel_ports.
* ``gain_in`` -- ``gain`` with its Multiply port (factor 2) an entrance flow_port, so it is
  the downstream end; ``gain_flow`` has a Multiply exit flow_port.
* ``reader_out`` / ``reader_in`` -- ``reader`` with a plain exit / entrance flow_port [y];
  ``reader_out_mul`` has a Multiply exit flow_port (factor 2) instead.
* ``collector2`` -- ``collector`` with a second "Sum" port (gain_port) on the same variable.
"""
import filecmp
import json
import os

import numpy as np
import pandas as pd
import pytest

from libcuflynx.generators.multi_port import normalise_port_multi_port
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.config_schemas import (normalise_module_config_entry,
                                                 normalise_module_array_columns,
                                                 normalise_vessel_record, module_array_path)


# --------------------------------------------------------------------------------------------
# fixture modules
# --------------------------------------------------------------------------------------------

def _var(name, units, interface, initial_value=None):
    init = f' initial_value="{initial_value}"' if initial_value is not None else ''
    return f'        <variable{init} name="{name}" public_interface="{interface}" units="{units}"/>\n'


def _eq(lhs, rhs):
    return f'            <apply><eq/>{lhs}{rhs}</apply>\n'


def _component(name, variables, equations):
    return (f'    <component name="{name}">\n' + _var('t', 'second', 'in') + ''.join(variables) +
            '        <math xmlns="http://www.w3.org/1998/Math/MathML">\n' + ''.join(equations) +
            '        </math>\n    </component>\n')


def _sinusoid(mean, amp, omega):
    """mean * (1 + amp * sin(omega * t))"""
    return (f'<apply><times/><ci>{mean}</ci><apply><plus/><cn cellml:units="dimensionless">1</cn>'
            f'<apply><times/><ci>{amp}</ci><apply><sin/><apply><times/><ci>{omega}</ci><ci>t</ci>'
            f'</apply></apply></apply></apply></apply>')


def _prescribed(name, variable, units):
    return _component(name, [
        _var(variable, units, 'out'),
        _var('mean', units, 'in'),
        _var('amp', 'dimensionless', 'in'),
        _var('omega', 'per_s', 'in'),
    ], [_eq(f'<ci>{variable}</ci>', _sinusoid('mean', 'amp', 'omega'))])


def _seen(name, variable, units):
    return _component(name, [_var(variable, units, 'in'), _var('v_seen', units, 'out')],
                      [_eq('<ci>v_seen</ci>', f'<ci>{variable}</ci>')])


MODULES_CELLML = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    '<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" '
    'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
    + _prescribed('flow_src_type', 'v', 'm3_per_s')
    + _prescribed('flow_sink_type', 'v', 'm3_per_s')
    + _seen('collector_type', 'v_in_sum', 'm3_per_s')
    + _seen('distributor_type', 'v_out_sum', 'm3_per_s')
    + _component('tank_type', [_var('q', 'm3', 'out', initial_value='1.0e-6'),
                               _var('rate', 'm3_per_s', 'in')],
                 [_eq('<apply><diff/><bvar><ci>t</ci></bvar><ci>q</ci></apply>', '<ci>rate</ci>')])
    + _seen('vol_sum_type', 'q_sum', 'm3')
    + _prescribed('gain_type', 'x', 'm3_per_s')
    + _seen('reader_type', 'y', 'm3_per_s')
    + _seen('gain_sum_type', 'y_sum', 'm3_per_s')
    + '</model>\n'
)


def _port(port_type, variables, multi_port=None, **extra):
    port = {"port_type": port_type, "variables": variables}
    if multi_port is not None:
        port["multi_port"] = multi_port
    port.update(extra)
    return port


def _module(module_type, component_type, entrance_ports, exit_ports, variables_and_units,
            module_subtype="nn"):
    """A module config entry."""
    return {
        "module_type": module_type,
        "module_subtype": module_subtype,
        "module_format": "cellml",
        "component_file": "schema_test_modules.cellml",
        "component_type": component_type,
        "entrance_ports": entrance_ports,
        "exit_ports": exit_ports,
        "general_ports": [],
        "variables_and_units": variables_and_units,
    }


def _prescribed_vars(variable):
    return [[variable, "m3_per_s", "access", "variable"],
            ["mean", "m3_per_s", "access", "constant"],
            ["amp", "dimensionless", "access", "constant"],
            ["omega", "per_s", "access", "constant"]]


def _seen_vars(variable, units="m3_per_s"):
    return [[variable, units, "access", "boundary_condition"],
            ["v_seen", units, "access", "variable"]]


def modules_config(volume_sum="sum", gain_factor=2.5):
    config = [
        _module("flow_src", "flow_src_type", [], [_port("flow_port", ["v"])], _prescribed_vars("v")),
        _module("flow_sink", "flow_sink_type", [_port("flow_port", ["v"])], [], _prescribed_vars("v")),
        _module("collector", "collector_type", [_port("flow_port", ["v_in_sum"], "Sum")], [],
                _seen_vars("v_in_sum")),
        _module("distributor", "distributor_type", [], [_port("flow_port", ["v_out_sum"], "Sum")],
                _seen_vars("v_out_sum")),
        _module("tank", "tank_type", [], [_port("volume_port", ["q"])],
                [["q", "m3", "access", "variable"], ["rate", "m3_per_s", "access", "constant"]]),
        _module("vol_sum", "vol_sum_type", [_port("volume_port", ["q_sum"], volume_sum)], [],
                _seen_vars("q_sum", "m3")),
        _module("gain", "gain_type", [],
                [_port("gain_port", ["x"], "Multiply", multiply_factor=gain_factor)],
                _prescribed_vars("x")),
        _module("gain_unit", "gain_type", [], [_port("gain_port", ["x"], "Multiply")],
                _prescribed_vars("x")),
        _module("reader", "reader_type", [_port("gain_port", ["y"])], [], _seen_vars("y")),
        _module("reader_multi", "reader_type", [_port("gain_port", ["y"], "True")], [], _seen_vars("y")),
        # the same on vessel_ports, which the generator maps on a separate path
        _module("gain_vp", "gain_type", [],
                [_port("vessel_port", ["x"], "multiply", multiply_factor=3)], _prescribed_vars("x")),
        _module("reader_vp", "reader_type", [_port("vessel_port", ["y"])], [], _seen_vars("y")),
        _module("gain_sum", "gain_sum_type", [_port("gain_port", ["y_sum"], "Sum")], [],
                _seen_vars("y_sum")),
        _module("gain_in", "gain_type",
                [_port("flow_port", ["x"], "Multiply", multiply_factor=2)], [], _prescribed_vars("x")),
        _module("gain_flow", "gain_type", [], [_port("flow_port", ["x"], "Multiply")],
                _prescribed_vars("x")),
        _module("reader_out", "reader_type", [], [_port("flow_port", ["y"])], _seen_vars("y")),
        _module("reader_in", "reader_type", [_port("flow_port", ["y"])], [], _seen_vars("y")),
        _module("reader_out_mul", "reader_type", [],
                [_port("flow_port", ["y"], "Multiply", multiply_factor=2)], _seen_vars("y")),
        _module("collector2", "collector_type",
                [_port("flow_port", ["v_in_sum"], "Sum"), _port("gain_port", ["v_in_sum"], "Sum")],
                [], _seen_vars("v_in_sum")),
    ]
    if gain_factor is None:
        del config[6]["exit_ports"][0]["multiply_factor"]
    return config


def _write_library(directory, files):
    """files: {file name: list of config entries}"""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "schema_test_modules.cellml"), "w") as f:
        f.write(MODULES_CELLML)
    for name, entries in files.items():
        with open(os.path.join(directory, name), "w") as f:
            json.dump(entries, f, indent=2)
    return str(directory)


@pytest.fixture(scope="module")
def library_dir(tmp_path_factory):
    return _write_library(tmp_path_factory.mktemp("schema_modules"),
                          {"schema_test_modules_config.json": modules_config()})


# --------------------------------------------------------------------------------------------
# model building helpers
# --------------------------------------------------------------------------------------------

def _prescribed_params(name, mean, amp, omega):
    return [(f"{name}/mean", "m3_per_s", mean), (f"{name}/amp", "dimensionless", amp),
            (f"{name}/omega", "per_s", omega)]


# the columns as PhLynx writes them, subtype before type
HEADER = "name,module_subtype,module_type,inp_instances,out_instances"


def _generate(work_dir, modules_dir, prefix, vessel_rows, params):
    """Write the module array and parameters and generate the model."""
    resources_dir = os.path.join(work_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    lines = [HEADER]
    for name, module_type, inp, out in vessel_rows:
        lines.append(f"{name},nn,{module_type},{' '.join(inp)},{' '.join(out)}")
    with open(os.path.join(resources_dir, f"{prefix}_module_array.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(resources_dir, f"{prefix}_parameters.csv"), "w") as f:
        f.write("variable_name,units,value,data_reference\n")
        for variable_name, units, value in params:
            f.write(f"{variable_name},{units},{value},test\n")

    generated_dir = os.path.join(work_dir, "generated_models")
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': resources_dir,
        'generated_models_dir': generated_dir,
        'external_modules_dir': modules_dir,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config), f"generation of {prefix} failed"
    return os.path.join(generated_dir, prefix, f"{prefix}.cellml")


def _simulate(cellml_path, names, sim_time=2.0, dt=0.01):
    sim = get_simulation_helper(model_path=cellml_path, model_type='cellml',
                                solver='CVODE_myokit', dt=dt, sim_time=sim_time, pre_time=0.0)
    assert sim.run(), "simulation failed"
    results = sim.get_results(names, flatten=True)
    return {name: np.asarray(result, dtype=float) for name, result in zip(names, results)}


def _read(path):
    with open(path) as f:
        return f.read()


EXACT = dict(rtol=1e-12, atol=0.0)

# a model that uses every fixture module: flow sums on both sides, a legacy volume sum and
# Multiply into a plain port and into a Sum port
FULL_ROWS = [
    ("src_a", "flow_src", [], ["coll"]),
    ("src_b", "flow_src", [], ["coll"]),
    ("coll", "collector", ["src_a", "src_b"], []),
    ("dist", "distributor", [], ["sink_a", "sink_b"]),
    ("sink_a", "flow_sink", ["dist"], []),
    ("sink_b", "flow_sink", ["dist"], []),
    ("tank_a", "tank", [], ["vsum"]),
    ("tank_b", "tank", [], ["vsum"]),
    ("vsum", "vol_sum", ["tank_a", "tank_b"], []),
    ("g1", "gain", [], ["rd", "gsum"]),
    ("g2", "gain_unit", [], ["gsum"]),
    ("rd", "reader", ["g1"], []),
    ("gsum", "gain_sum", ["g1", "g2"], []),
]
FULL_PARAMS = (_prescribed_params("src_a", 1.0e-5, 0.5, 6.0) +
               _prescribed_params("src_b", 2.0e-5, 0.3, 4.0) +
               _prescribed_params("sink_a", 3.0e-6, 0.4, 5.0) +
               _prescribed_params("sink_b", 4.0e-6, 0.2, 7.0) +
               [("tank_a/rate", "m3_per_s", 1.0e-6), ("tank_b/rate", "m3_per_s", 3.0e-6)] +
               _prescribed_params("g1", 1.0e-5, 0.5, 3.0) +
               _prescribed_params("g2", 2.0e-5, 0.4, 8.0))


def _generate_full(work_dir, modules_dir, **kwargs):
    return _generate(str(work_dir), modules_dir, "schema_full", FULL_ROWS, FULL_PARAMS, **kwargs)


def _assert_same_generated_models(path_a, path_b):
    """Every generated file of the two models is byte-identical."""
    dir_a, dir_b = os.path.dirname(path_a), os.path.dirname(path_b)
    names = sorted(f for f in os.listdir(dir_a) if not os.path.isdir(os.path.join(dir_a, f)))
    assert names == sorted(f for f in os.listdir(dir_b) if not os.path.isdir(os.path.join(dir_b, f)))
    assert any(name.endswith('.cellml') for name in names)
    _, mismatch, errors = filecmp.cmpfiles(dir_a, dir_b, names, shallow=False)
    assert not mismatch and not errors, f"generated files differ: {mismatch + errors}"


# --------------------------------------------------------------------------------------------
# 1. module config schemas
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_entries_come_back_with_module_type_and_subtype_first():
    entry = dict(reversed(list(modules_config()[0].items())))
    assert list(normalise_module_config_entry(entry))[:2] == ["module_type", "module_subtype"]
    record = normalise_vessel_record({"out_instances": "x y", "module_subtype": "nn",
                                      "module_type": "flow_src", "name": "a"})
    assert list(record) == ["name", "module_type", "module_subtype", "inp_instances",
                            "out_instances"]
    assert record["out_instances"] == ["x", "y"]


@pytest.mark.unit
@pytest.mark.parametrize("read, value, kind", [
    (normalise_module_config_entry,
     {"vessel_type": "a", "BC_type": "nn", "module_file": "f.cellml", "module_type": "a_type"},
     "module-config"),
    (normalise_module_config_entry,
     {"module_type": "pair", "module_subtype": "supermodule", "module_format": "supermodule",
      "default_parameters": "p.csv", "submodules": [
          {"name": "a", "module_type": "flow_src", "module_subtype": "nn"}]}, "module-config"),
    (normalise_module_config_entry,
     dict(_module("a", "a_type", [], [], []), default_instance="p"), "module-config"),
    (normalise_vessel_record,
     {"name": "a", "vessel_type": "flow_src", "BC_type": "nn", "out_vessels": ["z"]},
     "module-array"),
])
def test_an_old_key_is_refused_with_the_command_that_converts_it(read, value, kind):
    with pytest.raises(ValueError, match=rf"no longer reads.*cuflynx-upgrade {kind} lib\.json"):
        read(value, source="lib.json")


@pytest.mark.unit
def test_an_entry_missing_a_key_is_rejected():
    with pytest.raises(ValueError, match=r"missing \['component_file'\]"):
        normalise_module_config_entry({"module_type": "a", "module_subtype": "nn",
                                       "component_type": "a_type"}, source="lib.json")


@pytest.mark.unit
def test_old_module_array_columns_are_refused():
    old = pd.DataFrame([["a", "nn", "src", "", "b"]],
                       columns=["name", "BC_type", "vessel_type", "inp_vessels", "out_vessels"])
    with pytest.raises(ValueError, match="cuflynx-upgrade module-array m.csv"):
        normalise_module_array_columns(old, source="m.csv")
    current = pd.DataFrame([["a", "nn", "src", "", "b"]], columns=HEADER.split(","))
    assert normalise_module_array_columns(current) is current


@pytest.mark.unit
def test_the_old_vessel_array_file_name_is_refused(tmp_path):
    assert module_array_path(str(tmp_path), "m").endswith("m_module_array.csv")
    (tmp_path / "m_vessel_array.csv").write_text(HEADER + "\n")
    with pytest.raises(ValueError, match="cuflynx-upgrade module-array .*m_vessel_array.csv"):
        module_array_path(str(tmp_path), "m")
    (tmp_path / "m_module_array.csv").write_text(HEADER + "\n")
    assert module_array_path(str(tmp_path), "m").endswith("m_module_array.csv")


# --------------------------------------------------------------------------------------------
# 3. multi_port "sum" is case-insensitive; whole-port sum on non-volume ports
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_multi_port_normalisation():
    for value in ["sum", "Sum", "SUM"]:
        assert normalise_port_multi_port(_port("volume_port", ["q"], value), "m")["multi_port"] == "sum"
        assert normalise_port_multi_port(_port("flow_port", ["v"], value), "m")["multi_port"] == ["sum"]
    for value in ["multiply", "Multiply", "MULTIPLY"]:
        assert normalise_port_multi_port(_port("p", ["x"], value), "m")["multi_port"] == "Multiply"
    assert normalise_port_multi_port(_port("p", ["v", "u"], ["Sum", "True"]), "m")["multi_port"] == \
        ["sum", "True"]
    unchanged = _port("p", ["v"], "True")
    assert normalise_port_multi_port(unchanged, "m") is unchanged
    with pytest.raises(ValueError, match="exactly one variable"):
        normalise_port_multi_port(_port("flow_port", ["v", "u"], "Sum"), "m")
    with pytest.raises(ValueError, match="exactly one variable"):
        normalise_port_multi_port(_port("p", ["x", "y"], "Multiply"), "m")
    with pytest.raises(ValueError, match="not a number"):
        normalise_port_multi_port(_port("p", ["x"], "Multiply", multiply_factor="two"), "m")
    with pytest.raises(ValueError, match="only applies"):
        normalise_port_multi_port(_port("p", ["x"], "True", multiply_factor=2), "m")


@pytest.mark.integration
def test_volume_port_sum_is_case_insensitive(tmp_path):
    rows = [("tank_a", "tank", [], ["vsum"]), ("tank_b", "tank", [], ["vsum"]),
            ("vsum", "vol_sum", ["tank_a", "tank_b"], [])]
    params = [("tank_a/rate", "m3_per_s", 1.0e-6), ("tank_b/rate", "m3_per_s", 3.0e-6)]
    paths = {}
    for value in ["sum", "Sum", "SUM"]:
        modules_dir = _write_library(tmp_path / f"modules_{value}",
                                     {"m_modules_config.json": modules_config(volume_sum=value)})
        paths[value] = _generate(str(tmp_path / value), modules_dir, "schema_vol", rows, params)
    assert '<component name="sum_blood_volume">' in _read(paths["sum"])
    _assert_same_generated_models(paths["sum"], paths["Sum"])
    _assert_same_generated_models(paths["sum"], paths["SUM"])

    res = _simulate(paths["Sum"], ["tank_a/q", "tank_b/q", "vsum/v_seen"])
    np.testing.assert_allclose(res["vsum/v_seen"], res["tank_a/q"] + res["tank_b/q"], **EXACT)
    assert np.ptp(res["vsum/v_seen"]) > 0


@pytest.mark.integration
def test_whole_port_sum_on_flow_ports(tmp_path, library_dir):
    """A "Sum" entrance port sums its upstream neighbours; a "Sum" exit port (the
    microvasculature_network v_out_sum pattern) sums its downstream neighbours."""
    path = _generate_full(tmp_path, library_dir)
    text = _read(path)
    assert '<component name="multiport_sum_coll_v_in_sum">' in text
    assert '<component name="multiport_sum_dist_v_out_sum">' in text

    res = _simulate(path, ["src_a/v", "src_b/v", "coll/v_seen",
                           "sink_a/v", "sink_b/v", "dist/v_seen"])
    np.testing.assert_allclose(res["coll/v_seen"], res["src_a/v"] + res["src_b/v"], **EXACT)
    np.testing.assert_allclose(res["dist/v_seen"], res["sink_a/v"] + res["sink_b/v"], **EXACT)
    assert np.ptp(res["coll/v_seen"]) > 0 and np.ptp(res["dist/v_seen"]) > 0


@pytest.mark.integration
def test_sum_to_sum_connection_is_rejected(tmp_path, library_dir):
    """Only one side of a connection can sum (as in PhLynx)."""
    rows = [("dist", "distributor", [], ["coll"]),
            ("coll", "collector", ["dist"], [])]
    with pytest.raises(ValueError, match="only one side"):
        _generate(str(tmp_path), library_dir, "schema_sum_sum", rows, [])


# --------------------------------------------------------------------------------------------
# 4. multi_port "Multiply"
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("gain_factor, expected", [(2.5, 2.5), (None, 1.0), ("0.5", 0.5)])
def test_multiply(tmp_path, gain_factor, expected):
    modules_dir = _write_library(tmp_path / "modules",
                                 {"m_modules_config.json": modules_config(gain_factor=gain_factor)})
    path = _generate_full(tmp_path, modules_dir)
    text = _read(path)
    assert '<component name="multiport_multiply_rd_y">' in text

    res = _simulate(path, ["g1/x", "g2/x", "rd/v_seen", "gsum/v_seen"])
    # into a plain port: y = multiply_factor * x
    np.testing.assert_allclose(res["rd/v_seen"], expected * res["g1/x"], **EXACT)
    # into a Sum port: each Multiply term is scaled by its own port's factor (g2 has none)
    np.testing.assert_allclose(res["gsum/v_seen"], expected * res["g1/x"] + res["g2/x"], **EXACT)
    assert np.ptp(res["rd/v_seen"]) > 0


@pytest.mark.integration
def test_multiply_on_vessel_ports(tmp_path, library_dir):
    rows = [("g", "gain_vp", [], ["rd_a", "rd_b"]),
            ("rd_a", "reader_vp", ["g"], []), ("rd_b", "reader_vp", ["g"], [])]
    path = _generate(str(tmp_path), library_dir, "schema_mul_vp", rows,
                     _prescribed_params("g", 1.0e-5, 0.5, 3.0))
    res = _simulate(path, ["g/x", "rd_a/v_seen", "rd_b/v_seen"])
    for reader in ["rd_a", "rd_b"]:
        np.testing.assert_allclose(res[f"{reader}/v_seen"], 3.0 * res["g/x"], **EXACT)
    assert np.ptp(res["g/x"]) > 0


@pytest.mark.integration
def test_multiply_on_the_downstream_end(tmp_path, library_dir):
    """A Multiply entrance port scales its variable into the upstream neighbour: into a plain
    port (y = 2 x) and into a "Sum" port (one term of the sum)."""
    rows = [("rd", "reader_out", [], ["g"]), ("g", "gain_in", ["rd"], []),
            ("dist", "distributor", [], ["g2", "sink"]), ("g2", "gain_in", ["dist"], []),
            ("sink", "flow_sink", ["dist"], [])]
    params = (_prescribed_params("g", 1.0e-5, 0.5, 3.0) + _prescribed_params("g2", 2.0e-5, 0.4, 8.0) +
              _prescribed_params("sink", 3.0e-6, 0.4, 5.0))
    path = _generate(str(tmp_path), library_dir, "schema_mul_down", rows, params)
    res = _simulate(path, ["g/x", "rd/v_seen", "g2/x", "sink/v", "dist/v_seen"])
    np.testing.assert_allclose(res["rd/v_seen"], 2.0 * res["g/x"], **EXACT)
    np.testing.assert_allclose(res["dist/v_seen"], 2.0 * res["g2/x"] + res["sink/v"], **EXACT)
    assert np.ptp(res["rd/v_seen"]) > 0


@pytest.mark.integration
def test_multiply_on_both_ends_is_rejected(tmp_path, library_dir):
    rows = [("g1", "gain_flow", [], ["g2"]), ("g2", "gain_in", ["g1"], [])]
    params = _prescribed_params("g1", 1.0e-5, 0.5, 3.0) + _prescribed_params("g2", 2.0e-5, 0.4, 8.0)
    with pytest.raises(ValueError, match='both ports are "Multiply"'):
        _generate(str(tmp_path), library_dir, "schema_mul_both", rows, params)


@pytest.mark.integration
def test_a_variable_summed_through_two_ports_is_rejected(tmp_path, library_dir):
    rows = [("src", "flow_src", [], ["coll"]), ("g", "gain", [], ["coll"]),
            ("coll", "collector2", ["src", "g"], [])]
    params = _prescribed_params("src", 1.0e-5, 0.5, 6.0) + _prescribed_params("g", 2.0e-5, 0.4, 8.0)
    with pytest.raises(ValueError, match='"coll" sums "v_in_sum" through ports "flow_port" and '
                                         '"gain_port"; a variable can be summed through one port only'):
        _generate(str(tmp_path), library_dir, "schema_sum_two_ports", rows, params)


@pytest.mark.integration
@pytest.mark.parametrize("rows, expected", [
    # a BC that is only a sum term
    ([("b", "reader_out", [], ["coll"]), ("coll", "collector", ["b"], [])],
     {"b/v_seen": 3.0e-6, "coll/v_seen": 3.0e-6}),
    # a BC that is the source of a Multiply; its target BC is computed
    ([("b", "reader_out_mul", [], ["r"]), ("r", "reader_in", ["b"], [])],
     {"b/v_seen": 3.0e-6, "r/v_seen": 6.0e-6}),
])
def test_a_bc_that_only_feeds_a_sum_or_multiply_keeps_its_value(tmp_path, library_dir, rows,
                                                                 expected):
    path = _generate(str(tmp_path), library_dir, "schema_bc_term", rows,
                     [("b/y", "m3_per_s", 3.0e-6)])
    res = _simulate(path, list(expected), sim_time=0.1)
    for name, value in expected.items():
        np.testing.assert_allclose(res[name], value, rtol=1e-12, err_msg=name)


@pytest.mark.integration
def test_bcs_connected_only_to_each_other_are_rejected(tmp_path, library_dir):
    rows = [("a", "reader_out", [], ["b"]), ("b", "reader_in", ["a"], [])]
    with pytest.raises(ValueError, match='"y" of "a", "y" of "b" are connected only to each other'):
        _generate(str(tmp_path), library_dir, "schema_bc_only", rows,
                  [("a/y", "m3_per_s", 1.0), ("b/y", "m3_per_s", 2.0)])


@pytest.mark.integration
def test_two_multiply_ports_into_one_plain_port_are_rejected(tmp_path, library_dir):
    rows = [("g1", "gain", [], ["rd"]), ("g2", "gain_unit", [], ["rd"]),
            ("rd", "reader_multi", ["g1", "g2"], [])]
    params = _prescribed_params("g1", 1.0e-5, 0.5, 3.0) + _prescribed_params("g2", 2.0e-5, 0.4, 8.0)
    with pytest.raises(ValueError, match="set through \"Multiply\" ports by both"):
        _generate(str(tmp_path), library_dir, "schema_mul2", rows, params)
