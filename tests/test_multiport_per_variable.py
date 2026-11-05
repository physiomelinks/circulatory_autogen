"""Per-variable (list-form) ``multi_port`` semantics.

A port may give ``multi_port`` as a list aligned with its ``variables``::

    {"port_type": "vessel_port", "variables": ["v_in", "u"], "multi_port": ["sum", "True"]}

"sum" makes this module's variable the sum of the corresponding variable over every module
connected through the port; "True" maps this module's variable to the corresponding variable of
every connected module. This is what the algebraic node modules ``flow_merge`` (many inflows,
one outflow) and ``flow_split`` (one inflow, many outflows) of the module library need: the flows
are summed, the node pressure is shared.

The fixture modules below live in an ``external_modules_dir``:

* ``flow_source`` -- a prescribed flow ``v`` (sinusoid) that reads a pressure ``u_out``;
  ``flow_source_mm3`` is the same in mm3_per_s, ``bad_source`` has flow in pressure units.
* ``flow_merge`` (BC ``vp``) -- entrance vessel_port [v_in, u], multi_port ["sum", "True"];
  exit vessel_port [v_out, u_d]; ``v_out = v_in``, ``u = u_d``. ``flow_merge_plain`` is the
  same module with a plain entrance port.
* ``windkessel`` -- takes a flow, gives a pressure (one state).
* ``pressure_source`` -- a prescribed pressure ``u`` that reads a flow ``v``.
* ``flow_split`` (BC ``pv``) -- entrance vessel_port [v_in, u]; exit vessel_port [v_out, u_d],
  multi_port ["sum", "True"]; ``v_in = v_out``, ``u_d = u``.
* ``rl_sink`` -- takes a pressure, gives a flow through an R-L branch (one state).
* ``volume_store`` -- a prescribed volume ``q`` [m3] on an exit volume_link_port with
  multi_port "True"; ``volume_reader`` reads it as ``q_vc`` [litre], so every connection
  converts units. ``volume_reader_q`` reads it as ``q`` [litre]. ``volume_reader_up`` /
  ``volume_store_down`` are the same with the reader upstream of the store.
* ``temperature_source`` (``T`` in kelvin) and ``temperature_reader`` (``T`` in celsius).
* ``flow_source_one`` / ``windkessel_one`` -- one-variable vessel_ports [v] / [v_in];
  ``windkessel_blank`` has the entrance vessel_port [v_in, ""].
* ``venous_uptake`` (BC ``vp``) -- a venous node with a plain entrance vessel_port [v_in, u]
  and a separate entrance blood_uptake_port [v_uptake], multi_port "sum";
  ``v_out = v_in + v_uptake``. ``uptake_source`` feeds that port; ``uptake_source_two_ports``
  has two exit blood_uptake_ports [v_up], and ``uptake_source_shared`` one with multi_port
  "True".

Built-in ``arterial_simple`` vessels are mixed in as neighbours too.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from libcuflynx.generators.multi_port import (list_multi_port, is_multi_port,
                                              validate_list_multi_port,
                                              module_has_list_multi_port)
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.parameters_file import model_parameter_rows


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


def _flow_source(name, flow_units):
    return _component(name, [
        _var('v', flow_units, 'out'),
        _var('u_out', 'J_per_m3', 'in'),
        _var('v_mean', flow_units, 'in'),
        _var('amp', 'dimensionless', 'in'),
        _var('omega', 'per_s', 'in'),
        _var('u_seen', 'J_per_m3', 'out'),
    ], [_eq('<ci>v</ci>', _sinusoid('v_mean', 'amp', 'omega')),
        _eq('<ci>u_seen</ci>', '<ci>u_out</ci>')])


MODULES_CELLML = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    '<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" '
    'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
    + _flow_source('flow_source_type', 'm3_per_s')
    + _flow_source('flow_source_mm3_type', 'mm3_per_s')
    + _flow_source('bad_source_type', 'J_per_m3')
    + _component('flow_merge_type', [
        _var('v_in', 'm3_per_s', 'in'),
        _var('u', 'J_per_m3', 'out'),
        _var('v_out', 'm3_per_s', 'out'),
        _var('u_d', 'J_per_m3', 'in'),
    ], [_eq('<ci>v_out</ci>', '<ci>v_in</ci>'), _eq('<ci>u</ci>', '<ci>u_d</ci>')])
    + _component('flow_split_type', [
        _var('v_in', 'm3_per_s', 'out'),
        _var('u', 'J_per_m3', 'in'),
        _var('v_out', 'm3_per_s', 'in'),
        _var('u_d', 'J_per_m3', 'out'),
    ], [_eq('<ci>v_in</ci>', '<ci>v_out</ci>'), _eq('<ci>u_d</ci>', '<ci>u</ci>')])
    + _component('windkessel_type', [
        _var('v_in', 'm3_per_s', 'in'),
        _var('u', 'J_per_m3', 'out'),
        _var('q', 'm3', 'out', initial_value='0.0'),
        _var('R', 'Js_per_m6', 'in'),
        _var('C', 'm6_per_J', 'in'),
    ], [
        _eq('<apply><diff/><bvar><ci>t</ci></bvar><ci>q</ci></apply>',
            '<apply><minus/><ci>v_in</ci><apply><divide/><ci>u</ci><ci>R</ci></apply></apply>'),
        _eq('<ci>u</ci>', '<apply><divide/><ci>q</ci><ci>C</ci></apply>'),
    ])
    + _component('pressure_source_type', [
        _var('u', 'J_per_m3', 'out'),
        _var('v', 'm3_per_s', 'in'),
        _var('u_mean', 'J_per_m3', 'in'),
        _var('amp', 'dimensionless', 'in'),
        _var('omega', 'per_s', 'in'),
    ], [_eq('<ci>u</ci>', _sinusoid('u_mean', 'amp', 'omega'))])
    + _component('rl_sink_type', [
        _var('v', 'm3_per_s', 'out', initial_value='0.0'),
        _var('u_in', 'J_per_m3', 'in'),
        _var('R', 'Js_per_m6', 'in'),
        _var('L', 'Js2_per_m6', 'in'),
        _var('u_seen', 'J_per_m3', 'out'),
    ], [_eq('<apply><diff/><bvar><ci>t</ci></bvar><ci>v</ci></apply>',
            '<apply><divide/><apply><minus/><ci>u_in</ci><apply><times/><ci>R</ci><ci>v</ci>'
            '</apply></apply><ci>L</ci></apply>'),
        _eq('<ci>u_seen</ci>', '<ci>u_in</ci>')])
    + _component('volume_store_type', [
        _var('q', 'm3', 'out'),
        _var('q_mean', 'm3', 'in'),
        _var('amp', 'dimensionless', 'in'),
        _var('omega', 'per_s', 'in'),
    ], [_eq('<ci>q</ci>', _sinusoid('q_mean', 'amp', 'omega'))])
    + _component('volume_reader_type', [
        _var('q_vc', 'litre', 'in'),
        _var('q_seen', 'litre', 'out'),
    ], [_eq('<ci>q_seen</ci>', '<ci>q_vc</ci>')])
    + _component('volume_reader_q_type', [
        _var('q', 'litre', 'in'),
        _var('q_seen', 'litre', 'out'),
    ], [_eq('<ci>q_seen</ci>', '<ci>q</ci>')])
    + _component('temperature_source_type', [
        _var('T', 'kelvin', 'out'),
        _var('T_set', 'kelvin', 'in'),
    ], [_eq('<ci>T</ci>', '<ci>T_set</ci>')])
    + _component('temperature_reader_type', [
        _var('T', 'celsius', 'in'),
        _var('T_seen', 'celsius', 'out'),
    ], [_eq('<ci>T_seen</ci>', '<ci>T</ci>')])
    + _component('uptake_source_type', [
        _var('v_up', 'm3_per_s', 'out'),
        _var('v_mean', 'm3_per_s', 'in'),
        _var('amp', 'dimensionless', 'in'),
        _var('omega', 'per_s', 'in'),
    ], [_eq('<ci>v_up</ci>', _sinusoid('v_mean', 'amp', 'omega'))])
    + _component('venous_uptake_type', [
        _var('v_in', 'm3_per_s', 'in'),
        _var('v_uptake', 'm3_per_s', 'in'),
        _var('u', 'J_per_m3', 'out'),
        _var('v_out', 'm3_per_s', 'out'),
        _var('u_d', 'J_per_m3', 'in'),
    ], [_eq('<ci>v_out</ci>', '<apply><plus/><ci>v_in</ci><ci>v_uptake</ci></apply>'),
        _eq('<ci>u</ci>', '<ci>u_d</ci>')])
    + '</model>\n'
)


def _port(port_type, variables, multi_port=None):
    port = {"port_type": port_type, "variables": variables}
    if multi_port is not None:
        port["multi_port"] = multi_port
    return port


def _module(module_type, module_subtype, component_type, entrance_ports, exit_ports, variables_and_units):
    return {
        "module_type": module_type,
        "module_subtype": module_subtype,
        "module_format": "cellml",
        "component_file": "multiport_test_modules.cellml",
        "component_type": component_type,
        "entrance_ports": entrance_ports,
        "exit_ports": exit_ports,
        "general_ports": [],
        "variables_and_units": variables_and_units,
    }


def _source_config(module_type, component_type, flow_units):
    return _module(module_type, "nn", component_type, [], [_port("vessel_port", ["v", "u_out"])], [
        ["v", flow_units, "access", "variable"],
        ["u_out", "J_per_m3", "access", "boundary_condition"],
        ["v_mean", flow_units, "access", "constant"],
        ["amp", "dimensionless", "access", "constant"],
        ["omega", "per_s", "access", "constant"],
        ["u_seen", "J_per_m3", "access", "variable"],
    ])


MERGE_VARIABLES = [
    ["v_in", "m3_per_s", "access", "boundary_condition"],
    ["u", "J_per_m3", "access", "variable"],
    ["v_out", "m3_per_s", "access", "variable"],
    ["u_d", "J_per_m3", "access", "boundary_condition"],
]

MODULES_CONFIG = [
    _source_config("flow_source", "flow_source_type", "m3_per_s"),
    _source_config("flow_source_mm3", "flow_source_mm3_type", "mm3_per_s"),
    _source_config("bad_source", "bad_source_type", "J_per_m3"),
    _module("flow_merge", "vp", "flow_merge_type",
            [_port("vessel_port", ["v_in", "u"], ["sum", "True"])],
            [_port("vessel_port", ["v_out", "u_d"])], MERGE_VARIABLES),
    # the same node without a multi_port: a plain one-to-one entrance port
    _module("flow_merge_plain", "vp", "flow_merge_type",
            [_port("vessel_port", ["v_in", "u"])],
            [_port("vessel_port", ["v_out", "u_d"])], MERGE_VARIABLES),
    # malformed: two variables, one multi_port entry
    _module("flow_merge_bad_list", "vp", "flow_merge_type",
            [_port("vessel_port", ["v_in", "u"], ["sum"])],
            [_port("vessel_port", ["v_out", "u_d"])], MERGE_VARIABLES),
    _module("windkessel", "nn", "windkessel_type", [_port("vessel_port", ["v_in", "u"])], [], [
        ["v_in", "m3_per_s", "access", "boundary_condition"],
        ["u", "J_per_m3", "access", "variable"],
        ["q", "m3", "access", "variable"],
        ["R", "Js_per_m6", "access", "constant"],
        ["C", "m6_per_J", "access", "constant"],
    ]),
    _module("pressure_source", "nn", "pressure_source_type", [],
            [_port("vessel_port", ["v", "u"])], [
                ["u", "J_per_m3", "access", "variable"],
                ["v", "m3_per_s", "access", "boundary_condition"],
                ["u_mean", "J_per_m3", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("flow_split", "pv", "flow_split_type",
            [_port("vessel_port", ["v_in", "u"])],
            [_port("vessel_port", ["v_out", "u_d"], ["sum", "True"])], [
                ["v_in", "m3_per_s", "access", "variable"],
                ["u", "J_per_m3", "access", "boundary_condition"],
                ["v_out", "m3_per_s", "access", "boundary_condition"],
                ["u_d", "J_per_m3", "access", "variable"],
            ]),
    _module("rl_sink", "nn", "rl_sink_type", [_port("vessel_port", ["v", "u_in"])], [], [
        ["v", "m3_per_s", "access", "variable"],
        ["u_in", "J_per_m3", "access", "boundary_condition"],
        ["R", "Js_per_m6", "access", "constant"],
        ["L", "Js2_per_m6", "access", "constant"],
        ["u_seen", "J_per_m3", "access", "variable"],
    ]),
    _module("volume_store", "nn", "volume_store_type", [],
            [_port("volume_link_port", ["q"], "True")], [
                ["q", "m3", "access", "variable"],
                ["q_mean", "m3", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("volume_reader", "nn", "volume_reader_type",
            [_port("volume_link_port", ["q_vc"])], [], [
                ["q_vc", "litre", "access", "boundary_condition"],
                ["q_seen", "litre", "access", "variable"],
            ]),
    _module("volume_reader_q", "nn", "volume_reader_q_type",
            [_port("volume_link_port", ["q"])], [], [
                ["q", "litre", "access", "boundary_condition"],
                ["q_seen", "litre", "access", "variable"],
            ]),
    _module("volume_reader_up", "nn", "volume_reader_type",
            [], [_port("volume_link_port", ["q_vc"])], [
                ["q_vc", "litre", "access", "boundary_condition"],
                ["q_seen", "litre", "access", "variable"],
            ]),
    _module("volume_store_down", "nn", "volume_store_type",
            [_port("volume_link_port", ["q"])], [], [
                ["q", "m3", "access", "variable"],
                ["q_mean", "m3", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("temperature_source", "nn", "temperature_source_type", [],
            [_port("temperature_port", ["T"])], [
                ["T", "kelvin", "access", "variable"],
                ["T_set", "kelvin", "access", "constant"],
            ]),
    _module("temperature_reader", "nn", "temperature_reader_type",
            [_port("temperature_port", ["T"])], [], [
                ["T", "celsius", "access", "boundary_condition"],
                ["T_seen", "celsius", "access", "variable"],
            ]),
    _module("flow_source_one", "nn", "flow_source_type", [], [_port("vessel_port", ["v"])],
            _source_config("x", "flow_source_type", "m3_per_s")["variables_and_units"]),
    _module("windkessel_one", "nn", "windkessel_type", [_port("vessel_port", ["v_in"])], [], [
        ["v_in", "m3_per_s", "access", "boundary_condition"],
        ["u", "J_per_m3", "access", "variable"],
        ["q", "m3", "access", "variable"],
        ["R", "Js_per_m6", "access", "constant"],
        ["C", "m6_per_J", "access", "constant"],
    ]),
    _module("windkessel_blank", "nn", "windkessel_type", [_port("vessel_port", ["v_in", ""])], [], [
        ["v_in", "m3_per_s", "access", "boundary_condition"],
        ["u", "J_per_m3", "access", "variable"],
        ["q", "m3", "access", "variable"],
        ["R", "Js_per_m6", "access", "constant"],
        ["C", "m6_per_J", "access", "constant"],
    ]),
    _module("uptake_source", "nn", "uptake_source_type", [],
            [_port("blood_uptake_port", ["v_up"])], [
                ["v_up", "m3_per_s", "access", "variable"],
                ["v_mean", "m3_per_s", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("uptake_source_two_ports", "nn", "uptake_source_type", [],
            [_port("blood_uptake_port", ["v_up"]), _port("blood_uptake_port", ["v_up"])], [
                ["v_up", "m3_per_s", "access", "variable"],
                ["v_mean", "m3_per_s", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("uptake_source_shared", "nn", "uptake_source_type", [],
            [_port("blood_uptake_port", ["v_up"], "True")], [
                ["v_up", "m3_per_s", "access", "variable"],
                ["v_mean", "m3_per_s", "access", "constant"],
                ["amp", "dimensionless", "access", "constant"],
                ["omega", "per_s", "access", "constant"],
            ]),
    _module("venous_uptake", "vp", "venous_uptake_type",
            [_port("vessel_port", ["v_in", "u"]), _port("blood_uptake_port", ["v_uptake"], "sum")],
            [_port("vessel_port", ["v_out", "u_d"])], [
                ["v_in", "m3_per_s", "access", "boundary_condition"],
                ["v_uptake", "m3_per_s", "access", "boundary_condition"],
                ["u", "J_per_m3", "access", "variable"],
                ["v_out", "m3_per_s", "access", "variable"],
                ["u_d", "J_per_m3", "access", "boundary_condition"],
            ]),
]


@pytest.fixture(scope="module")
def external_modules_dir(tmp_path_factory):
    modules_dir = tmp_path_factory.mktemp("multiport_modules")
    (modules_dir / "multiport_test_modules.cellml").write_text(MODULES_CELLML)
    (modules_dir / "multiport_test_modules_config.json").write_text(
        json.dumps(MODULES_CONFIG, indent=2))
    return str(modules_dir)


# --------------------------------------------------------------------------------------------
# model building helpers
# --------------------------------------------------------------------------------------------

# parameters for every fixture/built-in module used below; the generator keeps only the ones
# the module array needs
def _source_params(name, v_mean, amp, omega, units="m3_per_s"):
    return [(f"{name}/v_mean", units, v_mean), (f"{name}/amp", "dimensionless", amp),
            (f"{name}/omega", "per_s", omega)]


def _arterial_params(name):
    return [(f"{name}/R", "Js_per_m6", 1.0e7), (f"{name}/C", "m6_per_J", 1.0e-9),
            (f"{name}/I", "Js2_per_m6", 1.0e5), (f"{name}/q_0", "m3", 1.0e-5),
            (f"{name}/u_0", "J_per_m3", 0.0), (f"{name}/u_ext", "J_per_m3", 0.0),
            (f"{name}/u_out", "J_per_m3", 1000.0)]


def _windkessel_params(name):
    return [(f"{name}/R", "Js_per_m6", 1.0e8), (f"{name}/C", "m6_per_J", 1.0e-8)]


def _rl_params(name, R):
    return [(f"{name}/R", "Js_per_m6", R), (f"{name}/L", "Js2_per_m6", 1.0e6)]


def _generate(tmp_path, external_modules_dir, prefix, vessel_rows, params, model_type="cellml"):
    """Write the module array and parameters for ``vessel_rows`` and generate the model."""
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    lines = ["name,module_subtype,module_type,inp_instances,out_instances"]
    for name, module_subtype, module_type, inp, out in vessel_rows:
        lines.append(f"{name},{module_subtype},{module_type},{' '.join(inp)},{' '.join(out)}")
    (resources_dir / f"{prefix}_module_array.csv").write_text("\n".join(lines) + "\n")
    param_lines = ["variable_name,units,value,data_reference"]
    for variable_name, units, value in params:
        param_lines.append(f"{variable_name},{units},{value},test")
    from _parameters import write_parameters
    write_parameters(resources_dir / f"{prefix}_parameters.json", "\n".join(param_lines) + "\n")

    generated_dir = tmp_path / "generated_models"
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.json',
        'model_type': model_type,
        'solver': 'CVODE_myokit' if model_type == 'cellml' else 'solve_ivp',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(generated_dir),
        'external_modules_dir': external_modules_dir,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config), f"generation of {prefix} failed"
    return os.path.join(str(generated_dir), prefix, f"{prefix}.cellml")


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


def _connections(cellml_text):
    """{(component_1, component_2): [(variable_1, variable_2), ...]} of a generated model."""
    out = {}
    for block in re.findall(r'<connection>(.*?)</connection>', cellml_text, re.S):
        comps = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block).groups()
        out[comps] = re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block)
    return out


def _mapped(connections, comp_a, var_a, comp_b, var_b):
    """Whether comp_a.var_a and comp_b.var_b are mapped to each other (either orientation)."""
    return ((var_a, var_b) in connections.get((comp_a, comp_b), []) or
            (var_b, var_a) in connections.get((comp_b, comp_a), []))


# --------------------------------------------------------------------------------------------
# helper / validation unit tests
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_list_multi_port_helpers():
    list_port = _port("vessel_port", ["v_in", "u"], ["sum", True])
    assert list_multi_port(list_port) == ["sum", "True"]
    assert is_multi_port(list_port)
    for value in ["True", True]:
        assert list_multi_port(_port("vessel_port", ["v"], value)) is None
        assert is_multi_port(_port("vessel_port", ["v"], value))
    # a raw (unnormalised) "sum" string is not a many-to-one flag: normalisation makes it a list
    assert not is_multi_port(_port("volume_port", ["q"], "sum"))
    assert not is_multi_port(_port("vessel_port", ["v"]))
    assert module_has_list_multi_port(pd.Series(
        {"entrance_ports": [list_port], "exit_ports": [], "general_ports": []}))
    assert not module_has_list_multi_port(pd.Series(
        {"entrance_ports": [_port("vessel_port", ["v"], "True")], "exit_ports": [],
         "general_ports": []}))


@pytest.mark.unit
@pytest.mark.parametrize("multi_port, message", [
    (["sum"], "one entry per port variable"),
    (["sum", "True", "True"], "one entry per port variable"),
    (["sum", "average"], 'must be "sum", "True" or "multiply"'),
])
def test_malformed_list_multi_port_is_rejected(multi_port, message):
    with pytest.raises(ValueError, match=message):
        validate_list_multi_port(_port("vessel_port", ["v_in", "u"], multi_port), "module m")
    validate_list_multi_port(_port("vessel_port", ["v_in", "u"], ["sum", "True"]), "module m")
    validate_list_multi_port(_port("vessel_port", ["v_in", "u"], "True"), "module m")


@pytest.mark.integration
def test_malformed_list_multi_port_fails_generation(tmp_path, external_modules_dir):
    rows = [("src_a", "nn", "flow_source", [], ["merge"]),
            ("merge", "vp", "flow_merge_bad_list", ["src_a"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = _source_params("src_a", 1e-5, 0.5, 6.0) + _windkessel_params("wk")
    with pytest.raises(ValueError, match="one entry per port variable"):
        _generate(tmp_path, external_modules_dir, "mp_bad_list", rows, params)


@pytest.mark.unit
def test_cpp_generator_refuses_list_multi_port_next_to_a_1d_vessel(tmp_path):
    """The 0D-1D coupling reads the ports of a module next to a 1D vessel from its config, so
    such a module cannot sum or multiply; a sum elsewhere (a volume sum) is ordinary CellML and
    gets past this check."""
    from libcuflynx.generators.cpp.generator import CVS0DCppGenerator

    def _model(merge_neighbours):
        class _Model:
            vessels_df = pd.DataFrame([
                {"name": "merge", "module_type": "flow_merge",
                 "inp_instances": merge_neighbours, "out_instances": [],
                 "entrance_ports": [_port("vessel_port", ["v_in", "u"], ["sum", "True"])],
                 "exit_ports": [], "general_ports": []},
                {"name": "FV1D_0", "module_type": "FV1D_vessel",
                 "inp_instances": [], "out_instances": [],
                 "entrance_ports": [], "exit_ports": [], "general_ports": []}])
        return _Model()

    with pytest.raises(NotImplementedError, match=r"\['merge'\]: a module next to a 1D vessel"):
        CVS0DCppGenerator(_model(["FV1D_0"]), str(tmp_path / "gen"), "mp", couple_to_1d=True,
                          cpp_generated_models_dir=str(tmp_path / "gen_cpp"))
    try:
        CVS0DCppGenerator(_model([]), str(tmp_path / "gen2"), "mp", couple_to_1d=True,
                          cpp_generated_models_dir=str(tmp_path / "gen_cpp2"))
    except NotImplementedError as e:
        pytest.fail(f"a sum away from the 1D vessels was refused: {e}")
    except Exception:
        pass  # the stub model is not enough for the rest of the constructor


# --------------------------------------------------------------------------------------------
# flow_merge: "sum" on an entrance port, "True" shares the node pressure upstream
# --------------------------------------------------------------------------------------------

MERGE_SOURCES = {
    "src_a": (1.0e-5, 0.5, 6.0),
    "src_b": (2.0e-5, 0.3, 4.0),
    "src_c": (0.5e-5, 0.8, 9.0),
}


@pytest.mark.integration
def test_flow_merge_two_upstream_sources(tmp_path, external_modules_dir):
    rows = [("src_a", "nn", "flow_source", [], ["merge"]),
            ("src_b", "nn", "flow_source", [], ["merge"]),
            ("merge", "vp", "flow_merge", ["src_a", "src_b"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = (_source_params("src_a", *MERGE_SOURCES["src_a"]) +
              _source_params("src_b", *MERGE_SOURCES["src_b"]) + _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_merge2", rows, params)

    text = _read(cellml_path)
    assert '<component name="multiport_sum_merge_v_in">' in text
    conns = _connections(text)
    for src in ["src_a", "src_b"]:
        assert _mapped(conns, f"{src}_module", "v", "multiport_sum_merge_v_in", f"v_{src}")
        # "True": the node pressure goes to every upstream module
        assert _mapped(conns, f"{src}_module", "u_out", "merge_module", "u")
        # and the flow is not also mapped one-to-one (it would over-determine v_in)
        assert not _mapped(conns, f"{src}_module", "v", "merge_module", "v_in")
    assert _mapped(conns, "multiport_sum_merge_v_in", "v_in", "merge_module", "v_in")

    res = _simulate(cellml_path, ["src_a/v", "src_b/v", "merge/v_out", "merge/u", "wk/u",
                                  "src_a/u_seen", "src_b/u_seen"])
    np.testing.assert_allclose(res["merge/v_out"], res["src_a/v"] + res["src_b/v"], **EXACT)
    assert np.ptp(res["merge/v_out"]) > 0
    np.testing.assert_allclose(res["merge/u"], res["wk/u"], **EXACT)
    assert np.max(np.abs(res["merge/u"])) > 0
    for src in ["src_a", "src_b"]:
        np.testing.assert_allclose(res[f"{src}/u_seen"], res["merge/u"], **EXACT)


@pytest.mark.integration
def test_flow_merge_three_upstream_vessels(tmp_path, external_modules_dir):
    """Two built-in arterial_simple (vp) vessels and one bare source flow into the node."""
    rows = [("src_a", "nn", "flow_source", [], ["art_a"]),
            ("art_a", "vp", "arterial_simple", ["src_a"], ["merge"]),
            ("src_b", "nn", "flow_source", [], ["art_b"]),
            ("art_b", "vp", "arterial_simple", ["src_b"], ["merge"]),
            ("src_c", "nn", "flow_source", [], ["merge"]),
            ("merge", "vp", "flow_merge", ["art_a", "art_b", "src_c"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = (_source_params("src_a", *MERGE_SOURCES["src_a"]) +
              _source_params("src_b", *MERGE_SOURCES["src_b"]) +
              _source_params("src_c", *MERGE_SOURCES["src_c"]) +
              _arterial_params("art_a") + _arterial_params("art_b") + _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_merge3", rows, params)

    conns = _connections(_read(cellml_path))
    for vessel in ["art_a", "art_b"]:
        assert _mapped(conns, f"{vessel}_module", "v", "multiport_sum_merge_v_in", f"v_{vessel}")
        assert _mapped(conns, f"{vessel}_module", "u_out", "merge_module", "u")
    assert _mapped(conns, "src_c_module", "v", "multiport_sum_merge_v_in", "v_src_c")
    assert _mapped(conns, "src_c_module", "u_out", "merge_module", "u")
    # the one-to-one connections upstream of the vessels are untouched
    assert _mapped(conns, "src_a_module", "v", "art_a_module", "v_in")
    assert _mapped(conns, "src_a_module", "u_out", "art_a_module", "u")

    res = _simulate(cellml_path, ["art_a/v", "art_b/v", "src_c/v", "merge/v_out", "merge/u",
                                  "wk/u", "src_c/u_seen"])
    np.testing.assert_allclose(res["merge/v_out"],
                               res["art_a/v"] + res["art_b/v"] + res["src_c/v"], **EXACT)
    assert np.max(np.abs(res["art_a/v"])) > 0 and np.max(np.abs(res["art_b/v"])) > 0
    np.testing.assert_allclose(res["merge/u"], res["wk/u"], **EXACT)
    np.testing.assert_allclose(res["src_c/u_seen"], res["merge/u"], **EXACT)


@pytest.mark.integration
def test_flow_merge_takes_a_terminal_inflow(tmp_path, external_modules_dir):
    """A built-in terminal upstream of the node is summed by the node, not by the legacy
    terminal_venous_connection (which would define the node's v_in a second time)."""
    rows = [("psrc", "nn", "pressure_source", [], ["term"]),
            ("term", "pp", "terminal", ["psrc"], ["merge"]),
            ("src_a", "nn", "flow_source", [], ["merge"]),
            ("merge", "vp", "flow_merge", ["term", "src_a"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = ([("psrc/u_mean", "J_per_m3", 1.0e4), ("psrc/amp", "dimensionless", 0.5),
               ("psrc/omega", "per_s", 6.0),
               ("term/R_T", "Js_per_m6", 1.0e8), ("term/C_T", "m6_per_J", 1.0e-9),
               ("term/u_ext", "J_per_m3", 0.0), ("term/q_us", "m3", 0.0),
               ("term/q_init", "m3", 1.0e-6)] +
              _source_params("src_a", *MERGE_SOURCES["src_a"]) + _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_term", rows, params)

    conns = _connections(_read(cellml_path))
    assert _mapped(conns, "term_module", "v_T", "multiport_sum_merge_v_in", "v_T_term")
    assert _mapped(conns, "term_module", "u_out", "merge_module", "u")
    assert not any(pair[1] == "v_in" or pair[0] == "v_in"
                   for pair in conns.get(("terminal_venous_connection", "merge_module"), []))

    res = _simulate(cellml_path, ["term/v_T", "src_a/v", "merge/v_out", "merge/u", "wk/u"])
    np.testing.assert_allclose(res["merge/v_out"], res["term/v_T"] + res["src_a/v"], **EXACT)
    assert np.max(np.abs(res["term/v_T"])) > 0
    np.testing.assert_allclose(res["merge/u"], res["wk/u"], **EXACT)


def _terminal_params(name):
    return [(name + "/R_T", "Js_per_m6", 1.0e8), (name + "/C_T", "m6_per_J", 1.0e-9),
            (name + "/u_ext", "J_per_m3", 0.0), (name + "/q_us", "m3", 0.0),
            (name + "/q_init", "m3", 1.0e-6)]


PSRC_PARAMS = [("psrc/u_mean", "J_per_m3", 1.0e4), ("psrc/amp", "dimensionless", 0.5),
               ("psrc/omega", "per_s", 6.0)]


@pytest.mark.integration
def test_terminal_inflow_with_separate_list_form_uptake_port(tmp_path, external_modules_dir):
    """A venous module whose vessel_port is plain still takes the terminal flow through the
    terminal_venous_connection when another of its entrance ports (here a blood_uptake_port
    with multi_port "sum", which is read as the list form ["sum"]) sums other inflows.

    Before the fix, any list-form entrance port skipped the terminal -> venous v_in mapping,
    so v_in was left unconnected and the model did not load."""
    rows = [("psrc", "nn", "pressure_source", [], ["term"]),
            ("term", "pp", "terminal", ["psrc"], ["ven"]),
            ("up_a", "nn", "uptake_source", [], ["ven"]),
            ("up_b", "nn", "uptake_source", [], ["ven"]),
            ("ven", "vp", "venous_uptake", ["term", "up_a", "up_b"], ["wk"]),
            ("wk", "nn", "windkessel", ["ven"], [])]
    params = (PSRC_PARAMS + _terminal_params("term") +
              [("up_a/v_mean", "m3_per_s", 1.0e-6), ("up_a/amp", "dimensionless", 0.5),
               ("up_a/omega", "per_s", 6.0),
               ("up_b/v_mean", "m3_per_s", 2.0e-6), ("up_b/amp", "dimensionless", 0.3),
               ("up_b/omega", "per_s", 4.0)] +
              _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_term_uptake", rows, params)

    conns = _connections(_read(cellml_path))
    # the terminal flow reaches v_in through the terminal_venous_connection
    assert _mapped(conns, "term_module", "v_T", "terminal_venous_connection", "v_term")
    assert _mapped(conns, "terminal_venous_connection", "v_ven", "ven_module", "v_in")
    assert _mapped(conns, "term_module", "u_out", "ven_module", "u")
    # the uptake port sums its two neighbours
    for src in ["up_a", "up_b"]:
        assert _mapped(conns, f"{src}_module", "v_up", "multiport_sum_ven_v_uptake", f"v_up_{src}")
    assert _mapped(conns, "multiport_sum_ven_v_uptake", "v_uptake", "ven_module", "v_uptake")

    res = _simulate(cellml_path, ["term/v_T", "up_a/v_up", "up_b/v_up", "ven/v_out", "ven/u",
                                  "wk/u"])
    np.testing.assert_allclose(res["ven/v_out"],
                               res["term/v_T"] + res["up_a/v_up"] + res["up_b/v_up"], **EXACT)
    assert np.max(np.abs(res["term/v_T"])) > 0
    np.testing.assert_allclose(res["ven/u"], res["wk/u"], **EXACT)


@pytest.mark.integration
def test_fan_out_to_two_modules_in_other_units_maps_directly(tmp_path, external_modules_dir):
    """One output (q, m3) shared through a multi_port "True" port with two modules that both
    read it in litre: each is mapped to it directly, and the solver converts the units."""
    rows = [("store", "nn", "volume_store", [], ["reader_a", "reader_b"]),
            ("reader_a", "nn", "volume_reader", ["store"], []),
            ("reader_b", "nn", "volume_reader", ["store"], [])]
    params = [("store/q_mean", "m3", 2.0e-3), ("store/amp", "dimensionless", 0.5),
              ("store/omega", "per_s", 6.0)]
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_fan_units", rows, params)

    text = _read(cellml_path)
    assert "unit_converter" not in text
    conns = _connections(text)
    for reader in ["reader_a", "reader_b"]:
        assert _mapped(conns, "store_module", "q", f"{reader}_module", "q_vc")

    res = _simulate(cellml_path, ["store/q", "reader_a/q_seen", "reader_b/q_seen"])
    assert np.ptp(res["store/q"]) > 0
    for reader in ["reader_a", "reader_b"]:
        np.testing.assert_allclose(res[f"{reader}/q_seen"], 1.0e3 * res["store/q"], rtol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("rows", [
    # the downstream module computes the variable the upstream one reads
    [("reader", "nn", "volume_reader_up", [], ["store"]),
     ("store", "nn", "volume_store_down", ["reader"], [])],
    # both variables are called q
    [("store", "nn", "volume_store", [], ["reader"]),
     ("reader", "nn", "volume_reader_q", ["store"], [])],
])
def test_a_connection_in_other_units_converts_in_the_direction_the_value_flows(
        tmp_path, external_modules_dir, rows):
    params = [("store/q_mean", "m3", 2.0e-3), ("store/amp", "dimensionless", 0.5),
              ("store/omega", "per_s", 6.0)]
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_units_dir", rows, params)
    res = _simulate(cellml_path, ["store/q", "reader/q_seen"])
    assert np.ptp(res["store/q"]) > 0
    np.testing.assert_allclose(res["reader/q_seen"], 1.0e3 * res["store/q"], rtol=1e-12)


@pytest.mark.integration
def test_a_connection_between_affine_units_is_rejected(tmp_path, external_modules_dir):
    rows = [("src", "nn", "temperature_source", [], ["reader"]),
            ("reader", "nn", "temperature_reader", ["src"], [])]
    with pytest.raises(ValueError, match='Cannot map "T".*needs an offset'):
        _generate(tmp_path, external_modules_dir, "mp_affine", rows,
                  [("src/T_set", "kelvin", 310.0)])


@pytest.mark.integration
@pytest.mark.parametrize("rows", [
    [("src", "nn", "flow_source", [], ["wk"]), ("wk", "nn", "windkessel_one", ["src"], [])],
    [("src", "nn", "flow_source_one", [], ["wk"]), ("wk", "nn", "windkessel", ["src"], [])],
])
def test_a_plain_port_pair_with_different_variable_counts_is_rejected(
        tmp_path, external_modules_dir, rows):
    params = (_source_params("src", *MERGE_SOURCES["src_a"]) + [("src/u_out", "J_per_m3", 0.0)] +
              _windkessel_params("wk"))
    with pytest.raises(ValueError, match="needs the same number of variables on both sides"):
        _generate(tmp_path, external_modules_dir, "mp_counts", rows, params)


@pytest.mark.integration
def test_a_bc_with_a_blank_partner_keeps_its_value(tmp_path, external_modules_dir):
    """flow_source [v, u_out] into [v_in, ""]: u_out is not connected, so it keeps its value."""
    rows = [("src", "nn", "flow_source", [], ["wk"]),
            ("wk", "nn", "windkessel_blank", ["src"], [])]
    params = (_source_params("src", *MERGE_SOURCES["src_a"]) + [("src/u_out", "J_per_m3", 1234.0)] +
              _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_blank", rows, params)
    res = _simulate(cellml_path, ["src/u_seen"], sim_time=0.1)
    np.testing.assert_allclose(res["src/u_seen"], 1234.0, rtol=1e-12)


@pytest.mark.integration
def test_flow_merge_one_neighbour_matches_one_to_one(tmp_path, external_modules_dir):
    params = _source_params("src_a", *MERGE_SOURCES["src_a"]) + _windkessel_params("wk")
    names = ["src_a/v", "merge/v_out", "merge/u", "wk/u", "wk/q", "src_a/u_seen"]
    results = {}
    for module_type in ["flow_merge", "flow_merge_plain"]:
        rows = [("src_a", "nn", "flow_source", [], ["merge"]),
                ("merge", "vp", module_type, ["src_a"], ["wk"]),
                ("wk", "nn", "windkessel", ["merge"], [])]
        cellml_path = _generate(tmp_path / module_type, external_modules_dir,
                                f"mp_one_{module_type}", rows, params)
        results[module_type] = _simulate(cellml_path, names)
    for name in names:
        np.testing.assert_allclose(results["flow_merge"][name], results["flow_merge_plain"][name],
                                   rtol=1e-12, atol=1e-30, err_msg=name)
    np.testing.assert_allclose(results["flow_merge"]["merge/v_out"],
                               results["flow_merge"]["src_a/v"], **EXACT)


@pytest.mark.integration
def test_flow_merge_without_neighbours_takes_v_in_from_the_parameters(tmp_path,
                                                                      external_modules_dir):
    """An unconnected "sum" is a boundary condition: a constant valued from the parameters."""
    rows = [("merge", "vp", "flow_merge", [], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_none", rows,
                            _windkessel_params("wk") + [("merge/v_in", "m3_per_s", 3.0e-6)])
    assert "multiport_sum_merge_v_in" not in _read(cellml_path)

    parameters = model_parameter_rows(
        os.path.join(os.path.dirname(cellml_path), "mp_none_parameters.json"))
    assert "v_in_merge" in {r["variable_name"] for r in parameters}
    res = _simulate(cellml_path, ["merge/v_out"], sim_time=0.5)
    np.testing.assert_array_equal(res["merge/v_out"], 3.0e-6)


@pytest.mark.integration
def test_flow_merge_without_neighbours_or_a_v_in_value_is_a_missing_parameter(
        tmp_path, external_modules_dir, capsys):
    """With no value for the unconnected "sum", generation fails with the report every unset
    boundary condition gets."""
    rows = [("merge", "vp", "flow_merge", [], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    capsys.readouterr()
    with pytest.raises(AssertionError, match="generation of mp_none failed"):
        _generate(tmp_path, external_modules_dir, "mp_none", rows, _windkessel_params("wk"))
    out = capsys.readouterr().out
    assert 'merge/v_in: a boundary condition of "merge" that no connected module' in out


@pytest.mark.integration
def test_an_unconnected_sum_that_is_not_a_boundary_condition_is_rejected(tmp_path,
                                                                         external_modules_dir):
    """A "sum" variable that is not a boundary_condition has nothing to give it a value when
    nothing is connected through its port."""
    config = [dict(module, module_type="flow_merge_var", variables_and_units=[
        ["v_in", "m3_per_s", "access", "variable"]] + MERGE_VARIABLES[1:])
        for module in MODULES_CONFIG if module["module_type"] == "flow_merge"]
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "multiport_test_modules.cellml").write_text(MODULES_CELLML)
    (modules_dir / "multiport_test_modules_config.json").write_text(
        json.dumps(MODULES_CONFIG + config, indent=2))
    rows = [("merge", "vp", "flow_merge_var", [], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    with pytest.raises(ValueError, match='"merge" variable "v_in" is a multi_port "sum".*'
                                         'make "v_in" a boundary_condition'):
        _generate(tmp_path, str(modules_dir), "mp_none_var", rows, _windkessel_params("wk"))


@pytest.mark.integration
@pytest.mark.parametrize("module_type, outputs", [
    # two ports of the matching type, both connected by one listing
    ("uptake_source_two_ports", ["ven"]),
    # a multi_port "True" port, listed twice
    ("uptake_source_shared", ["ven", "ven"]),
])
def test_a_sum_term_connected_twice_is_rejected(tmp_path, external_modules_dir, module_type,
                                                outputs):
    rows = [("src", "nn", "flow_source", [], ["ven"]),
            ("up", "nn", module_type, [], outputs),
            ("ven", "vp", "venous_uptake", ["src"] + ["up"] * len(outputs), ["wk"]),
            ("wk", "nn", "windkessel", ["ven"], [])]
    params = (_source_params("src", *MERGE_SOURCES["src_a"]) +
              [("up/v_mean", "m3_per_s", 1.0e-6), ("up/amp", "dimensionless", 0.5),
               ("up/omega", "per_s", 6.0)] + _windkessel_params("wk"))
    with pytest.raises(ValueError, match='"up" variable "v_up" reaches the multi_port "sum" '
                                         'variable "v_uptake" of "ven".*counted twice'):
        _generate(tmp_path, external_modules_dir, "mp_twice", rows, params)


@pytest.mark.integration
def test_a_multiply_term_connected_twice_is_rejected(tmp_path):
    """A neighbour reaching a "Multiply" target twice (here through two Multiply ports of the
    matching type, both connected by one listing) is refused like a "sum" term."""
    uptake_variables = next(module["variables_and_units"] for module in MODULES_CONFIG
                            if module["module_type"] == "uptake_source")
    venous_uptake = next(module for module in MODULES_CONFIG
                         if module["module_type"] == "venous_uptake")
    config = MODULES_CONFIG + [
        _module("uptake_source_multiply", "nn", "uptake_source_type", [],
                [_port("blood_uptake_port", ["v_up"], "Multiply")] * 2,
                uptake_variables),
        dict(venous_uptake, module_type="venous_uptake_plain", entrance_ports=[
            _port("vessel_port", ["v_in", "u"]), _port("blood_uptake_port", ["v_uptake"])]),
    ]
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "multiport_test_modules.cellml").write_text(MODULES_CELLML)
    (modules_dir / "multiport_test_modules_config.json").write_text(json.dumps(config, indent=2))
    rows = [("src", "nn", "flow_source", [], ["ven"]),
            ("up", "nn", "uptake_source_multiply", [], ["ven"]),
            ("ven", "vp", "venous_uptake_plain", ["src", "up"], ["wk"]),
            ("wk", "nn", "windkessel", ["ven"], [])]
    params = (_source_params("src", *MERGE_SOURCES["src_a"]) +
              [("up/v_mean", "m3_per_s", 1.0e-6), ("up/amp", "dimensionless", 0.5),
               ("up/omega", "per_s", 6.0)] + _windkessel_params("wk"))
    with pytest.raises(ValueError, match='"up" variable "v_up" reaches the multi_port "Multiply" '
                                         'variable "v_uptake" of "ven".*counted twice'):
        _generate(tmp_path, str(modules_dir), "mp_mul_twice", rows, params)


@pytest.mark.integration
def test_flow_merge_converts_neighbour_units(tmp_path, external_modules_dir):
    rows = [("src_a", "nn", "flow_source", [], ["merge"]),
            ("src_mm3", "nn", "flow_source_mm3", [], ["merge"]),
            ("merge", "vp", "flow_merge", ["src_a", "src_mm3"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = (_source_params("src_a", *MERGE_SOURCES["src_a"]) +
              _source_params("src_mm3", 2.0e4, 0.3, 4.0, units="mm3_per_s") +
              _windkessel_params("wk"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_units", rows, params)

    text = _read(cellml_path)
    component = re.search(r'<component name="multiport_sum_merge_v_in">.*?</component>', text,
                          re.S).group(0)
    assert 'name="v_src_mm3" public_interface="in" units="mm3_per_s"' in component
    assert 'name="v_in" public_interface="out" units="m3_per_s"' in component

    res = _simulate(cellml_path, ["src_a/v", "src_mm3/v", "merge/v_out"])
    np.testing.assert_allclose(res["merge/v_out"], res["src_a/v"] + 1e-9 * res["src_mm3/v"],
                               rtol=1e-12)
    # the mm3 source contributes the larger part, so a missed conversion could not pass
    assert np.max(1e-9 * res["src_mm3/v"]) > np.max(res["src_a/v"])


@pytest.mark.integration
def test_flow_merge_rejects_incompatible_units(tmp_path, external_modules_dir):
    rows = [("src_a", "nn", "flow_source", [], ["merge"]),
            ("src_bad", "nn", "bad_source", [], ["merge"]),
            ("merge", "vp", "flow_merge", ["src_a", "src_bad"], ["wk"]),
            ("wk", "nn", "windkessel", ["merge"], [])]
    params = (_source_params("src_a", *MERGE_SOURCES["src_a"]) +
              _source_params("src_bad", 1.0, 0.3, 4.0, units="J_per_m3") +
              _windkessel_params("wk"))
    with pytest.raises(ValueError, match='"v" of "src_bad" has units J_per_m3.*cannot be converted'):
        _generate(tmp_path, external_modules_dir, "mp_bad_units", rows, params)


# --------------------------------------------------------------------------------------------
# flow_split: "sum" on an exit port, "True" shares the node pressure downstream
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_flow_split_three_downstream_vessels(tmp_path, external_modules_dir):
    """Two R-L sinks and a built-in arterial_simple (pp) vessel take flow from the node."""
    rows = [("psrc", "nn", "pressure_source", [], ["split"]),
            ("split", "pv", "flow_split", ["psrc"], ["sink_a", "sink_b", "art_d"]),
            ("sink_a", "nn", "rl_sink", ["split"], []),
            ("sink_b", "nn", "rl_sink", ["split"], []),
            ("art_d", "pp", "arterial_simple", ["split"], [])]
    params = ([("psrc/u_mean", "J_per_m3", 1.0e4), ("psrc/amp", "dimensionless", 0.5),
               ("psrc/omega", "per_s", 6.0)] +
              _rl_params("sink_a", 1.0e8) + _rl_params("sink_b", 3.0e8) + _arterial_params("art_d"))
    cellml_path = _generate(tmp_path, external_modules_dir, "mp_split", rows, params)

    conns = _connections(_read(cellml_path))
    for sink in ["sink_a", "sink_b"]:
        assert _mapped(conns, f"{sink}_module", "v", "multiport_sum_split_v_out", f"v_{sink}")
        assert _mapped(conns, "split_module", "u_d", f"{sink}_module", "u_in")
    assert _mapped(conns, "art_d_module", "v", "multiport_sum_split_v_out", "v_art_d")
    assert _mapped(conns, "split_module", "u_d", "art_d_module", "u_in")
    assert _mapped(conns, "psrc_module", "v", "split_module", "v_in")

    res = _simulate(cellml_path, ["sink_a/v", "sink_b/v", "art_d/v", "split/v_in", "split/u_d",
                                  "psrc/u", "sink_a/u_seen", "sink_b/u_seen"])
    np.testing.assert_allclose(res["split/v_in"],
                               res["sink_a/v"] + res["sink_b/v"] + res["art_d/v"], **EXACT)
    assert np.max(np.abs(res["sink_a/v"])) > 0 and np.max(np.abs(res["art_d/v"])) > 0
    assert not np.allclose(res["sink_a/v"], res["sink_b/v"])
    np.testing.assert_allclose(res["split/u_d"], res["psrc/u"], **EXACT)
    for sink in ["sink_a", "sink_b"]:
        np.testing.assert_allclose(res[f"{sink}/u_seen"], res["split/u_d"], **EXACT)


@pytest.mark.integration
def test_flow_split_python_model(tmp_path, external_modules_dir):
    """The generated sum component also goes through PythonGenerator (libCellML Analyser)."""
    rows = [("psrc", "nn", "pressure_source", [], ["split"]),
            ("split", "pv", "flow_split", ["psrc"], ["sink_a", "sink_b"]),
            ("sink_a", "nn", "rl_sink", ["split"], []),
            ("sink_b", "nn", "rl_sink", ["split"], [])]
    params = ([("psrc/u_mean", "J_per_m3", 1.0e4), ("psrc/amp", "dimensionless", 0.5),
               ("psrc/omega", "per_s", 6.0)] +
              _rl_params("sink_a", 1.0e8) + _rl_params("sink_b", 3.0e8))
    _generate(tmp_path, external_modules_dir, "mp_split_py", rows, params, model_type="python")
    generated = os.listdir(tmp_path / "generated_models" / "mp_split_py")
    assert any(name.endswith(".py") for name in generated), generated
