"""The built-in efferent_respiratory_effector module: its config's variables are its CellML component's."""
import json
import xml.etree.ElementTree as ET
from importlib.resources import files

RESOURCES = files('libcuflynx.generators') / 'resources'
CELLML_NS = '{http://www.cellml.org/cellml/1.1#}'


def _entries():
    return json.loads((RESOURCES / 'control_modules_config.json').read_text())


def _entry(component_type):
    return next(e for e in _entries() if e['component_type'] == component_type)


def _component_variables(component_type):
    root = ET.fromstring((RESOURCES / 'control_modules.cellml').read_text())
    comp = next(c for c in root.iter(f'{CELLML_NS}component') if c.get('name') == component_type)
    return {v.get('name'): v.get('units') for v in comp.findall(f'{CELLML_NS}variable')}


def test_efferent_respiratory_effector_config_matches_its_cellml():
    entry = _entry('efferent_respiratory_effector_type')
    cellml = _component_variables('efferent_respiratory_effector_type')
    names = [row[0] for row in entry['variables_and_units']]
    # the muscle pressure the lung reads through respiratory_lung_muscle_pressure_port, and its clamps
    for name in ('Delta_P_mus', 'Delta_P_mus_c', 'Delta_P_mus_p', 'Delta_P_mus_max', 'Delta_P_mus_min'):
        assert cellml.get(name) == 'J_per_m3', (name, cellml.get(name))
    missing = [n for n in names if n not in cellml]
    assert not missing, f'config variables not in the CellML component: {missing}'


def test_no_control_module_lists_a_variable_twice():
    twice = {}
    for entry in _entries():
        names = [row[0] for row in entry['variables_and_units']]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            twice[entry['component_type']] = dup
    assert not twice, f'listed twice: {twice}'
