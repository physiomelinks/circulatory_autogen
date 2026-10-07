"""SN cell-module physics: the soma conserves Na (every Na flux SN_soma computes enters dNai/dt),
and the Ca Nernst potentials have valence 2.

i_Na1_6 (Nav1.6) carried its Na flux j_Na1_6 = -i_Na1_6/F, computed exactly like j_Na, but it was
left out of the Na balance, so Nav1.6 moved charge without moving Na.
"""
import os
import xml.etree.ElementTree as ET

import pytest

import libcuflynx

CELL_MODULES = os.path.join(os.path.dirname(libcuflynx.__file__), 'generators', 'resources', 'cell_modules.cellml')
MATHML = '{http://www.w3.org/1998/Math/MathML}'


def _ns(root):
    return root.tag[:root.tag.index('}') + 1] if root.tag.startswith('{') else ''


def _component(name):
    root = ET.parse(CELL_MODULES).getroot()
    for comp in root.iter(f'{_ns(root)}component'):
        if comp.get('name') == name:
            return comp
    raise AssertionError(f'{name} not in {CELL_MODULES}')


def _rhs_names_of_derivative(comp, state):
    """The variable names in the right-hand side of d<state>/dt."""
    for eq in comp.iter(f'{MATHML}apply'):
        children = list(eq)
        if len(children) != 3 or children[0].tag != f'{MATHML}eq':
            continue
        lhs, rhs = children[1], children[2]
        if lhs.tag == f'{MATHML}apply' and lhs[0].tag == f'{MATHML}diff' and \
                (lhs.find(f'{MATHML}ci').text or '').strip() == state:
            return {(ci.text or '').strip() for ci in rhs.iter(f'{MATHML}ci')}
    raise AssertionError(f'no d{state}/dt equation')


@pytest.mark.unit
def test_every_na_flux_of_the_soma_enters_the_na_balance():
    comp = _component('SN_soma')
    na_fluxes = {v.get('name') for v in comp if v.tag.endswith('}variable')
                 if (v.get('name') or '').startswith('j_') and (v.get('name') or '').endswith(('Na', 'Na1_6'))
                 and v.get('units') == 'mol_per_s'}
    assert 'j_Na1_6' in na_fluxes
    missing = na_fluxes - _rhs_names_of_derivative(comp, 'Nai')
    assert not missing, f'Na fluxes computed but not in dNai/dt: {sorted(missing)}'


def _assignment(comp, name):
    for eq in comp.iter(f'{MATHML}apply'):
        children = list(eq)
        if len(children) == 3 and children[0].tag == f'{MATHML}eq' and children[1].tag == f'{MATHML}ci' \
                and (children[1].text or '').strip() == name:
            return children[2]
    raise AssertionError(f'no {name} = ... equation')


@pytest.mark.unit
@pytest.mark.parametrize('component', ['SN_varicosity', 'electric_potentials_Paci_2013'])
def test_the_ca_nernst_potential_has_valence_two(component):
    """E_Ca = RT/(zF) ln(Cao/Cai) with z = 2: the expression carries a factor 2 (or 0.5) next to F."""
    rhs = _assignment(_component(component), 'E_Ca')
    numbers = {float((cn.text or '').strip()) for cn in rhs.iter(f'{MATHML}cn')
               if (cn.text or '').strip().replace('.', '', 1).isdigit()}
    assert numbers & {2.0, 0.5}, f'{component}: E_Ca has no valence factor (numbers {sorted(numbers)})'
