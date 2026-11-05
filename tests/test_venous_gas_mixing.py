"""The mixed venous O2 and CO2 contents weight each region's content by that region's own flow.

Each tissue_GE_simple gives its outflow ``v_out`` and the gas it carries, ``j = v C``, on its
gas_flux_port. A venous_gas_mixing module sums them over its multi_port "sum" entrance port
and gives ``C_mix = sum(j_i) / sum(v_i)`` to the venous gas_transport_simple. The regions below
are listed in different orders for the terminals and the gas-exchange modules, so a mix that
paired contents and flows by position would weight a content by another region's flow.
"""
import json
import os
import re

import numpy as np
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

# region: (terminal inlet pressure [J/m3] -> flow u_in / 1e8 m3/s, O2 consumption, CO2 production [m3/s])
REGIONS = {'a': (400.0, -1.8e-7, 1.44e-7),
           'b': (200.0, -5.0e-8, 4.0e-8),
           'c': (800.0, -3.0e-7, 2.4e-7)}
TERMINAL_ORDER = ['a', 'b', 'c']
GE_ORDER = ['c', 'a', 'b']


def _rows(mixing, ge_subtypes=None):
    """Module array rows: three regions mixed by ``mix`` into the venous gas transport ``vgt``,
    or with ``mixing`` False the gas-exchange modules connected straight to ``vgt``.
    ``ge_subtypes`` gives a region's tissue_GE_simple subtype (default nn)."""
    ge_subtypes = ge_subtypes or {}
    venous_inputs = ['mix'] if mixing else [f'g{r}' for r in GE_ORDER]
    rows = [(f't{r}', 'pp_I', 'terminal', [], [f'g{r}']) for r in TERMINAL_ORDER]
    rows += [(f'g{r}', ge_subtypes.get(r, 'nn'), 'tissue_GE_simple', [f't{r}'],
              ['mix' if mixing else 'vgt'])
             for r in GE_ORDER]
    if mixing:
        rows.append(('mix', 'nn', 'venous_gas_mixing', [f'g{r}' for r in GE_ORDER], ['vgt']))
    rows.append(('vgt', 'nn', 'gas_transport_simple', venous_inputs, []))
    return rows


def _params():
    params = []
    for r, (u_in, m_o2, m_co2) in REGIONS.items():
        params += [(f't{r}/R_1_T', 'Js_per_m6', 5.0e7), (f't{r}/R_2_T', 'Js_per_m6', 5.0e7),
                   (f't{r}/C_T', 'm6_per_J', 1.0e-9), (f't{r}/I_T', 'Js2_per_m6', 2.5e5),
                   (f't{r}/u_ext', 'J_per_m3', 0.0), (f't{r}/u_in', 'J_per_m3', u_in),
                   (f't{r}/u_out', 'J_per_m3', 0.0), (f't{r}/q_us', 'm3', 0.0),
                   (f't{r}/q_init', 'm3', 0.5e-9 * u_in),
                   (f'g{r}/C_O2_a', 'dimensionless', 0.199),
                   (f'g{r}/C_CO2_a', 'dimensionless', 0.485),
                   (f'g{r}/q_0', 'm3', 5.0e-5),
                   (f'g{r}/C_O2_p_init', 'dimensionless', 0.15),
                   (f'g{r}/C_CO2_p_init', 'dimensionless', 0.52),
                   (f'g{r}/M_O2', 'm3_per_s', m_o2), (f'g{r}/M_CO2', 'm3_per_s', m_co2)]
    # the gas transport's dissociation constants (Spencer 1979, as in control_parasymp)
    params += [('overcap_solubility', 'J_per_m3', 1.0e10), ('saturation_cap', 'dimensionless', 0.9999),
               ('k_1', 'J_per_m3', 1998.5), ('k_2', 'J_per_m3', 25917.4),
               ('h_1', 'dimensionless', 0.3836), ('h_2', 'dimensionless', 1.819),
               ('alpha_1', 'm3_per_J', 2.4e-4), ('alpha_2', 'm3_per_J', 4.19e-4),
               ('beta_1', 'm3_per_J', 6.21e-5), ('beta_2', 'm3_per_J', 2.44e-4),
               ('C_O2_sat', 'dimensionless', 0.20448), ('C_CO2_sat', 'dimensionless', 1.9564)]
    params += [('vgt/q_0', 'm3', 1.0e-3),
               ('vgt/C_O2_out_init', 'dimensionless', 0.15),
               ('vgt/C_CO2_out_init', 'dimensionless', 0.52)]
    return params


def _generate(tmp_path, prefix, mixing, ge_subtypes=None, extra_params=(), external_modules_dir=None):
    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir(parents=True, exist_ok=True)
    lines = ['name,module_subtype,module_type,inp_instances,out_instances']
    for name, module_subtype, module_type, inp, out in _rows(mixing, ge_subtypes):
        lines.append(f"{name},{module_subtype},{module_type},{' '.join(inp)},{' '.join(out)}")
    (resources_dir / f'{prefix}_module_array.csv').write_text('\n'.join(lines) + '\n')
    param_lines = ['variable_name,units,value,data_reference']
    param_lines += [f'{name},{units},{value},test'
                    for name, units, value in _params() + list(extra_params)]
    from _parameters import write_parameters
    write_parameters(resources_dir / f'{prefix}_parameters.json', '\n'.join(param_lines) + '\n')

    generated_dir = tmp_path / 'generated_models'
    config = {'file_prefix': prefix,
              'input_param_file': f'{prefix}_parameters.json',
              'model_type': 'cellml',
              'solver': 'CVODE_myokit',
              'resources_dir': str(resources_dir),
              'generated_models_dir': str(generated_dir),
              'DEBUG': False}
    if external_modules_dir:
        config['external_modules_dir'] = external_modules_dir
    assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    return os.path.join(str(generated_dir), prefix, f'{prefix}.cellml')


def _connections(cellml_path):
    """The (component, variable) pairs a generated model maps to each other."""
    with open(cellml_path) as f:
        text = f.read()
    pairs = set()
    for block in re.findall(r'<connection>(.*?)</connection>', text, re.S):
        comp_1, comp_2 = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block).groups()
        for var_1, var_2 in re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block):
            pairs.add(((comp_1, var_1), (comp_2, var_2)))
            pairs.add(((comp_2, var_2), (comp_1, var_1)))
    return pairs


def _flow_weighted_mix(cellml_path):
    """The mixed contents, and sum(v_i C_i) / sum(v_i) with each region's flow taken from its
    own terminal and its content from its own gas-exchange module."""
    names = (['mix/C_O2_mix', 'mix/C_CO2_mix', 'mix/v_mix'] + [f't{r}/v_T' for r in REGIONS] +
             [f'g{r}/C_O2_p' for r in REGIONS] + [f'g{r}/C_CO2_p' for r in REGIONS])
    sim = get_simulation_helper(model_path=cellml_path, model_type='cellml', solver='CVODE_myokit',
                                dt=0.01, sim_time=5.0, pre_time=0.0)
    assert sim.run(), 'simulation failed'
    res = {n: np.asarray(x, dtype=float)
           for n, x in zip(names, sim.get_results(names, flatten=True))}
    v_total = sum(res[f't{r}/v_T'] for r in REGIONS)
    with np.errstate(invalid='ignore', divide='ignore'):  # no flow at t = 0
        expected = {gas: sum(res[f't{r}/v_T'] * res[f'g{r}/C_{gas}_p'] for r in REGIONS) / v_total
                    for gas in ('O2', 'CO2')}
    return res, v_total, expected


@pytest.mark.integration
def test_each_region_content_is_weighted_by_its_own_flow(tmp_path):
    cellml_path = _generate(tmp_path, 'venous_mix', mixing=True)
    # the venous gas transport takes the mix and its total flow
    connections = _connections(cellml_path)
    for mixed, transported in [('C_O2_mix', 'C_O2_in'), ('C_CO2_mix', 'C_CO2_in'), ('v_mix', 'v')]:
        assert (('mix_module', mixed), ('vgt_module', transported)) in connections

    res, v_total, expected = _flow_weighted_mix(cellml_path)

    # the regions' flows and contents differ, so a content weighted by another region's flow
    # would give another mix
    assert np.ptp([res[f't{r}/v_T'][-1] for r in REGIONS]) > 1e-6
    assert np.ptp([res[f'g{r}/C_O2_p'][-1] for r in REGIONS]) > 1e-3
    np.testing.assert_allclose(res['mix/v_mix'], v_total, rtol=1e-12, atol=0.0)
    flowing = np.abs(v_total) > 1e-12
    assert flowing[1:].all()
    np.testing.assert_allclose(res['mix/C_O2_mix'][flowing], expected['O2'][flowing], rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(res['mix/C_CO2_mix'][flowing], expected['CO2'][flowing], rtol=1e-10,
                               atol=0.0)


@pytest.mark.integration
def test_gas_exchange_straight_into_gas_transport_is_refused(tmp_path):
    """Without a venous_gas_mixing module nothing forms the mix, so generation stops and names
    the module to add."""
    with pytest.raises(ValueError, match='venous_gas_mixing'):
        _generate(tmp_path, 'venous_unmixed', mixing=False)


def _no_flux_port_ge_modules(tmp_path):
    """An external modules dir with tissue_GE_simple subtype ``no_flux``: the built-in nn
    without its gas_flux_port, as tissue_GE_simple/Ursino2001_v01 in the module library."""
    from libcuflynx.utilities.package_resources import builtin_modules_dir
    builtin = builtin_modules_dir()
    with open(os.path.join(builtin, 'gas_exchange_modules_config.json')) as f:
        entry = next(e for e in json.load(f) if e['module_type'] == 'tissue_GE_simple')
    with open(os.path.join(builtin, 'gas_exchange_modules.cellml')) as f:
        text = f.read()
    start = text.index('<component name="tissue_GE_simple_type">')
    component = text[start:text.index('</component>', start) + len('</component>')]
    component = component.replace('tissue_GE_simple_type', 'tissue_GE_simple_no_flux_type')
    entry = dict(entry, module_subtype='no_flux', component_file='no_flux_modules.cellml',
                 component_type='tissue_GE_simple_no_flux_type',
                 exit_ports=[p for p in entry['exit_ports'] if p['port_type'] != 'gas_flux_port'])
    modules_dir = tmp_path / 'external_modules'
    modules_dir.mkdir()
    (modules_dir / 'no_flux_modules.cellml').write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<model name="no_flux" xmlns="http://www.cellml.org/cellml/1.1#" '
        'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n' + component + '\n</model>\n')
    (modules_dir / 'no_flux_modules_config.json').write_text(json.dumps([entry], indent=2))
    return str(modules_dir)


# values a module library's mixer parameterisation may give its sums when nothing is connected
MIXER_SUM_DEFAULTS = [('mix/v', 'm3_per_s', 1.0e-4), ('mix/J_O2', 'm3_per_s', 1.5e-5),
                      ('mix/J_CO2', 'm3_per_s', 5.2e-5)]


@pytest.mark.integration
@pytest.mark.parametrize('no_flux_regions', [['b'], ['a', 'b', 'c']])
def test_a_gas_exchange_module_without_a_flux_port_is_refused(tmp_path, no_flux_regions):
    """A region whose gas-exchange module has no gas_flux_port matches no port of the mixer, so
    its flow and gas would be left out of the mix (or, with every region so, the sums would take
    their parameter values); generation stops instead and names the edge."""
    named = '"gb"' if no_flux_regions == ['b'] else '"g[abc]"'
    with pytest.raises(ValueError, match=named + '.*"mix"'):
        _generate(tmp_path, 'venous_no_flux', mixing=True,
                  ge_subtypes={r: 'no_flux' for r in no_flux_regions},
                  extra_params=MIXER_SUM_DEFAULTS,
                  external_modules_dir=_no_flux_port_ge_modules(tmp_path))
