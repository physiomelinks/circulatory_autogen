"""
The first two letters of a ``BC_type`` (``module_subtype``) are a vessel BC pair (``vv``,
``vp``, ``pv``, ``pp``) only for vessel modules. Any other module may have any BC_type,
e.g. ``lv_test``, ``rv``, ``SN_soma`` or ``cardiomyocyte``, without an ``nn`` prefix.

Before, every connection whose sides did not start with ``nn`` had its BC letters checked,
so a non-vessel named ``lv_...`` (ends in ``v``) feeding one not starting with ``p`` stopped
generation, and a non-vessel neighbour of an Nout junction lost its connection.
"""
import textwrap

import numpy as np
import pandas as pd
import pytest

from libcuflynx.generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.vessel_bc import has_vessel_ports, is_vessel_bc, is_vessel_module

VESSEL_PORTS = {
    'entrance_ports': [{'port_type': 'vessel_port', 'variables': ['v_in', 'u']}],
    'exit_ports': [{'port_type': 'vessel_port', 'variables': ['v', 'u_out']}],
    'general_ports': [],
}
MICRO_PORTS = {
    'entrance_ports': [{'port_type': 'flow_port', 'variables': ['v_in']},
                       {'port_type': 'pressure_port', 'variables': ['u']}],
    'exit_ports': [{'port_type': 'flow_port', 'variables': ['v']},
                   {'port_type': 'pressure_port', 'variables': ['u_out']}],
    'general_ports': [],
}
SIGNAL_PORTS = {
    'entrance_ports': [{'port_type': 'signal_port', 'variables': ['s_in']}],
    'exit_ports': [{'port_type': 'signal_port', 'variables': ['s']}],
    'general_ports': [],
}


# ---------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('bc', ['vv', 'vp', 'pv', 'pp', 'vp_nonlinear', 'pv_micro'])
def test_vessel_bc_prefixes(bc):
    assert is_vessel_bc(bc)


@pytest.mark.parametrize('bc', ['nn', 'nn_constant', 'lv_test', 'rv', 'SN_soma', 'cardiomyocyte',
                                'Up_ctrl', 'ventilated', 'v', 'p', '', None])
def test_non_vessel_bc_prefixes(bc):
    assert not is_vessel_bc(bc)


def test_vessel_module_needs_vessel_ports():
    assert is_vessel_module({'BC_type': 'vp', **VESSEL_PORTS})
    assert is_vessel_module({'BC_type': 'pv_micro', **MICRO_PORTS})
    # a BC-like name on a module without vessel ports is not a vessel
    assert not is_vessel_module({'BC_type': 'pp_cell', **SIGNAL_PORTS})
    # nn modules with a vessel_port (the constant BCs) stay non-vessels
    assert not is_vessel_module({'BC_type': 'nn_constant', **VESSEL_PORTS})
    assert not is_vessel_module({'BC_type': 'lv_test', **VESSEL_PORTS})
    # without port info (the raw vessel array) the prefix decides
    assert has_vessel_ports({'BC_type': 'vp'}) is None
    assert is_vessel_module({'BC_type': 'vp'})
    assert not is_vessel_module({'BC_type': 'lv'})
    # pandas rows and itertuples() rows work too
    df = pd.DataFrame([{'name': 'a', 'BC_type': 'vp', **VESSEL_PORTS},
                       {'name': 'b', 'BC_type': 'lv_test', **SIGNAL_PORTS}])
    assert [is_vessel_module(row) for _, row in df.iterrows()] == [True, False]
    assert [is_vessel_module(row) for row in df.itertuples()] == [True, False]


# ---------------------------------------------------------------------------------------
# the connection check
# ---------------------------------------------------------------------------------------

_check = CVS0DCellMLGenerator._CVS0DCellMLGenerator__check_input_output_modules


def _pair_df(main_bc, main_ports, out_bc, out_ports):
    return pd.DataFrame([
        {'name': 'a', 'BC_type': main_bc, 'module_type': 'a_type', 'vessel_type': 'a_type',
         'inp_vessels': [], 'out_vessels': ['b'], **main_ports},
        {'name': 'b', 'BC_type': out_bc, 'module_type': 'b_type', 'vessel_type': 'b_type',
         'inp_vessels': ['a'], 'out_vessels': [], **out_ports},
    ])


def _run_check(df):
    a, b = df.iloc[0], df.iloc[1]
    _check(None, df, 'a', 'b', a.BC_type, b.BC_type, a.module_type, b.module_type)


@pytest.mark.parametrize('main_bc, out_bc', [('vv', 'vp'), ('vp', 'pv'), ('pv_micro', 'vv_micro')])
def test_mismatched_vessel_pair_still_fails(main_bc, out_bc):
    ports = MICRO_PORTS if main_bc.endswith('micro') else VESSEL_PORTS
    with pytest.raises(SystemExit):
        _run_check(_pair_df(main_bc, ports, out_bc, ports))


@pytest.mark.parametrize('main_bc, out_bc', [('vv', 'pv'), ('vp', 'vv'), ('pv', 'pp')])
def test_matched_vessel_pair_passes(main_bc, out_bc):
    _run_check(_pair_df(main_bc, VESSEL_PORTS, out_bc, VESSEL_PORTS))


@pytest.mark.parametrize('main, out', [
    (('vv', VESSEL_PORTS), ('lv_test', SIGNAL_PORTS)),       # vessel -> non-vessel
    (('lv_test', SIGNAL_PORTS), ('rv_test', SIGNAL_PORTS)),  # non-vessel -> non-vessel
    (('Up_ctrl', SIGNAL_PORTS), ('vp', VESSEL_PORTS)),       # non-vessel -> vessel
    (('vv', VESSEL_PORTS), ('pp_cell', SIGNAL_PORTS)),       # BC-like name, no vessel ports
    (('vv', VESSEL_PORTS), ('nn', VESSEL_PORTS)),            # nn unchanged
    (('nn_constant', VESSEL_PORTS), ('vv', VESSEL_PORTS)),
])
def test_non_vessel_neighbour_skips_bc_check(main, out):
    _run_check(_pair_df(main[0], main[1], out[0], out[1]))


# ---------------------------------------------------------------------------------------
# generation with non-vessel modules of arbitrary names
# ---------------------------------------------------------------------------------------

SOURCE_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="bc_test_source">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="s_init" public_interface="in" units="dimensionless"/>
        <variable name="k" public_interface="in" units="bc_test_per_second"/>
        <variable initial_value="s_init" name="s" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>s</ci></apply>
                <apply><minus/><apply><times/><ci>k</ci><ci>s</ci></apply></apply>
            </apply>
        </math>
    </component>
</model>
"""

SINK_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="bc_test_sink">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="s_in" public_interface="in" units="dimensionless"/>
        <variable name="gain" public_interface="in" units="dimensionless"/>
        <variable name="y" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><eq/><ci>y</ci><apply><times/><ci>gain</ci><ci>s_in</ci></apply></apply>
        </math>
    </component>
</model>
"""

VOLUME_READER_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="bc_test_volume_reader">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="q_a" public_interface="in" units="m3"/>
        <variable name="q_b" public_interface="in" units="m3"/>
        <variable name="q_read" public_interface="out" units="m3"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><eq/><ci>q_read</ci><apply><plus/><ci>q_a</ci><ci>q_b</ci></apply></apply>
        </math>
    </component>
</model>
"""

UNITS_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="Units" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <units name="bc_test_per_second">
        <unit exponent="-1" units="second"/>
    </units>
</model>
"""


def _config_entry(vessel_type, bc_type, file_stem, module_type, entrance, exit_, variables):
    import json
    return json.dumps([{
        'vessel_type': vessel_type, 'BC_type': bc_type, 'module_format': 'cellml',
        'module_file': f'{file_stem}_modules.cellml', 'module_type': module_type,
        'entrance_ports': entrance, 'exit_ports': exit_, 'general_ports': [],
        'variables_and_units': variables,
    }], indent=2)


def _write_library_module(library_dir, dir_name, cellml, config):
    module_dir = library_dir / dir_name
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / f'{dir_name}_modules.cellml').write_text(cellml)
    (module_dir / f'{dir_name}_modules_config.json').write_text(config)
    (module_dir / f'{dir_name}_units.cellml').write_text(UNITS_CELLML)


def _write_source_and_sink(library_dir, source_bc, sink_bc):
    _write_library_module(library_dir, 'bc_test_source', SOURCE_CELLML, _config_entry(
        'bc_test_source', source_bc, 'bc_test_source', 'bc_test_source', [],
        [{'port_type': 'signal_port', 'variables': ['s']}],
        [['s_init', 'dimensionless', 'access', 'constant'],
         ['k', 'bc_test_per_second', 'access', 'constant'],
         ['s', 'dimensionless', 'access', 'variable']]))
    _write_library_module(library_dir, 'bc_test_sink', SINK_CELLML, _config_entry(
        'bc_test_sink', sink_bc, 'bc_test_sink', 'bc_test_sink',
        [{'port_type': 'signal_port', 'variables': ['s_in']}], [],
        [['gain', 'dimensionless', 'access', 'constant'],
         ['s_in', 'dimensionless', 'access', 'variable'],
         ['y', 'dimensionless', 'access', 'variable']]))


def _generate(tmp_path, prefix, vessel_array, parameters, library_dir, use_builtin_modules):
    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir(parents=True, exist_ok=True)
    (resources_dir / f'{prefix}_vessel_array.csv').write_text(textwrap.dedent(vessel_array))
    (resources_dir / f'{prefix}_parameters.csv').write_text(textwrap.dedent(parameters))
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'module_library_dirs': [str(library_dir)],
        'use_builtin_modules': use_builtin_modules,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config)
    return tmp_path / 'generated_models' / prefix


def _run(model_path, variables, sim_time=1.0):
    helper = get_simulation_helper(model_path=str(model_path), solver='CVODE_myokit',
                                   model_type='cellml', dt=0.01, sim_time=sim_time,
                                   pre_time=0.0)
    helper.run()
    results = helper.get_results([[v] for v in variables], flatten=True)
    return np.asarray(helper.get_time()), {v: np.asarray(r).ravel()
                                           for v, r in zip(variables, results)}


@pytest.mark.integration
@pytest.mark.parametrize('source_bc, sink_bc', [
    ('lv_test', 'rv_test'),           # 'lv' ends in v, 'rv' does not start with p
    ('Up_ctrl', 'lv_test'),           # 'Up' ends in p, 'lv' does not start with v
    ('SN_soma_v01', 'cardiomyocyte_v01'),
])
def test_non_vessel_modules_with_arbitrary_names_connect_and_generate(tmp_path, source_bc, sink_bc):
    library_dir = tmp_path / 'library'
    _write_source_and_sink(library_dir, source_bc, sink_bc)
    generated_dir = _generate(
        tmp_path, 'bc_names',
        f"""\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        src,{source_bc},bc_test_source,,snk
        snk,{sink_bc},bc_test_sink,src,
        """,
        """\
        variable_name,units,value,data_reference
        s_init_src,dimensionless,2.0,test
        k_src,bc_test_per_second,0.5,test
        gain_snk,dimensionless,3.0,test
        """,
        library_dir, use_builtin_modules=False)

    model_text = (generated_dir / 'bc_names.cellml').read_text()
    assert '<map_components component_1="src_module" component_2="snk_module"/>' in model_text
    t, res = _run(generated_dir / 'bc_names.cellml', ['snk/y'], sim_time=2.0)
    np.testing.assert_allclose(res['snk/y'], 3.0 * 2.0 * np.exp(-0.5 * t), rtol=1e-4)


def _constant_bcs_connect():
    """The constant inlet/outlet BCs can be connected (#529) and join junction nodes (#524)
    only once the generation fixes of #531 are in this branch."""
    import json
    import os
    from libcuflynx.utilities.package_resources import builtin_modules_dir
    path = os.path.join(builtin_modules_dir(), 'boundary_condition_modules_config.json')
    for entry in json.load(open(path)):
        if entry['vessel_type'] == 'inlet_flow' and entry['BC_type'] == 'nn_constant':
            return any(v[0] == 'P' for v in entry['variables_and_units'])
    return False


requires_constant_bcs = pytest.mark.skipif(
    not _constant_bcs_connect(), reason='needs the constant-BC generation fixes of #531')


# Min_junction vp and the constant BCs as in test_generation_fixes (#524).
JUNCTION_PARAMETERS = """\
    variable_name,units,value,data_reference
    v_inflow_a,m3_per_s,4.0e-5,test
    v_inflow_b,m3_per_s,7.0e-5,test
    P_outp,J_per_m3,12665.6,test
    u_0_junc,J_per_m3,10640.0,test
    u_ext_junc,J_per_m3,0.0,test
    theta_junc,dimensionless,90.0,test
    E_junc,J_per_m3,4.0e5,test
    l_junc,metre,0.014796,test
    r_0_junc,metre,0.0133364,test
    mu,Js_per_m3,1.0,test
    rho,Js2_per_m5,1.0,test
    g,m_per_s2,9.81,test
    beta_g,dimensionless,0.0,test
    a_vessel,dimensionless,0.2802,test
    b_vessel,per_m,-505.3,test
    c_vessel,dimensionless,0.1324,test
    d_vessel,per_m,-11.14,test
    """


def _write_volume_reader(library_dir, bc_type, n_ports):
    variables = ['q_a', 'q_b'][:n_ports]
    entrance = [{'port_type': 'volume_port', 'variables': [v]} for v in variables]
    cellml = VOLUME_READER_CELLML
    if n_ports == 1:
        cellml = (cellml.replace('<variable name="q_b" public_interface="in" units="m3"/>\n        ', '')
                  .replace('<apply><plus/><ci>q_a</ci><ci>q_b</ci></apply>', '<ci>q_a</ci>'))
    _write_library_module(library_dir, 'bc_test_volume_reader', cellml, _config_entry(
        'bc_test_volume_reader', bc_type, 'bc_test_volume_reader', 'bc_test_volume_reader',
        entrance, [], [[v, 'm3', 'access', 'variable'] for v in variables]
        + [['q_read', 'm3', 'access', 'variable']]))


@pytest.mark.integration
@requires_constant_bcs
@pytest.mark.parametrize('reader_bc', ['lv_test', 'SN_soma_v01', 'cardiomyocyte_v01'])
def test_min_junction_with_non_vessel_neighbour(tmp_path, reader_bc):
    """A non-vessel module reading a Min_junction's volume connects whatever its name, and
    is not taken into the junction node. 'lv_test' used to stop generation ("junc output BC
    is p, the input BC of reader should be v")."""
    library_dir = tmp_path / 'library'
    _write_volume_reader(library_dir, reader_bc, n_ports=1)
    generated_dir = _generate(
        tmp_path, 'junction_reader',
        f"""\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        inflow_a,nn_constant,inlet_flow,,junc
        inflow_b,nn_constant,inlet_flow,,junc
        junc,vp,Min_junction,inflow_a inflow_b,outp reader
        outp,nn_constant,outlet_pressure,junc,
        reader,{reader_bc},bc_test_volume_reader,junc,
        """,
        JUNCTION_PARAMETERS,
        library_dir, use_builtin_modules=True)

    model_text = (generated_dir / 'junction_reader.cellml').read_text()
    assert ('<map_components component_1="junc_module" component_2="reader_module"/>\n'
            '   <map_variables variable_1="q" variable_2="q_a"/>') in model_text
    # the reader has no vessel_port, so it is not one of the junction's node neighbours
    assert 'v_reader_Min' not in model_text
    _, res = _run(generated_dir / 'junction_reader.cellml', ['junc/q', 'reader/q_read'])
    np.testing.assert_allclose(res['reader/q_read'], res['junc/q'], rtol=1e-12)
    # the junction still passes the summed inflow
    assert res['junc/q'][-1] > 0.0


@pytest.mark.integration
@requires_constant_bcs
def test_non_vessel_neighbour_of_nout_junction_keeps_its_connections(tmp_path):
    """A non-vessel reading volumes from an Nout_junction and from one of the junction's
    vessels gets both connections. With a name not starting nn, the vessel -> reader
    connection used to be dropped as if the reader were a vessel fed by the junction."""
    library_dir = tmp_path / 'library'
    _write_volume_reader(library_dir, 'lv_test', n_ports=2)
    generated_dir = _generate(
        tmp_path, 'nout_reader',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        p_in,nn_constant,inlet_pressure,,junc
        junc,pv,Nout_junction,p_in,va fb reader
        va,pv,arterial_simple,junc,fa reader
        fa,nn_constant,outlet_flow,va,
        fb,nn_constant,outlet_flow,junc,
        reader,lv_test,bc_test_volume_reader,junc va,
        """,
        """\
        variable_name,units,value,data_reference
        P_p_in,J_per_m3,12000.0,test
        v_fa,m3_per_s,4.0e-5,test
        v_fb,m3_per_s,7.0e-5,test
        u_0_junc,J_per_m3,10640.0,test
        u_ext_junc,J_per_m3,0.0,test
        theta_junc,dimensionless,90.0,test
        E_junc,J_per_m3,4.0e5,test
        l_junc,metre,0.014796,test
        r_0_junc,metre,0.0133364,test
        R_va,Js_per_m6,1.0e7,test
        C_va,m6_per_J,1.0e-8,test
        I_va,Js2_per_m6,1.0e5,test
        q_0_va,m3,0,test
        u_0_va,J_per_m3,0,test
        u_ext_va,J_per_m3,0,test
        mu,Js_per_m3,1.0,test
        rho,Js2_per_m5,1.0,test
        g,m_per_s2,9.81,test
        beta_g,dimensionless,0.0,test
        a_vessel,dimensionless,0.2802,test
        b_vessel,per_m,-505.3,test
        c_vessel,dimensionless,0.1324,test
        d_vessel,per_m,-11.14,test
        """,
        library_dir, use_builtin_modules=True)

    model_text = (generated_dir / 'nout_reader.cellml').read_text()
    assert ('<map_components component_1="junc_module" component_2="reader_module"/>\n'
            '   <map_variables variable_1="q" variable_2="q_a"/>') in model_text
    assert ('<map_components component_1="va_module" component_2="reader_module"/>\n'
            '   <map_variables variable_1="q" variable_2="q_b"/>') in model_text


@pytest.mark.integration
@pytest.mark.slow
def test_renaming_junction_non_vessel_neighbours_changes_nothing(tmp_path):
    """generic_junction_test_open_loop has a K_tube (material_prop_visco_const, BC_type nn)
    next to every Nout_junction. Giving those K_tubes the version name ``lv_test`` instead
    of ``nn`` generates the same model: they are still not taken into the junction nodes
    nor checked as vessel BCs."""
    import json
    import os
    import re
    from libcuflynx.utilities.package_resources import builtin_modules_dir

    builtin = builtin_modules_dir()
    entry = next(m for m in json.load(open(os.path.join(builtin, 'vessel_properties_modules_config.json')))
                 if m['vessel_type'] == 'material_prop_visco_const' and m['BC_type'] == 'nn')
    cellml = open(os.path.join(builtin, entry['module_file'])).read()
    component = re.search(r'<component name="material_prop_visco_const_type".*?</component>',
                          cellml, re.S).group(0)
    component = component.replace('"material_prop_visco_const_type"', '"K_tube_lv_test_type"', 1)
    library_dir = tmp_path / 'library'
    module_dir = library_dir / 'K_tube_lv_test'
    module_dir.mkdir(parents=True)
    (module_dir / 'K_tube_lv_test_modules.cellml').write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" '
        'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n' + component + '\n</model>\n')
    entry = dict(entry, BC_type='lv_test', module_type='K_tube_lv_test_type',
                 module_file='K_tube_lv_test_modules.cellml')
    (module_dir / 'K_tube_lv_test_modules_config.json').write_text(json.dumps([entry]))

    resources = os.path.join(os.path.dirname(__file__), '..', 'resources')
    vessel_array = open(os.path.join(resources, 'generic_junction_test_open_loop_vessel_array.csv')).read()
    parameters = open(os.path.join(resources, 'generic_junction_test_open_loop_parameters.csv')).read()
    renamed = re.sub(r'^(K_tube_\w+),nn,material_prop_visco_const,', r'\1,lv_test,material_prop_visco_const,',
                     vessel_array, flags=re.M)
    assert renamed.count(',lv_test,') > 50

    original_dir = _generate(tmp_path / 'original', 'gj', vessel_array, parameters,
                             library_dir, use_builtin_modules=True)
    renamed_dir = _generate(tmp_path / 'renamed', 'gj', renamed, parameters,
                            library_dir, use_builtin_modules=True)
    original = (original_dir / 'gj.cellml').read_text()
    renamed_model = (renamed_dir / 'gj.cellml').read_text()
    assert original == renamed_model.replace('K_tube_lv_test_type', 'material_prop_visco_const_type')
