"""
Regression tests for generation bugs:

- #526: ``model_type: python`` code calls libCellML helpers (``eq_func``, ``or_func``,
  ``min``, ``sec``, ...) that the generated utilities never defined.
- #525: a ``sum`` multi-port (e.g. ``volume_sum``) with no inputs raised ``IndexError``.
- #529: the constant pressure / flow BCs lacked their port variable in ``variables_and_units``.
- #524: generic junctions left out neighbours whose BC_type starts with ``nn`` (e.g. constant BCs).
"""
import math
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

REPO_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')


def _read_resource(filename):
    with open(os.path.join(REPO_RESOURCES_DIR, filename)) as f:
        return f.read()


def _generate(tmp_path, prefix, module_array, parameters, model_type='cellml',
              external_modules_dir=None, use_builtin_modules=None):
    """Write ``{prefix}_module_array.csv`` / ``_parameters.csv`` and generate the model.

    Returns the directory the model was generated into.
    """
    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir(exist_ok=True)
    (resources_dir / f'{prefix}_module_array.csv').write_text(textwrap.dedent(module_array))
    (resources_dir / f'{prefix}_parameters.csv').write_text(textwrap.dedent(parameters))
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': model_type,
        'solver': 'solve_ivp' if model_type == 'python' else 'CVODE_myokit',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'external_modules_dir': external_modules_dir,
        'DEBUG': False,
    }
    if use_builtin_modules is not None:
        config['use_builtin_modules'] = use_builtin_modules
    if model_type == 'python':
        config['solver_info'] = {'method': 'RK45', 'max_step': 0.01}
    assert generate_with_new_architecture(False, config)
    return tmp_path / 'generated_models' / prefix


def _run(model_path, variables, model_type='cellml', sim_time=1.0, dt=0.01):
    """Run a generated model; return (t, {variable: series})."""
    if model_type == 'python':
        helper = get_simulation_helper(model_path=str(model_path), solver='solve_ivp',
                                       model_type='python', dt=dt, sim_time=sim_time,
                                       solver_info={'method': 'RK45', 'max_step': dt,
                                                    'rtol': 1e-8, 'atol': 1e-10},
                                       pre_time=0.0)
    else:
        helper = get_simulation_helper(model_path=str(model_path), solver='CVODE_myokit',
                                       model_type='cellml', dt=dt, sim_time=sim_time,
                                       pre_time=0.0)
    helper.run()
    results = helper.get_results([[v] for v in variables], flatten=True)
    t = np.asarray(helper.get_time())
    return t, {v: np.asarray(r).ravel() for v, r in zip(variables, results)}


# ---------------------------------------------------------------------------------------
# #526: every libCellML Python-profile helper is defined for model_type python
# ---------------------------------------------------------------------------------------

LOGIC_MODULES_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="eq_rate_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="k" public_interface="in" units="dimensionless"/>
        <variable initial_value="0" name="x" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
                <piecewise>
                    <piece>
                        <cn cellml:units="per_s">1</cn>
                        <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">1</cn></apply>
                    </piece>
                    <otherwise><cn cellml:units="per_s">0</cn></otherwise>
                </piecewise>
            </apply>
        </math>
    </component>
    <component name="logic_rate_type">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="k" public_interface="in" units="dimensionless"/>
        <variable name="a" public_interface="in" units="per_s"/>
        <variable name="b" public_interface="in" units="per_s"/>
        <variable initial_value="0" name="y" public_interface="out" units="dimensionless"/>
        <variable initial_value="0" name="z" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <!-- dy/dt = min(a, b) if (k > 2 or not(k == 2)) else 0: true for k = 1 -->
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply>
                <piecewise>
                    <piece>
                        <apply><min/><ci>a</ci><ci>b</ci></apply>
                        <apply><or/>
                            <apply><gt/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            <apply><not/>
                                <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            </apply>
                        </apply>
                    </piece>
                    <otherwise><cn cellml:units="per_s">0</cn></otherwise>
                </piecewise>
            </apply>
            <!-- dz/dt = a if (k > 2 or k == 2) else b: false for k = 1 -->
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>z</ci></apply>
                <piecewise>
                    <piece>
                        <ci>a</ci>
                        <apply><or/>
                            <apply><gt/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                            <apply><eq/><ci>k</ci><cn cellml:units="dimensionless">2</cn></apply>
                        </apply>
                    </piece>
                    <otherwise><ci>b</ci></otherwise>
                </piecewise>
            </apply>
        </math>
    </component>
</model>
"""

LOGIC_MODULES_CONFIG = """\
[
  {
    "vessel_type": "eq_rate",
    "BC_type": "nn",
    "module_format": "cellml",
    "module_file": "logic_test_modules.cellml",
    "module_type": "eq_rate_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["k", "dimensionless", "access", "constant"],
      ["x", "dimensionless", "access", "variable"]
    ]
  },
  {
    "vessel_type": "logic_rate",
    "BC_type": "nn",
    "module_format": "cellml",
    "module_file": "logic_test_modules.cellml",
    "module_type": "logic_rate_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["k", "dimensionless", "access", "constant"],
      ["a", "per_s", "access", "constant"],
      ["b", "per_s", "access", "constant"],
      ["y", "dimensionless", "access", "variable"],
      ["z", "dimensionless", "access", "variable"]
    ]
  }
]
"""


@pytest.mark.integration
@pytest.mark.parametrize('k, expected_x_rate', [(1.0, 1.0), (3.0, 0.0)])
def test_python_model_with_eq_or_not_min_runs(tmp_path, k, expected_x_rate):
    """#526: a piecewise with <eq/> (and <or/>, <not/>, <min/>) generates and runs as
    model_type python. It used to fail with ``NameError: name 'eq_func' is not defined``."""
    modules_dir = tmp_path / 'external_modules'
    modules_dir.mkdir()
    (modules_dir / 'logic_test_modules.cellml').write_text(LOGIC_MODULES_CELLML)
    (modules_dir / 'logic_test_modules_config.json').write_text(LOGIC_MODULES_CONFIG)

    generated_dir = _generate(
        tmp_path, 'logic_test',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        eqmod,nn,eq_rate,,
        logic,nn,logic_rate,,
        """,
        f"""\
        variable_name,units,value,data_reference
        k_eqmod,dimensionless,{k},test
        k_logic,dimensionless,{k},test
        a_logic,per_s,2.0,test
        b_logic,per_s,0.5,test
        """,
        model_type='python', external_modules_dir=str(modules_dir))

    source = (generated_dir / 'logic_test.py').read_text()
    for helper in ('eq_func', 'or_func', 'not_func', 'min('):
        assert helper in source

    t, res = _run(generated_dir / 'logic_test.py',
                  ['eqmod/x', 'logic/y', 'logic/z'], model_type='python')
    np.testing.assert_allclose(res['eqmod/x'], expected_x_rate * t, atol=1e-8)
    # k = 1: k > 2 or not(k == 2) -> min(a, b) = 0.5; k == 2 is false -> dz/dt = b.
    # k = 3: k > 2 is true           -> min(a, b) = 0.5; k > 2 is true   -> dz/dt = a.
    np.testing.assert_allclose(res['logic/y'], 0.5 * t, atol=1e-8)
    np.testing.assert_allclose(res['logic/z'], (0.5 if k == 1.0 else 2.0) * t, atol=1e-8)


def _helper_namespace(casadi_compat=False):
    namespace = {}
    exec('from math import *\nimport casadi as ca\n', namespace)
    lines = PythonGenerator._comparison_helper_lines(casadi_compat=casadi_compat)
    exec('\n'.join(lines), namespace)
    return namespace


def _libcellml_profile_namespace():
    """The helpers exactly as libCellML's Python profile writes them."""
    lc = pytest.importorskip('libcellml')
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
    namespace = {}
    exec('from math import *\n', namespace)
    for name in _profile_function_names():
        exec(getattr(profile, f'{name}FunctionString')(), namespace)
    return namespace


def _profile_function_names():
    return ['eq', 'neq', 'lt', 'leq', 'gt', 'geq', 'and', 'or', 'xor', 'not', 'min', 'max',
            'sec', 'csc', 'cot', 'sech', 'csch', 'coth',
            'asec', 'acsc', 'acot', 'asech', 'acsch', 'acoth']


def test_python_helpers_cover_the_libcellml_profile():
    """#526: every helper libCellML's Python profile can emit is defined and exported,
    under the name the profile calls it by."""
    lc = pytest.importorskip('libcellml')
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
    profile_names = set()
    for name in _profile_function_names():
        code = getattr(profile, f'{name}FunctionString')()
        profile_names.add(code.split('def ', 1)[1].split('(', 1)[0])
    assert profile_names == set(PythonGenerator.HELPER_FUNCTION_NAMES)

    for casadi_compat in (False, True):
        namespace = _helper_namespace(casadi_compat)
        for name in profile_names:
            assert callable(namespace.get(name)), (name, casadi_compat)
    aadc_code = '\n'.join(PythonGenerator._comparison_helper_lines(False, aadc_compat=True))
    for name in profile_names:
        assert f'def {name}(' in aadc_code


@pytest.mark.parametrize('casadi_compat', [False, True])
def test_python_helpers_match_libcellml_semantics(casadi_compat):
    """#526: the plain and CasADi helpers return what libCellML's own helpers return."""
    reference = _libcellml_profile_namespace()
    ours = _helper_namespace(casadi_compat)

    def value(v):
        return float(v)

    binary = ['eq_func', 'neq_func', 'lt_func', 'leq_func', 'gt_func', 'geq_func',
              'and_func', 'or_func', 'xor_func', 'min', 'max']
    for name in binary:
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.5, -1.0), (-3.0, 2.0)]:
            if casadi_compat and name == 'and_func' and (x < 0 or y < 0):
                continue  # the CasADi and_func treats only positive values as true
            assert value(ours[name](x, y)) == reference[name](x, y), (name, x, y)
    for x in (0.0, 1.0, -2.0):
        assert value(ours['not_func'](x)) == reference['not_func'](x)

    unary = {'sec': 0.4, 'csc': 0.4, 'cot': 0.4, 'sech': 0.4, 'csch': 0.4, 'coth': 0.4,
             'asec': 2.0, 'acsc': 2.0, 'acot': 2.0, 'asech': 0.5, 'acsch': 0.5, 'acoth': 2.0}
    for name, x in unary.items():
        assert math.isclose(value(ours[name](x)), reference[name](x), rel_tol=1e-12), name


def test_casadi_helpers_accept_symbolic_arguments():
    """#526: the CasADi variants build expressions from SX symbols rather than failing on
    Python truthiness."""
    ca = pytest.importorskip('casadi')
    ours = _helper_namespace(casadi_compat=True)
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    for name in ('eq_func', 'neq_func', 'or_func', 'xor_func', 'min', 'max'):
        f = ca.Function(name, [x, y], [ours[name](x, y)])
        assert float(f(1.0, 2.0)) == float(_helper_namespace()[name](1.0, 2.0))
    f = ca.Function('not', [x], [ours['not_func'](x)])
    assert float(f(0.0)) == 1.0
    for name in ('sec', 'acoth'):
        f = ca.Function(name, [x], [ours[name](x)])
        assert math.isclose(float(f(2.0)), _helper_namespace()[name](2.0))


# ---------------------------------------------------------------------------------------
# #525: an empty 'sum' multi-port is 0
# ---------------------------------------------------------------------------------------

@pytest.mark.integration
def test_empty_sum_multi_port_is_zero(tmp_path, capsys):
    """#525: a volume_sum with nothing connected generates, is mapped to its vessel, and is
    0, with a warning naming the vessel. It used to raise ``IndexError``."""
    module_array = _read_resource('3compartment_module_array.csv').rstrip('\n') + \
        '\nextra_sum, nn, volume_sum, , \n'
    generated_dir = _generate(tmp_path, 'empty_sum', module_array,
                              _read_resource('3compartment_parameters.csv'))
    warnings = [line for line in capsys.readouterr().out.splitlines() if 'WARNING' in line]
    assert any('extra_sum' in line for line in warnings), warnings

    model_text = (generated_dir / 'empty_sum.cellml').read_text()
    assert '<cn cellml:units="m3">0</cn>' in model_text
    # ... and the sum is still mapped to the vessel's port variable.
    assert 'variable_1="q_extra_sum_sum" variable_2="q"' in model_text

    # Myokit merges connected variables, so read the sums where they are computed.
    _, res = _run(generated_dir / 'empty_sum.cellml',
                  ['sum_blood_volume/q_extra_sum_sum', 'sum_blood_volume/q_volume_sum_sum'],
                  sim_time=0.1)
    np.testing.assert_array_equal(res['sum_blood_volume/q_extra_sum_sum'], 0.0)
    # The populated sum is untouched.
    assert np.all(res['sum_blood_volume/q_volume_sum_sum'] > 0.0)


# ---------------------------------------------------------------------------------------
# #529: the constant pressure / flow BCs can be connected
# ---------------------------------------------------------------------------------------

VESSEL_PARAMETERS = """\
    R_va,Js_per_m6,1.0e7,test
    C_va,m6_per_J,1.0e-8,test
    I_va,Js2_per_m6,1.0e5,test
    q_0_va,m3,0,test
    u_0_va,J_per_m3,0,test
    u_ext_va,J_per_m3,0,test
    """


@pytest.mark.integration
def test_constant_flow_into_vessel_draining_to_constant_pressure(tmp_path):
    """#529: inlet_flow / outlet_pressure nn_constant connect to a vessel. Generation used
    to stop with "the port variable v is not a variable for vessel type: outlet_pressure"."""
    generated_dir = _generate(
        tmp_path, 'const_flow_bc',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        flow_in,nn_constant,inlet_flow,,va
        va,vp,arterial_simple,flow_in,p_out
        p_out,nn_constant,outlet_pressure,va,
        """,
        """\
        variable_name,units,value,data_reference
        v_flow_in,m3_per_s,1.0e-4,test
        P_p_out,J_per_m3,666.0,test
        """ + VESSEL_PARAMETERS)
    _, res = _run(generated_dir / 'const_flow_bc.cellml', ['va/v', 'va/u'], sim_time=2.0)
    # Steady state: the vessel passes the imposed inflow, at P_out + R v.
    assert res['va/v'][-1] == pytest.approx(1.0e-4, rel=1e-3)
    assert res['va/u'][-1] == pytest.approx(666.0 + 1.0e7 * 1.0e-4, abs=1.0)


@pytest.mark.integration
def test_constant_pressure_into_vessel_drained_by_constant_flow(tmp_path):
    """#529: inlet_pressure / outlet_flow nn_constant connect to a vessel."""
    generated_dir = _generate(
        tmp_path, 'const_pressure_bc',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        p_in,nn_constant,inlet_pressure,,va
        va,pv,arterial_simple,p_in,flow_out
        flow_out,nn_constant,outlet_flow,va,
        """,
        """\
        variable_name,units,value,data_reference
        P_p_in,J_per_m3,12000.0,test
        v_flow_out,m3_per_s,1.0e-4,test
        """ + VESSEL_PARAMETERS)
    _, res = _run(generated_dir / 'const_pressure_bc.cellml', ['va/v', 'va/u'], sim_time=2.0)
    # Steady state: the vessel carries the imposed outflow, at P_in - R v.
    assert res['va/v'][-1] == pytest.approx(1.0e-4, rel=1e-3)
    assert res['va/u'][-1] == pytest.approx(12000.0 - 1.0e7 * 1.0e-4, abs=1.0)


# ---------------------------------------------------------------------------------------
# #524: generic junctions include boundary-condition neighbours
# ---------------------------------------------------------------------------------------

# Min_junction vp geometry and wall law as in circulatory-autogen-modules (modules/BG,
# Min_junction vp), with the library's test viscosity/density (mu = rho = 1) so the
# resistive pressure drop across the junction is large enough to check.
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


@pytest.mark.integration
def test_min_junction_fed_by_boundary_conditions(tmp_path):
    """#524: a Min_junction whose inlet node is two constant-inflow BCs (BC_type
    nn_constant) generates and runs. It used to stop with "Min_junction junc has NO other
    vessels connected to its inlet node"."""
    generated_dir = _generate(
        tmp_path, 'junction_bc',
        """\
        name,BC_type,vessel_type,inp_vessels,out_vessels
        inflow_a,nn_constant,inlet_flow,,junc
        inflow_b,nn_constant,inlet_flow,,junc
        junc,vp,Min_junction,inflow_a inflow_b,outp
        outp,nn_constant,outlet_pressure,junc,
        """,
        JUNCTION_PARAMETERS, use_builtin_modules=True)

    model_text = (generated_dir / 'junction_bc.cellml').read_text()
    # Each BC's port flow goes into the junction's flow sum, and the junction's pressure
    # to each BC's port pressure.
    for bc in ('inflow_a', 'inflow_b'):
        assert f'variable_1="v" variable_2="v_{bc}_Min"' in model_text
        assert (f'<map_components component_1="junc_module" component_2="{bc}_module"/>\n'
                f'   <map_variables variable_1="u" variable_2="P"/>') in model_text

    # Myokit merges connected variables, so the junction's inflow v_in_sum is read where
    # it is computed.
    v_in_sum = 'generic_junction_connection/v_junc_sum_Min'
    _, res = _run(generated_dir / 'junction_bc.cellml',
                  ['junc/v', 'junc/u', 'junc/R', v_in_sum], sim_time=1.0)
    v_sum = 4.0e-5 + 7.0e-5
    np.testing.assert_allclose(res[v_in_sum], v_sum, rtol=1e-12)
    # Steady state: the junction passes the summed inflow, at P_out + R v.
    assert res['junc/v'][-1] == pytest.approx(v_sum, rel=1e-4)
    R = 8 * 1.0 * 0.014796 / (np.pi * 0.0133364 ** 4)
    np.testing.assert_allclose(res['junc/R'], R, rtol=1e-10)
    assert R * v_sum > 100.0  # the drop is well above the tolerance below
    assert res['junc/u'][-1] == pytest.approx(12665.6 + R * v_sum, abs=0.1)
