"""
CVODE_myokit on models whose modules use other time units than the model (issue #604).

CellML takes a module's derivatives w.r.t. its own time variable, so ``dx/dt = 1 [1/ms]``
in a module whose ``t`` is in ms makes x grow by 1000 per second of model time. Myokit
converts the module's ``t`` but takes every derivative w.r.t. the free variable; the
Myokit helper rescales those derivatives after import.
"""
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities import parameters_file

MIXED_TIME_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="mixed_time" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
  <units name="ms"><unit prefix="milli" units="second"/></units>
  <units name="per_ms"><unit prefix="milli" units="second" exponent="-1"/></units>
  <units name="per_s"><unit units="second" exponent="-1"/></units>
  <component name="environment">
    <variable name="time" units="second" public_interface="out"/>
  </component>
  <component name="fast">
    <variable name="t" units="ms" public_interface="in"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <variable name="t_copy" units="ms"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply><cn cellml:units="per_ms">1</cn></apply>
      <apply><eq/><ci>t_copy</ci><ci>t</ci></apply>
    </math>
  </component>
  <component name="slow">
    <variable name="t" units="second" public_interface="in"/>
    <variable name="y" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply><cn cellml:units="per_s">1</cn></apply>
    </math>
  </component>
  <connection><map_components component_1="environment" component_2="fast"/><map_variables variable_1="time" variable_2="t"/></connection>
  <connection><map_components component_1="environment" component_2="slow"/><map_variables variable_1="time" variable_2="t"/></connection>
</model>
"""


def test_derivative_wrt_ms_time_is_scaled_to_model_seconds(tmp_path):
    path = tmp_path / 'mixed_time.cellml'
    path.write_text(MIXED_TIME_CELLML)
    helper = get_simulation_helper(model_path=str(path), solver='CVODE_myokit',
                                   model_type='cellml', dt=0.5, sim_time=1.0, pre_time=0.0)
    helper.run()
    x, t_copy, y = (np.asarray(r).ravel() for r in helper.get_results(
        [['fast/x'], ['fast/t_copy'], ['slow/y']], flatten=True))
    t = np.asarray(helper.get_time())
    np.testing.assert_allclose(x, 1000.0 * t, rtol=1e-6)       # 1 per ms
    np.testing.assert_allclose(t_copy, 1000.0 * t, rtol=1e-12)  # t itself in ms
    np.testing.assert_allclose(y, t, rtol=1e-6)                # the seconds module unchanged


MS_DERIVATIVE_CELLML2 = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="ms_derivative" xmlns="http://www.cellml.org/cellml/2.0#">
  <units name="ms"><unit prefix="milli" units="second"/></units>
  <units name="per_ms"><unit prefix="milli" units="second" exponent="-1"/></units>
  <component name="environment">
    <variable name="time" units="second" interface="public"/>
  </component>
  <component name="decay">
    <variable name="t_ms" units="ms" interface="public"/>
    <variable name="k" units="per_ms" initial_value="0.001"/>
    <variable name="x" units="dimensionless" initial_value="1"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t_ms</ci></bvar><ci>x</ci></apply>
        <apply><times/><apply><minus/><ci>k</ci></apply><ci>x</ci></apply></apply>
    </math>
  </component>
  <connection component_1="environment" component_2="decay">
    <map_variables variable_1="time" variable_2="t_ms"/>
  </connection>
</model>
"""


def test_myokit_import_does_not_yet_scale_derivatives_for_bvar_time_units(tmp_path):
    """Guard for CA's correction: Myokit's own CellML import leaves dx/dt_ms per second
    unscaled (-k*x = -0.001 at x=1, where the CellML means -1 per second).
    scale_derivatives_to_free_time in myokit_helper.py supplies the factor of 1000."""
    from libcuflynx.solver_wrappers.myokit_helper import myokit
    import myokit.formats

    path = tmp_path / 'ms_derivative.cellml'
    path.write_text(MS_DERIVATIVE_CELLML2)
    model = myokit.formats.importer('cellml').model(str(path))
    dot_x = model.get('decay.x').rhs().eval()
    if dot_x == pytest.approx(-1.0):
        pytest.fail(f'Myokit {myokit.__version__} now scales derivatives for bvar time units;'
                    ' remove CA\'s correction in myokit_helper.py'
                    ' (scale_derivatives_to_free_time, #604/#605) or it will double-scale.')
    assert dot_x == pytest.approx(-1e-3), (
        f'Myokit {myokit.__version__} imports dx/dt_ms as {dot_x} per second (expected the'
        ' unscaled -0.001 that scale_derivatives_to_free_time corrects); check the'
        ' correction in myokit_helper.py still applies.')


COUPLED_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="coupled_{time_units}" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
  <units name="minute"><unit multiplier="60" units="second"/></units>
  <units name="per_second"><unit units="second" exponent="-1"/></units>
  <units name="per_minute"><unit units="minute" exponent="-1"/></units>
  <component name="environment">
    <variable name="time" units="second" public_interface="out"/>
  </component>
  <component name="a">
    <variable name="t" units="second" public_interface="in"/>
    <variable name="y" units="dimensionless" public_interface="in"/>
    <variable name="x" units="dimensionless" initial_value="1" public_interface="out"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><times/><apply><minus/><cn cellml:units="per_second">0.1</cn></apply>
          <apply><minus/><ci>x</ci><ci>y</ci></apply></apply></apply>
    </math>
  </component>
  <component name="b">
    <variable name="t" units="{time_units}" public_interface="in"/>
    <variable name="x" units="dimensionless" public_interface="in"/>
    <variable name="y" units="dimensionless" initial_value="0" public_interface="out"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply>
        <apply><plus/>
          <apply><times/><cn cellml:units="per_{time_units}">{k}</cn>
            <apply><minus/><ci>x</ci><ci>y</ci></apply></apply>
          <apply><times/><cn cellml:units="per_{time_units}">{f}</cn>
            <apply><sin/><apply><times/><cn cellml:units="per_{time_units}">{w}</cn><ci>t</ci></apply></apply></apply>
        </apply></apply>
    </math>
  </component>
  <connection><map_components component_1="environment" component_2="a"/><map_variables variable_1="time" variable_2="t"/></connection>
  <connection><map_components component_1="environment" component_2="b"/><map_variables variable_1="time" variable_2="t"/></connection>
  <connection><map_components component_1="a" component_2="b"/><map_variables variable_1="x" variable_2="x"/><map_variables variable_1="y" variable_2="y"/></connection>
</model>
"""


def _run_coupled(tmp_path, time_units, k, f, w):
    path = tmp_path / f'coupled_{time_units}.cellml'
    path.write_text(COUPLED_CELLML.format(time_units=time_units, k=k, f=f, w=w))
    helper = get_simulation_helper(model_path=str(path), solver='CVODE_myokit',
                                   model_type='cellml', dt=1.0, sim_time=180.0, pre_time=0.0)
    helper.run()
    return [np.asarray(r).ravel() for r in helper.get_results([['a/x'], ['b/y']], flatten=True)]


def test_minute_module_solves_like_the_same_module_in_seconds(tmp_path):
    """Module b written in minutes (rates per minute, a forcing of period 1 min in its own
    t) coupled to module a in seconds gives the same solution as b rewritten in seconds:
    the scaled derivatives are the same ODEs, so the two agree to round-off."""
    x_s, y_s = _run_coupled(tmp_path, 'second', k=3 / 60, f=6 / 60, w=2 * np.pi / 60)
    x_min, y_min = _run_coupled(tmp_path, 'minute', k=3, f=6, w=2 * np.pi)
    assert np.ptp(y_s) > 0.5   # the forcing and coupling both act over the run
    np.testing.assert_allclose(x_min, x_s, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(y_min, y_s, rtol=1e-9, atol=1e-12)


MS_DECAY_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="ms_decay_type">
        <variable name="t" public_interface="in" units="ms_test"/>
        <variable name="x_init" public_interface="in" units="dimensionless"/>
        <variable name="k" public_interface="in" units="per_ms_test"/>
        <variable initial_value="x_init" name="x" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
                <apply><minus/><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
            </apply>
        </math>
    </component>
</model>
"""

MS_DECAY_CONFIG = """\
[
  {
    "module_type": "ms_decay",
    "module_subtype": "nn",
    "module_format": "cellml",
    "component_file": "ms_decay_modules.cellml",
    "component_type": "ms_decay_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["x_init", "dimensionless", "access", "constant"],
      ["k", "per_ms_test", "access", "constant"],
      ["x", "dimensionless", "access", "variable"]
    ]
  }
]
"""

MS_UNITS_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="Units" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <units name="ms_test">
        <unit prefix="milli" units="second"/>
    </units>
    <units name="per_ms_test">
        <unit prefix="milli" units="second" exponent="-1"/>
    </units>
</model>
"""


@pytest.mark.integration
def test_generated_model_with_ms_module_runs_at_model_time(tmp_path):
    """A module written in ms, generated into CA's seconds-based model, decays at its own
    rate: k = 5e-4 /ms is 0.5 /s."""
    external_dir = tmp_path / 'external_modules'
    external_dir.mkdir()
    (external_dir / 'ms_decay_modules.cellml').write_text(MS_DECAY_CELLML)
    (external_dir / 'ms_decay_modules_config.json').write_text(MS_DECAY_CONFIG)
    (external_dir / 'ms_decay_units.cellml').write_text(MS_UNITS_CELLML)

    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir()
    (resources_dir / 'ms_decay_module_array.csv').write_text(textwrap.dedent("""\
        name,module_subtype,module_type,inp_instances,out_instances
        decay,nn,ms_decay,,
        """))
    parameters = parameters_file.new_document()
    parameters['decay'] = {
        'x_init': {'value': 2.0, 'units': 'dimensionless', 'data_reference': 'test'},
        'k': {'value': 5e-4, 'units': 'per_ms_test', 'data_reference': 'test'},
    }
    parameters_file.write_parameters_file(str(resources_dir / 'ms_decay_parameters.json'),
                                          parameters)

    config = {
        'file_prefix': 'ms_decay',
        'input_param_file': 'ms_decay_parameters.json',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'external_modules_dir': str(external_dir),
        'use_builtin_modules': False,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config)

    model_path = os.path.join(config['generated_models_dir'], 'ms_decay', 'ms_decay.cellml')
    helper = get_simulation_helper(model_path=model_path, solver='CVODE_myokit',
                                   model_type='cellml', dt=0.01, sim_time=2.0, pre_time=0.0)
    helper.run()
    x = np.asarray(helper.get_results([['decay/x']], flatten=True)).ravel()
    t = np.asarray(helper.get_time())
    np.testing.assert_allclose(x, 2.0 * np.exp(-0.5 * t), rtol=1e-4)
