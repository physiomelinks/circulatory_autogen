"""A state whose initial value is a computed constant (a gate starting at its steady state, m = m_init with
m_init = 1/(1 + exp(-V/k))): libCellML 0.7 sets it in computeComputedConstants, which the generated Python
gives no states argument. The generated model (plain, CasADi and AADC forms) loads and starts there."""
import importlib.util
import math

import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator

MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="gate">
  <component name="g">
    <variable name="t" units="second"/>
    <variable name="V" units="dimensionless" initial_value="-10"/>
    <variable name="k" units="dimensionless" initial_value="5"/>
    <variable name="tau" units="second" initial_value="0.01"/>
    <variable name="m_init" units="dimensionless"/>
    <variable name="m" units="dimensionless" initial_value="m_init"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>m_init</ci><apply><divide/><cn cellml:units="dimensionless">1</cn><apply><plus/>
        <cn cellml:units="dimensionless">1</cn><apply><exp/><apply><divide/><apply><minus/><ci>V</ci></apply>
        <ci>k</ci></apply></apply></apply></apply></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>m</ci></apply><apply><divide/><apply><minus/>
        <ci>m_init</ci><ci>m</ci></apply><ci>tau</ci></apply></apply>
    </math>
  </component>
</model>
"""


@pytest.mark.parametrize('kind', ['python', 'casadi', 'aadc'])
def test_a_state_initialised_from_a_computed_constant_loads(tmp_path, kind):
    if kind != 'python':
        pytest.importorskip(kind)
    path = tmp_path / 'gate.cellml'
    path.write_text(MODEL)
    generated = PythonGenerator(str(path), output_dir=str(tmp_path / kind), casadi_compat=kind == 'casadi',
                                aadc_compat=kind == 'aadc').generate()
    spec = importlib.util.spec_from_file_location(f'gate_{kind}', generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states, rates, variables = module.create_states_array(), module.create_states_array(), module.create_variables_array()
    module.initialise_variables(states, rates, variables)
    module.compute_computed_constants(variables)
    assert float(states[0]) == pytest.approx(1 / (1 + math.exp(2.0)))
