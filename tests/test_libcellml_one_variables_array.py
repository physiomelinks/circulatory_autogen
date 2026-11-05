"""libCellML's generated code in the one-``variables``-array layout the solvers and the C++ generator use (#271)."""
import os
import re
import shutil
import subprocess

import libcellml as lc
import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator
from libcuflynx.utilities.libcellml_helper_funcs import analyser_variables, c_code_with_one_variables_array

# dx/dt = -c x + e, with c = 2 k computed from the constant k, y = c x algebraic, and e external (C only)
MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="m">
  <component name="c">
    <variable name="t" units="second" interface="public"/>
    <variable name="x" units="dimensionless" initial_value="3" interface="public"/>
    <variable name="k" units="per_second" initial_value="0.5" interface="public"/>
    <variable name="rate" units="per_second" interface="public"/>
    <variable name="y" units="per_second" interface="public"/>
    <variable name="e" units="per_second" initial_value="0" interface="public"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><ci>rate</ci><apply><times/><cn cellml:units="dimensionless">2</cn><ci>k</ci></apply></apply>
      <apply><eq/><ci>y</ci><apply><times/><ci>rate</ci><ci>x</ci></apply></apply>
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
        <apply><plus/><apply><minus/><ci>y</ci></apply><ci>e</ci></apply></apply>
    </math>
  </component>
  <units name="per_second"><unit units="second" exponent="-1"/></units>
</model>
"""


def analysed(external):
    model = lc.Parser().parseModel(MODEL)
    analyser = lc.Analyser()
    if external:
        analyser.addExternalVariable(lc.AnalyserExternalVariable(model.component("c").variable("e")))
    analyser.analyseModel(model)
    assert analyser.errorCount() == 0
    return analyser.analyserModel()


@pytest.mark.unit
def test_python_code_has_one_variables_array_with_each_variables_type():
    am = analysed(external=False)
    code = PythonGenerator._as_one_variables_array(
        lc.Generator().implementationCode(am, lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)))
    ns = {}
    exec(code, ns)
    info = {d["name"]: (i, d["type"].name) for i, d in enumerate(ns["VARIABLE_INFO"])}
    assert ns["VARIABLE_COUNT"] == len(info) == 4
    assert {n: t for n, (_, t) in info.items()} == {
        "k": "CONSTANT", "e": "CONSTANT", "rate": "COMPUTED_CONSTANT", "y": "ALGEBRAIC"}

    states, rates, variables = ns["create_states_array"](), ns["create_states_array"](), ns["create_variables_array"]()
    ns["initialise_variables"](states, rates, variables)
    ns["compute_computed_constants"](variables)
    ns["compute_rates"](0.0, states, rates, variables)
    ns["compute_variables"](0.0, states, rates, variables)
    assert variables[info["rate"][0]] == 1.0
    assert variables[info["y"][0]] == 3.0
    assert rates == [-3.0]


@pytest.mark.unit
def test_analyser_variables_are_indexed_like_the_one_array():
    am = analysed(external=True)
    names = [(av.variable().name(), index) for av, index in analyser_variables(am)]
    assert names == [("k", 0), ("rate", 1), ("y", 2), ("e", 3)]


DRIVER = r"""
#include <stdio.h>
#include "model.h"
static double e(double voi, double *states, double *rates, double *variables, size_t index) { return 0.25; }
int main(void) {
    double *s = createStatesArray(), *r = createStatesArray(), *v = createVariablesArray();
    initialiseVariables(0.0, s, r, v, e);
    computeComputedConstants(0.0, s, r, v);
    computeRates(0.0, s, r, v, e);
    computeVariables(0.0, s, r, v, e);
    for (size_t i = 0; i < VARIABLE_COUNT; ++i) printf("%s %d %g\n", VARIABLE_INFO[i].name, VARIABLE_INFO[i].type, v[i]);
    printf("rate %g\n", r[0]);
    return 0;
}
"""


@pytest.mark.unit
def test_c_code_has_one_variables_array_and_fetches_externals_into_it(tmp_path):
    am = analysed(external=True)
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.C)
    header = c_code_with_one_variables_array(lc.Generator().interfaceCode(am, profile), header=True)
    implementation = c_code_with_one_variables_array(lc.Generator().implementationCode(am, profile), header=False)
    assert "extern const size_t VARIABLE_COUNT;" in header and "VariableType type;" in header
    assert "void initialiseVariables(double voi, double *states, double *rates, double *variables, " \
           "ExternalVariable externalVariable);" in header
    for old in ("constants", "computedConstants", "algebraicVariables", "externalVariables", "initialiseArrays"):
        assert old not in header and old not in implementation
    assert "variables[3] = externalVariable(voi, states, rates, variables, 3);" in implementation

    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler to run the code")
    (tmp_path / "model.h").write_text(header)
    (tmp_path / "model.c").write_text(implementation)
    (tmp_path / "driver.c").write_text(DRIVER)
    subprocess.run([compiler, "-o", "run", "driver.c", "model.c", "-lm"], cwd=tmp_path, check=True)
    out = subprocess.run([os.path.join(tmp_path, "run")], capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[:5] == ["k 2 0.5", "rate 3 1", "y 4 3", "e 5 0.25", "rate -2.75"]


@pytest.mark.unit
def test_c_code_of_a_model_with_no_algebraic_variables_lists_every_variable_once():
    """libCellML writes an empty kind as "= {\\n};", here followed by the external variables' block."""
    model = lc.Parser().parseModel(MODEL.replace("<ci>rate</ci><ci>x</ci>", "<ci>rate</ci><ci>k</ci>"))
    analyser = lc.Analyser()
    analyser.addExternalVariable(lc.AnalyserExternalVariable(model.component("c").variable("e")))
    analyser.analyseModel(model)
    am = analyser.analyserModel()
    assert am.algebraicVariableCount() == 0
    code = c_code_with_one_variables_array(
        lc.Generator().implementationCode(am, lc.GeneratorProfile(lc.GeneratorProfile.Profile.C)), header=False)
    block = code[code.index("VARIABLE_INFO[] = {"):]
    block = block[:block.index("};")]
    assert re.findall(r'\{"(\w+)", "\w+", "c", (\w+)\}', block) == [
        ("k", "CONSTANT"), ("rate", "COMPUTED_CONSTANT"), ("y", "COMPUTED_CONSTANT"), ("e", "EXTERNAL")]
    assert block.count("{") == 5


# x starts at x0 = 2 k, a computed constant: libCellML 0.7 sets that state in computeComputedConstants
COMPUTED_INITIAL_VALUE = MODEL.replace(
    '<variable name="x" units="dimensionless" initial_value="3" interface="public"/>',
    '<variable name="x" units="dimensionless" initial_value="x0" interface="public"/>'
    '<variable name="x0" units="dimensionless" interface="public"/>').replace(
    '<apply><eq/><ci>rate</ci>',
    '<apply><eq/><ci>x0</ci><apply><times/><cn cellml:units="second">2</cn><ci>k</ci></apply></apply>'
    '<apply><eq/><ci>rate</ci>')

COMPUTED_INITIAL_DRIVER = r"""
#include <stdio.h>
#include "model.h"
int main(void) {
    double *s = createStatesArray(), *r = createStatesArray(), *v = createVariablesArray();
    initialiseVariables(s, r, v);
    computeComputedConstants(0.0, s, r, v);
    printf("%g\n", s[0]);
    return 0;
}
"""


@pytest.mark.unit
def test_c_code_sets_a_state_initialised_by_a_computed_constant(tmp_path):
    model = lc.Parser().parseModel(COMPUTED_INITIAL_VALUE)
    analyser = lc.Analyser()
    analyser.analyseModel(model)
    assert analyser.errorCount() == 0
    am = analyser.analyserModel()
    profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.C)
    header = c_code_with_one_variables_array(lc.Generator().interfaceCode(am, profile), header=True)
    implementation = c_code_with_one_variables_array(lc.Generator().implementationCode(am, profile), header=False)
    assert "void computeComputedConstants(double voi, double *states, double *rates, double *variables)" in header
    assert "states[0] = variables[" in implementation.split("void computeComputedConstants")[1]

    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler to run the code")
    (tmp_path / "model.h").write_text(header)
    (tmp_path / "model.c").write_text(implementation)
    (tmp_path / "driver.c").write_text(COMPUTED_INITIAL_DRIVER)
    subprocess.run([compiler, "-o", "run", "driver.c", "model.c", "-lm"], cwd=tmp_path, check=True)
    out = subprocess.run([os.path.join(tmp_path, "run")], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "1"
