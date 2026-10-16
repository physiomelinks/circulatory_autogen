"""A model whose parameters are hoisted the way another generator hoists them.

circulatory_autogen builds its flat CellML with the parameters collected into a
``parameters`` component and named ``<param>_<vessel>`` -- ``C_aortic_root``.
Nothing about CellML requires that, and PhLynx does not do it: it writes
``<component>_<param>`` (``soma_SN_c_ER``), and drops the prefix entirely for a
name that is already unique in the model (``g_Na``). Globals go to
``parameters_global``.

That difference is invisible until the model is loaded, because Myokit's CellML
importer **merges connected variables**: the copy in ``soma_SN`` and the copy in
``parameters`` become one variable, and the one that survives is the one holding
the definition. So after import ``soma_SN.g_Na`` is not a qname -- only
``parameters.g_Na`` is -- and every name the user wrote in their obs_data and
params_for_id resolves to nothing.

What that looked like from CUFLynx (#300): a study built in PhLynx loaded, its
sliders appeared, and the run failed with "Pacing parameter soma_SN/I_in must
resolve to a valid variable", the valid list being 500 names none of which the
user had ever typed.
"""

import libcellml
import myokit
import numpy as np
import pytest

from libcuflynx.parsers.PrimitiveParsers import model_qname_candidates
from libcuflynx.solver_wrappers.myokit_helper import SimulationHelper
from libcuflynx.solver_wrappers.name_resolver import VariableNameResolver

#: A two-parameter model in PhLynx's shape: one parameter hoisted with its
#: component as a prefix, one hoisted bare, one global, and a state whose initial
#: value is a *reference* to a hoisted parameter rather than a number.
HOISTED_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#"
       xmlns:cellml="http://www.cellml.org/cellml/2.0#"
       name="HoistedParameters">
  <units name="per_second"><unit exponent="-1" units="second"/></units>
  <units name="mV"><unit prefix="milli" units="volt"/></units>

  <component name="environment">
    <variable name="t" units="second" interface="public"/>
  </component>

  <component name="parameters_global">
    <variable name="R" units="dimensionless" initial_value="2" interface="public"/>
  </component>

  <component name="parameters">
    <variable name="soma_k" units="per_second" initial_value="3" interface="public"/>
    <variable name="g" units="dimensionless" initial_value="5" interface="public"/>
    <variable name="soma_V_rest" units="mV" initial_value="-70" interface="public"/>
  </component>

  <component name="soma">
    <variable name="t" units="second" interface="public"/>
    <variable name="R" units="dimensionless" interface="public"/>
    <variable name="k" units="per_second" interface="public"/>
    <variable name="g" units="dimensionless" interface="public"/>
    <variable name="V_rest" units="mV" interface="public"/>
    <variable name="V" units="mV" initial_value="V_rest"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/>
        <apply><diff/><bvar><ci>t</ci></bvar><ci>V</ci></apply>
        <apply><times/>
          <apply><times/><ci>k</ci><ci>g</ci></apply>
          <apply><divide/>
            <apply><minus/><cn cellml:units="mV">0</cn><ci>V</ci></apply>
            <ci>R</ci>
          </apply>
        </apply>
      </apply>
    </math>
  </component>

  <connection component_1="environment" component_2="soma">
    <map_variables variable_1="t" variable_2="t"/>
  </connection>
  <connection component_1="parameters_global" component_2="soma">
    <map_variables variable_1="R" variable_2="R"/>
  </connection>
  <connection component_1="parameters" component_2="soma">
    <map_variables variable_1="soma_k" variable_2="k"/>
    <map_variables variable_1="g" variable_2="g"/>
    <map_variables variable_1="soma_V_rest" variable_2="V_rest"/>
  </connection>
</model>
"""


@pytest.fixture(scope="module")
def hoisted_helper(tmp_path_factory):
    path = tmp_path_factory.mktemp("hoisted") / "hoisted.cellml"
    path.write_text(HOISTED_CELLML)
    return SimulationHelper(str(path), 0.01, 1.0, pre_time=0.0)


# ---------------------------------------------------------------------------
# The alias table
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "written,qname",
    [
        ("soma/k", "parameters.soma_k"),           # component-first prefix
        ("soma/g", "parameters.g"),                # hoisted with no prefix
        ("soma/R", "parameters_global.R"),         # hoisted global constant
    ],
)
def test_a_hoisted_parameter_resolves_to_where_the_model_kept_it(written, qname):
    """The three forms this repository did not previously know about."""
    qnames = {"parameters.soma_k": 1, "parameters.g": 1, "parameters_global.R": 1}
    assert VariableNameResolver.resolve_key(
        written, [("var", qnames)], separator="."
    ) == ("var", qname)


@pytest.mark.unit
def test_this_repositorys_own_convention_still_wins():
    """Ordering, not coverage, is what keeps the CA models reading the CA way.

    A model that happens to satisfy both conventions -- ``C_aortic_root`` and
    ``aortic_root_C`` both present -- must still be read as circulatory_autogen
    writes it, because that is the one this repository generated.
    """
    qnames = {"parameters.C_aortic_root": 1, "parameters.aortic_root_C": 1}
    assert VariableNameResolver.resolve_key(
        "aortic_root/C", [("var", qnames)], separator="."
    ) == ("var", "parameters.C_aortic_root")


@pytest.mark.unit
def test_a_bare_name_elsewhere_does_not_answer_for_a_hoisted_one():
    """``parameters/Var`` is tried before the bare name, and only then.

    Without the ordering, ``soma/g`` in a model with a ``membrane.g`` and no
    hoisted ``g`` would silently resolve to the membrane's -- a wrong answer is
    worse here than no answer, because it calibrates the wrong variable.
    """
    qnames = {"membrane.g": 1}
    kind, qname = VariableNameResolver.resolve_key(
        "soma/g", [("var", qnames)], separator="."
    )
    assert (kind, qname) == (None, None)


# ---------------------------------------------------------------------------
# The same names through a loaded model
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize(
    "written,qname",
    [
        ("soma/k", "parameters.soma_k"),
        ("soma/g", "parameters.g"),
        ("soma/R", "parameters_global.R"),
        ("soma/V_rest", "parameters.soma_V_rest"),
    ],
)
def test_the_loaded_model_answers_to_the_names_the_user_wrote(
    hoisted_helper, written, qname
):
    """The end the user meets: names from their params_for_id, against the model.

    ``soma.k`` does not exist after import -- Myokit merged it away -- so this is
    the assertion that a study written against the CellML resolves against the
    Myokit model it became.
    """
    assert hoisted_helper._resolve_name(written) == ("var", qname)


@pytest.mark.integration
def test_an_initial_value_that_names_a_hoisted_parameter_survives_import(
    hoisted_helper,
):
    """``initial_value="V_rest"`` must keep pointing at the parameter.

    The importer's own component is the wrong place to look for it once the
    variables are merged, and the old code looked only there: the reference came
    out as ``soma.V_rest``, Myokit could not resolve it, and the step was
    abandoned with a printed warning. The state then kept a *numeric* initial
    value, so calibrating ``V_rest`` moved nothing.
    """
    state = next(s for s in hoisted_helper.model.states() if s.qname() == "soma.V")
    assert str(state.initial_value()) == "parameters.soma_V_rest"


@pytest.mark.integration
def test_the_model_runs_and_the_hoisted_parameters_reach_the_solver(hoisted_helper):
    """Resolution is not the point on its own -- setting the value is."""
    def final_V(g):
        hoisted_helper.reset_and_clear()
        hoisted_helper.set_param_vals(["soma/g"], [g])
        assert hoisted_helper.run()
        # (variable, sub-experiment, time) -- one of each here.
        return np.asarray(hoisted_helper.get_results(["soma/V"])).ravel()[-1]

    fast, frozen = final_V(5.0), final_V(0.0)

    # g scales the whole right-hand side, so g=0 holds V at its initial value
    # and g=5 decays it towards zero. Anything else means the write did not land.
    assert abs(frozen - (-70.0)) < 1e-6
    assert fast > frozen + 1.0


# ---------------------------------------------------------------------------------------------
# PhLynx v0.3.0 on: instance_parameters / global_parameters (issue #558). Every reader of
# hoisted names takes them from parameter_names.HOISTED_PARAMETER_CONVENTIONS.
# ---------------------------------------------------------------------------------------------

# Flattened as PhLynx exports: soma_SN and other share g_M; only soma_SN has tau.
_MMT = """
[[model]]
soma_SN.V = -65
other.V = -65
[environment]
time = 0 [ms] bind time
[instance_parameters]
soma_SN_g_M = 0.1
other_g_M = 0.2
tau = 5
[global_parameters]
I_in = 0
[soma_SN]
dot(V) = (-instance_parameters.soma_SN_g_M * V + global_parameters.I_in) / instance_parameters.tau
[other]
dot(V) = -instance_parameters.other_g_M * V
"""

_EXPECTED = {
    "soma_SN/g_M": "instance_parameters.soma_SN_g_M",
    "soma_SN/tau": "instance_parameters.tau",
    "soma_SN/I_in": "global_parameters.I_in",
    "other/g_M": "instance_parameters.other_g_M",
}

# The same model as CellML, before flattening: soma_SN's inputs are connected to the
# components that define them.
_CELLML = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" name="phlynx_export">
  <component name="environment">
    <variable name="time" units="second" interface="public_and_private"/>
  </component>
  <component name="instance_parameters">
    <variable name="soma_SN_g_M" units="dimensionless" initial_value="0.1" interface="public"/>
    <variable name="tau" units="second" initial_value="5" interface="public"/>
  </component>
  <component name="global_parameters">
    <variable name="I_in" units="dimensionless" initial_value="0" interface="public"/>
  </component>
  <component name="soma_SN">
    <variable name="time" units="second" interface="public_and_private"/>
    <variable name="g_M" units="dimensionless" interface="public_and_private"/>
    <variable name="tau" units="second" interface="public_and_private"/>
    <variable name="I_in" units="dimensionless" interface="public_and_private"/>
    <variable name="V" units="dimensionless" initial_value="-65"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>time</ci></bvar><ci>V</ci></apply>
        <apply><divide/>
          <apply><plus/><apply><times/><apply><minus/><ci>g_M</ci></apply><ci>V</ci></apply>
            <ci>I_in</ci></apply>
          <ci>tau</ci></apply></apply>
    </math>
  </component>
  <connection component_1="environment" component_2="soma_SN">
    <map_variables variable_1="time" variable_2="time"/>
  </connection>
  <connection component_1="instance_parameters" component_2="soma_SN">
    <map_variables variable_1="soma_SN_g_M" variable_2="g_M"/>
    <map_variables variable_1="tau" variable_2="tau"/>
  </connection>
  <connection component_1="global_parameters" component_2="soma_SN">
    <map_variables variable_1="I_in" variable_2="I_in"/>
  </connection>
</model>
"""


def _keyed():
    m = myokit.parse_model(_MMT)
    return [("state", {v.qname(): i for i, v in enumerate(m.states())}),
            ("var", {v.qname(): v for v in m.variables(deep=True)})]


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_phlynx_instance_and_global_parameters_resolve(name):
    assert VariableNameResolver.resolve_key(name, _keyed(), separator=".") == (
        "var", _EXPECTED[name])


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_model_qname_candidates_agree_with_resolver(name):
    """CUFLynx picks the first candidate the model has; that must be what the solvers use."""
    present = {qname.replace(".", "/") for _, d in _keyed() for qname in d}
    first = next(c for c in model_qname_candidates(name) if c in present)
    assert first == _EXPECTED[name].replace(".", "/")


@pytest.mark.unit
def test_calibrated_values_written_into_instance_parameters(tmp_path):
    from libcuflynx.utilities.utility_funcs import change_parameter_values_and_save

    src, out = tmp_path / "phlynx_export.cellml", tmp_path / "calibrated.cellml"
    src.write_text(_CELLML)
    change_parameter_values_and_save(str(src), ["soma_SN/g_M", "soma_SN/tau", "soma_SN/I_in"],
                                     [0.7, 9.0, 2.0], str(out))

    model = libcellml.Parser(False).parseModel(out.read_text())
    value = lambda comp, var: model.component(comp).variable(var).initialValue()
    assert float(value("instance_parameters", "soma_SN_g_M")) == 0.7
    assert float(value("instance_parameters", "tau")) == 9.0
    assert float(value("global_parameters", "I_in")) == 2.0


@pytest.mark.unit
def test_variable_mapping_reads_instance_parameters(tmp_path):
    from libcuflynx.reporting.variable_mapping import default_variable_mapping

    path = tmp_path / "phlynx_export.cellml"
    path.write_text(_CELLML)
    rows = {r["variable_name"]: r for r in default_variable_mapping(str(path))}

    for name in ("instance_parameters/soma_SN_g_M", "instance_parameters/tau",
                 "global_parameters/I_in"):
        assert rows[name]["kind"] == "parameter", name
    assert rows["soma_SN/V"]["kind"] == "state"
    # a hoisted name with the instance first is shown as the variable of that instance
    assert rows["instance_parameters/soma_SN_g_M"]["latex"] == r"g_{M,\mathrm{soma\_SN}}"
