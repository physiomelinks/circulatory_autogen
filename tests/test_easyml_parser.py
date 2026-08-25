"""Reading openCARP's EasyML ``.model`` files.

The strongest check here is the round trip: Myokit ships an EasyML *exporter*,
so exporting a model and reading it back must reproduce the same equations. That
compares the importer against an implementation nobody involved in it wrote, on
real published ionic models, rather than against fixtures chosen to suit it.

Two things it cannot check, because they are lost when EasyML is *written* and
are not in the file for any reader to recover -- see
``test_the_round_trip_is_exact_except_where_easyml_cannot_say`` for both.
"""

import os
import tempfile

import pytest

from libcuflynx.parsers.EasyMLParsers import (
    EasyMLImportError,
    cellml_from_easyml,
    is_easyml_filename,
    looks_like_easyml,
    parse_easyml,
    protocol_info_from_easyml,
)

myokit = pytest.importorskip("myokit")


# A whole model in miniature: an external V, a current sum, one explicit state,
# one tau/inf gate, one alpha/beta gate, a parameter and a method group.
TINY = b"""
/*
name: Tiny
desc: a two-gate cell
*/
V; .nodal(); .external(Vm);
Iion; .nodal(); .external();

V_init = -80.0;
Ca_init = 0.0002;
m_init = 0.001;

g_Na = 16.0; .param();

m_inf = 1.0 / (1.0 + exp(-(V + 40.0) / 10.0));
tau_m = 0.1;

alpha_h = 0.07 * exp(-(V + 80.0) / 20.0);
beta_h = 1.0 / (1.0 + exp(-(V + 50.0) / 10.0));

INa = g_Na * m * m * m * h * (V - 50.0);
diff_Ca = -0.001 * INa - 0.01 * Ca;

Iion = INa;

group {
  m;
  h;
}.method(rush_larsen);

group {
  Ca;
}.method(cvode);

group {
  INa;
  V;
}.trace();
"""


def parse(data=TINY, **kw):
    return parse_easyml(data, filename=kw.pop("filename", "tiny.model"), **kw)


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def test_recognises_the_extension():
    assert is_easyml_filename("Courtemanche.model")
    assert is_easyml_filename("/tmp/UPPER.MODEL")
    assert not is_easyml_filename("model.cellml")


def test_recognises_the_content():
    assert looks_like_easyml(TINY)


def test_a_cellml_file_named_model_is_not_easyml():
    """``.model`` is a generic suffix; other tools use it for unrelated files."""
    assert not looks_like_easyml(b'<?xml version="1.0"?><model name="x"/>')


def test_a_file_with_no_markup_is_still_easyml_if_it_declares_an_init():
    assert looks_like_easyml(b"V_init = -80.0;\ndiff_V = 1.0;\n")


def test_prose_is_not_easyml():
    assert not looks_like_easyml(b"This is a note about a model.\n")


# ---------------------------------------------------------------------------
# The implicit parts: gates, initial values, the membrane equation
# ---------------------------------------------------------------------------
def test_a_tau_inf_pair_becomes_a_state():
    r = parse()
    m = r.model.get(f"{r.model.name()}.m")
    assert m.is_state()
    assert "m_inf" in str(m.rhs()) and "tau_m" in str(m.rhs())


def test_an_alpha_beta_pair_becomes_a_state():
    r = parse()
    h = r.model.get(f"{r.model.name()}.h")
    assert h.is_state()
    # alpha * (1 - h) - beta * h
    assert "alpha_h" in str(h.rhs()) and "beta_h" in str(h.rhs())


def test_a_gate_without_an_init_starts_at_its_steady_state():
    """openCARP generates the missing ``X_init``; starting at 0 is a different
    simulation for the first beats, so this is not a detail."""
    r = parse()
    h = r.model.get(f"{r.model.name()}.h")
    alpha = r.model.get(f"{r.model.name()}.alpha_h").rhs().eval()
    beta = r.model.get(f"{r.model.name()}.beta_h").rhs().eval()
    assert h.initial_value(True) == pytest.approx(alpha / (alpha + beta))
    assert any("steady state" in w for w in r.warnings)


def test_an_explicit_init_is_used_as_written():
    r = parse()
    assert r.model.get(f"{r.model.name()}.m").initial_value(True) == pytest.approx(0.001)


def test_a_diff_statement_becomes_that_states_equation():
    r = parse()
    ca = r.model.get(f"{r.model.name()}.Ca")
    assert ca.is_state()
    assert "INa" in str(ca.rhs())


def test_the_membrane_equation_is_synthesised_and_reported():
    r = parse()
    v = r.model.get(f"{r.model.name()}.V")
    assert v.is_state()
    assert r.synthesised_membrane
    assert str(v.rhs()) == f"-({r.model.name()}.Iion + {r.model.name()}.{r.stimulus_name})"
    assert any("carries no membrane equation" in w for w in r.warnings)


def test_the_stimulus_is_a_plain_variable_a_protocol_can_drive():
    """Not a state: CA drives a protocol by setting a parameter between
    sub-experiments, and a state is integrated rather than set."""
    r = parse()
    stim = r.model.get(f"{r.model.name()}.{r.stimulus_name}")
    assert not stim.is_state()
    assert stim.rhs().eval() == 0


def test_a_model_that_only_adds_to_a_current_is_refused_by_name():
    plugin = b"V; .nodal(); .external(Vm);\nV_init = -80;\nI_extra = 0.1 * V;\n"
    with pytest.raises(EasyMLImportError, match="plugin"):
        parse_easyml(plugin, filename="plugin.model")


# ---------------------------------------------------------------------------
# The .method() groups are reported, not executed
# ---------------------------------------------------------------------------
def test_a_fixed_step_method_is_reported_as_a_warning():
    r = parse()
    warned = [w for w in r.warnings if "rush_larsen" in w]
    assert warned, r.warnings
    assert "m" in warned[0] and "h" in warned[0]
    assert "discretisation" in warned[0]


def test_cvode_groups_are_not_warned_about():
    """They already mean "as accurately as the solver can"."""
    r = parse()
    assert not any("cvode" in w for w in r.warnings)


def test_the_methods_are_still_reported_as_data():
    r = parse()
    assert r.methods["m"] == "rush_larsen"
    assert r.methods["Ca"] == "cvode"


def test_param_and_trace_groups_are_carried_out():
    r = parse()
    assert "g_Na" in r.parameters
    assert set(r.traces) == {"INa", "V"}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr, expected",
    [
        (b"x = 2.0 + 3.0 * 4.0;", 14.0),
        (b"x = (2.0 + 3.0) * 4.0;", 20.0),
        (b"x = -2.0 ^ 2.0;", -4.0),           # unary binds looser than power
        (b"x = pow(2.0, 10.0);", 1024.0),
        (b"x = expm1(0.0);", 0.0),            # the exporter's exp(x) - 1
        (b"x = fabs(-3.0);", 3.0),
        (b"x = sqrt(9.0);", 3.0),
        (b"x = (1.0 > 2.0) ? 10.0 : 20.0;", 20.0),
        (b"x = 7.0 % 4.0;", 3.0),
        (b"x = max(3.0, 5.0);", 5.0),
        (b"x = 1.0 < 2.0 && 3.0 > 2.0 ? 1.0 : 0.0;", 1.0),
    ],
)
def test_expressions_evaluate_as_c_does(expr, expected):
    src = b"V; .nodal(); .external(Vm);\nIion; .nodal(); .external();\n" \
          b"V_init = -80;\ndiff_y = 1.0;\ny_init = 0;\nIion = y;\n" + expr
    r = parse_easyml(src, filename="e.model")
    assert r.model.get(f"{r.model.name()}.x").rhs().eval() == pytest.approx(expected)


def test_an_unknown_function_is_named_rather_than_guessed():
    src = b"V_init = -80;\ndiff_y = wobble(1.0);\ny_init = 0;\n"
    with pytest.raises(EasyMLImportError, match="wobble"):
        parse_easyml(src, filename="e.model")


def test_a_conditional_block_becomes_a_conditional_expression():
    src = (b"V; .nodal(); .external(Vm);\nIion; .nodal(); .external();\nV_init = -80;\n"
           b"diff_y = 1.0;\ny_init = 0;\nIion = y;\n"
           b"if (V > 0.0) { z = 1.0; } else { z = 2.0; }\n")
    r = parse_easyml(src, filename="e.model")
    assert r.model.get(f"{r.model.name()}.z").rhs().eval() == pytest.approx(2.0)


def test_if_is_a_legal_variable_name():
    """The Noble 1962 funny current exports as a variable literally called
    ``if``; deciding on the word alone would make that file a syntax error."""
    src = (b"V; .nodal(); .external(Vm);\nIion; .nodal(); .external();\nV_init = -80;\n"
           b"diff_y = 1.0;\ny_init = 0;\nif = 3.0;\nIion = y + if;\n")
    r = parse_easyml(src, filename="e.model")
    assert r.model.get(f"{r.model.name()}.if").rhs().eval() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_name_assigned_twice_is_refused():
    src = b"V_init = -80;\ndiff_y = 1.0;\ny_init = 0;\nz = 1.0;\nz = 2.0;\n"
    with pytest.raises(EasyMLImportError, match="assigned more than once"):
        parse_easyml(src, filename="e.model")


def test_an_undefined_name_is_named():
    src = b"V_init = -80;\ndiff_y = nowhere * 2.0;\ny_init = 0;\n"
    with pytest.raises(EasyMLImportError, match="nowhere"):
        parse_easyml(src, filename="e.model")


def test_reading_opencarps_timestep_is_refused_with_the_reason():
    """``dt`` only exists inside openCARP's fixed-step solver."""
    src = b"V_init = -80;\ndiff_y = 1.0 / dt;\ny_init = 0;\n"
    with pytest.raises(EasyMLImportError, match="fixed-step"):
        parse_easyml(src, filename="e.model")


def test_an_empty_assignment_names_the_variable():
    """Myokit's own exporter writes ``Iion = ;`` for a model whose membrane
    currents it could not identify, so this file shape really occurs."""
    src = b"V; .nodal(); .external(Vm);\nV_init = -80;\nIion = ;\n"
    with pytest.raises(EasyMLImportError, match="Iion is assigned nothing"):
        parse_easyml(src, filename="e.model")


def test_a_file_with_no_equations_is_refused():
    with pytest.raises(EasyMLImportError, match="no equations"):
        parse_easyml(b"// just a comment\n", filename="e.model")


# ---------------------------------------------------------------------------
# CellML out
# ---------------------------------------------------------------------------
def test_the_conversion_produces_cellml_and_keeps_a_copy():
    with tempfile.TemporaryDirectory() as td:
        cellml, saved, warnings = cellml_from_easyml(
            TINY, filename="tiny.model", out_dir=td)
        assert cellml.lstrip().startswith(b"<?xml")
        assert b"cellml" in cellml
        assert saved == os.path.join(td, "tiny.cellml")
        assert open(saved, "rb").read() == cellml
    assert warnings, "the membrane equation was synthesised; that must be said"


def test_the_component_is_named_after_the_model():
    """CA addresses everything as ``component/variable``; EasyML is flat, so the
    file's own name is what makes those names meaningful."""
    r = parse()
    assert r.model.name() == "Tiny"
    assert r.model.get("Tiny.g_Na") is not None


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------
def _mmt_dir():
    return os.path.join(os.path.dirname(myokit.__file__), "tests", "data")


def _reference(orig):
    """The model the exporter actually writes out, so the comparison is fair.

    The exporter rewrites Markov models into a compact form and strips the
    embedded stimulus before writing anything; those are its transformations,
    not the importer's to undo.
    """
    import myokit.lib.guess as guess
    import myokit.lib.markov as markov

    ref = markov.convert_markov_models_to_compact_form(orig)
    guess.remove_embedded_protocol(ref)
    for label in ("pace", "diffusion_current"):
        var = ref.binding(label)
        if var is not None and not var.is_state():
            var.set_rhs(0)
    return ref


def _derivatives(model):
    evaluate = getattr(model, "evaluate_derivatives", None) or model.eval_state_derivatives
    return dict(zip([v.name() for v in model.states()], evaluate()))


#: Models whose EasyML export is exact, and which therefore must import exactly.
ROUND_TRIP = [
    "beeler-1977-model.mmt",
    "beeler-1977-units.mmt",
    "beeler-1977-model-compare-a.mmt",
    "conditional.mmt",
    "cv1d.mmt",
    "decker-2009.mmt",       # 46 states, two Markov models, ternaries, expm1
    "dn-1985-normalised.mmt",  # exports a variable named `if`
    "lr-1991.mmt",
    "lr-1991-testing.mmt",
    "noble-1962.mmt",
]


@pytest.mark.parametrize("name", ROUND_TRIP)
def test_the_round_trip_reproduces_the_equations(name):
    path = os.path.join(_mmt_dir(), name)
    if not os.path.exists(path):  # pragma: no cover - myokit trimmed its fixtures
        pytest.skip(f"myokit no longer ships {name}")
    orig = myokit.load_model(path)

    from myokit.formats.easyml import EasyMLExporter

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "m.model")
        EasyMLExporter().model(out, orig)
        data = open(out, "rb").read()

    imported = parse_easyml(data, filename="m.model")
    ref, got = _derivatives(_reference(orig)), _derivatives(imported.model)
    common = sorted(set(ref) & set(got))
    assert len(common) >= orig.count_states() // 2, "too few states matched by name"
    for state in common:
        assert got[state] == pytest.approx(ref[state], rel=1e-12, abs=1e-14), state


def test_the_round_trip_is_exact_except_where_easyml_cannot_say():
    """The two things a ``.model`` file cannot carry, recorded so a future
    failure here is read as what it is.

    ``lr-1991-dep`` writes ``dot(V) = i_stim - (INa + ...)``: its stimulus enters
    with the opposite sign to the other currents. EasyML has no place to say
    that -- ``Iion`` is a plain sum and openCARP fixes ``dV/dt = -Iion`` -- so
    the export negates it.

    ``beeler-1977-model-compare-b`` has neither a unit on V nor a capacitance the
    exporter can find, so its currents cannot be converted to A/F and the
    ``1/C`` factor in ``dot(V)`` is dropped on the way out.

    Neither is recoverable by any reader of the resulting file, and neither
    affects a model *written* in EasyML, which is the case that matters.
    """
    import myokit.lib.guess as guess

    from myokit.formats.easyml import EasyMLExporter

    for name, why in [("lr-1991-dep.mmt", "sign"), ("beeler-1977-model-compare-b.mmt", "1/C")]:
        path = os.path.join(_mmt_dir(), name)
        if not os.path.exists(path):  # pragma: no cover
            continue
        orig = myokit.load_model(path)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "m.model")
            EasyMLExporter().model(out, orig)
            data = open(out, "rb").read()
        imported = parse_easyml(data, filename="m.model")
        ref = _reference(orig)
        v_ref = guess.membrane_potential(ref).rhs().eval()
        v_got = imported.model.get(
            f"{imported.model.name()}.{imported.v_name}").rhs().eval()
        assert v_got != pytest.approx(v_ref), (
            f"{name} now round-trips exactly ({why} is no longer lost); "
            f"move it into ROUND_TRIP"
        )


# ---------------------------------------------------------------------------
# End to end: an imported model that fires
# ---------------------------------------------------------------------------
def test_the_files_own_t_is_left_alone():
    """``t`` is EasyML's absolute time, but Myokit's exporter will happily emit
    an ordinary variable called ``t`` -- decker.model has one. The file's
    spelling wins and the time variable steps aside, so neither is lost."""
    src = (b"V; .nodal(); .external(Vm);\nIion; .nodal(); .external();\nV_init = -80;\n"
           b"t = 3.0;\ndiff_y = t;\ny_init = 0;\nIion = y;\n")
    r = parse_easyml(src, filename="e.model")
    assert r.model.get(f"{r.model.name()}.t").rhs().eval() == pytest.approx(3.0)
    assert r.model.time().name() != "t"


def test_a_synthesised_stimulus_paces_the_model():
    info, notes = protocol_info_from_easyml(parse())
    assert list(info["params_to_change"]) == ["Tiny/i_stim"]
    assert info["sim_times"] == [[2000.0]]
    assert any("carries no stimulus" in n for n in notes)


def test_a_model_that_owns_its_membrane_potential_gets_no_stimulus():
    src = (b"V_init = -80;\ndiff_V = 0.1;\nIion = 0.0;\n")
    with pytest.raises(EasyMLImportError, match="integrates its own"):
        protocol_info_from_easyml(parse_easyml(src, filename="e.model"))


def test_an_imported_model_produces_an_action_potential():
    """The whole point, end to end: read a model out of EasyML, pace it, and get
    the upstroke and repolarisation the source publication describes."""
    path = os.path.join(_mmt_dir(), "lr-1991.mmt")
    if not os.path.exists(path):  # pragma: no cover
        pytest.skip("myokit no longer ships lr-1991.mmt")
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        pytest.skip("numpy not available")

    from myokit.formats.easyml import EasyMLExporter

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "m.model")
        EasyMLExporter().model(out, myokit.load_model(path))
        data = open(out, "rb").read()

    r = parse_easyml(data, filename="m.model")
    model = r.model
    model.get(f"{model.name()}.{r.stimulus_name}").set_binding("pace")
    protocol = myokit.Protocol()
    protocol.schedule(-80, 5, 1, 1000, 0)
    try:
        sim = myokit.Simulation(model, protocol)
        log = sim.run(500, log=[model.time().qname(), f"{model.name()}.{r.v_name}"])
    except Exception as exc:  # pragma: no cover - no compiler in this environment
        pytest.skip(f"myokit cannot compile a simulation here: {exc}")

    t = np.array(log[model.time().qname()])
    v = np.array(log[f"{model.name()}.{r.v_name}"])
    assert -95 < v[0] < -75, f"resting potential {v[0]:.1f} mV"
    assert v.max() > 0, f"no upstroke; peak {v.max():.1f} mV"
    assert np.max(np.diff(v) / np.diff(t)) > 50, "upstroke too slow to be an AP"
    # And it repolarises rather than sitting depolarised.
    assert v[-1] < v[0] + 15
