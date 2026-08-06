"""The params_for_id `prior` column is a declared vocabulary, not free text.

An unrecognised prior used to be accepted: the column was read straight to a numpy
array, and `get_lnprior_from_params` matched it against 'uniform'/'exponential'/
'normal' and fell through every branch when it matched none. Falling through skips
that parameter's own range check, so a mis-spelled prior stopped bounding the
parameter at all -- an MCMC walker could leave [min, max] with a finite lnprior
instead of -inf, silently. No model/MPI needed.
"""
import io

import numpy as np
import pandas as pd
import pytest

from parsers.PrimitiveParsers import (DEFAULT_PARAM_PRIOR_TYPE, PARAM_PRIOR_TYPES,
                                      ObsAndParamDataParser, normalise_prior_type)


def _info(csv):
    return ObsAndParamDataParser()._build_param_id_info_from_df(pd.read_csv(io.StringIO(csv)))


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------
def test_every_declared_prior_is_handled_by_the_likelihood():
    """PARAM_PRIOR_TYPES and get_lnprior_from_params must not drift apart: a prior
    the schema advertises but the likelihood cannot evaluate would raise mid-run."""
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    for prior in PARAM_PRIOR_TYPES:
        pid.param_id_info = {
            "param_prior_types": np.array([prior]),
            "param_mins": np.array([1.0]),
            "param_maxs": np.array([2.0]),
        }
        assert np.isfinite(pid.get_lnprior_from_params([1.5])), f"{prior} not handled"


def test_the_default_is_one_of_the_declared_priors():
    assert DEFAULT_PARAM_PRIOR_TYPE in PARAM_PRIOR_TYPES


# ---------------------------------------------------------------------------
# Normalisation and validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["Normal", " normal ", "NORMAL"])
def test_case_and_whitespace_are_normalised(raw):
    """'Normal' unambiguously means 'normal'. Accepting it is friendlier than
    erroring, and far better than the old behaviour of taking it verbatim and
    unbounding the parameter."""
    assert normalise_prior_type(raw) == "normal"


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_a_blank_prior_means_the_default(blank):
    assert normalise_prior_type(blank) == DEFAULT_PARAM_PRIOR_TYPE


def test_an_unknown_prior_is_rejected_with_the_valid_ones_named():
    with pytest.raises(ValueError, match="unknown prior 'gaussian'"):
        normalise_prior_type("gaussian")
    with pytest.raises(ValueError, match="exponential, normal, uniform"):
        normalise_prior_type("gaussian")


def test_the_offending_row_is_named():
    with pytest.raises(ValueError, match="row 3"):
        normalise_prior_type("gaussian", row_idx=3)


# ---------------------------------------------------------------------------
# Through the params_for_id parser
# ---------------------------------------------------------------------------
def test_the_prior_column_is_canonicalised():
    info = _info("vessel_name,param_name,min,max,prior\na,k,1,2,Normal\nb,j,1,2,\n")
    assert list(info["param_prior_types"]) == ["normal", DEFAULT_PARAM_PRIOR_TYPE]


def test_no_prior_column_means_the_default_throughout():
    info = _info("vessel_name,param_name,min,max\na,k,1,2\nb,j,3,4\n")
    assert list(info["param_prior_types"]) == [DEFAULT_PARAM_PRIOR_TYPE] * 2


def test_an_unusable_prior_fails_the_parse_not_the_run():
    with pytest.raises(ValueError, match="unknown prior 'lognormal'"):
        _info("vessel_name,param_name,min,max,prior\na,k,1,2,lognormal\n")


# ---------------------------------------------------------------------------
# The bug itself
# ---------------------------------------------------------------------------
def test_a_mis_spelled_prior_no_longer_unbounds_the_parameter():
    """The regression this all exists for. 'Normal' used to survive the parser
    verbatim, match no branch, and leave the parameter unbounded: lnprior was 0
    at a value far outside [min, max] where it must be -inf."""
    from param_id.paramID import OpencorParamID

    info = _info("vessel_name,param_name,min,max,prior\na,k,1,2,Normal\n")
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info

    assert pid.get_lnprior_from_params([999.0]) == -np.inf


def test_a_prior_the_parser_never_saw_raises_rather_than_falling_through():
    """Defence in depth for param_id_info built by hand rather than parsed."""
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = {
        "param_prior_types": np.array(["gaussian"]),
        "param_mins": np.array([1.0]),
        "param_maxs": np.array([2.0]),
    }
    with pytest.raises(ValueError, match="unknown prior 'gaussian'"):
        pid.get_lnprior_from_params([1.5])


# ---------------------------------------------------------------------------
# The values each prior takes
#
# These were hardcoded in get_lnprior_from_params -- the exponential's rate behind
# a "TODO make this user modifiable", the normal's mean and std behind a
# "temporarily". A prior whose centre is fixed to the middle of the bounds cannot
# express most of what a prior is for, and nothing told the user the number was
# fixed. They are declared in the schema and settable per row now.
# ---------------------------------------------------------------------------
from parsers.PrimitiveParsers import PARAM_PRIOR_PARAM_NAMES, normalise_prior_params


def _lnprior(csv, vals):
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = _info(csv)
    return pid.get_lnprior_from_params(vals)


def test_every_declared_prior_param_is_a_known_column():
    assert set(PARAM_PRIOR_PARAM_NAMES) == {
        "prior_lambda", "prior_origin", "prior_scale", "prior_mean", "prior_std"}


def test_each_prior_declares_the_values_it_takes():
    assert [s["name"] for s in PARAM_PRIOR_TYPES["exponential"]["params"]] == [
        "prior_lambda", "prior_origin", "prior_scale"]
    assert [s["name"] for s in PARAM_PRIOR_TYPES["normal"]["params"]] == ["prior_mean", "prior_std"]
    assert PARAM_PRIOR_TYPES["uniform"]["params"] == []


def test_an_unstated_value_falls_back_to_the_declared_default():
    assert normalise_prior_params("exponential", {})["prior_lambda"] == 1.0
    # None means "derived from the bounds", which this function does not own.
    assert normalise_prior_params("normal", {})["prior_std"] is None


def test_a_stated_value_is_read():
    got = normalise_prior_params("normal", {"prior_mean": 3.0, "prior_std": 0.5})
    assert got == {"prior_mean": 3.0, "prior_std": 0.5}


def test_a_value_for_a_prior_that_ignores_it_is_rejected():
    """Silently dropping it would give the user a different posterior than the one
    they believe they asked for."""
    with pytest.raises(ValueError, match="does not use it"):
        normalise_prior_params("uniform", {"prior_std": 2.0})


def test_a_non_positive_scale_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        normalise_prior_params("normal", {"prior_std": 0.0})
    with pytest.raises(ValueError, match="greater than zero"):
        normalise_prior_params("exponential", {"prior_lambda": -1.0})


def test_a_mean_may_be_negative_or_zero():
    """Only the scales are constrained; a centre is free."""
    assert normalise_prior_params("normal", {"prior_mean": -4.0})["prior_mean"] == -4.0


def test_a_non_numeric_value_is_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        normalise_prior_params("normal", {"prior_std": "wide"})


def test_the_columns_are_parsed_per_row():
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_mean,prior_std\n"
        "a,k,0,10,normal,7.0,0.5\n"
        "b,j,0,10,uniform,,\n"
    )
    assert info["param_prior_params"][0] == {"prior_mean": 7.0, "prior_std": 0.5}
    assert info["param_prior_params"][1] == {}


def test_the_stated_mean_moves_the_peak():
    """The regression that motivates this: with mean fixed to the range centre the
    prior peaked at 5; stating 9 must move it there."""
    csv = ("vessel_name,param_name,min,max,prior,prior_mean,prior_std\n"
           "a,k,0,10,normal,9.0,1.0\n")
    assert _lnprior(csv, [9.0]) == pytest.approx(0.0)
    assert _lnprior(csv, [5.0]) == pytest.approx(-8.0)


def test_the_default_normal_is_unchanged():
    """Existing params_for_id files must behave exactly as before: centre of the
    range, std one sixth of it."""
    csv = "vessel_name,param_name,min,max,prior\na,k,0,6,normal\n"
    # centre 3, std 1 -> lnprior at 4 is -0.5
    assert _lnprior(csv, [4.0]) == pytest.approx(-0.5)


def test_the_default_exponential_rate_is_unchanged():
    csv = "vessel_name,param_name,min,max,prior\na,k,0,10,exponential\n"
    assert _lnprior(csv, [5.0]) == pytest.approx(-0.5)


def test_a_stated_rate_steepens_the_exponential():
    csv = ("vessel_name,param_name,min,max,prior,prior_lambda\n"
           "a,k,0,10,exponential,4.0\n")
    assert _lnprior(csv, [5.0]) == pytest.approx(-2.0)


def test_bounds_still_win_over_the_stated_values():
    csv = ("vessel_name,param_name,min,max,prior,prior_mean,prior_std\n"
           "a,k,0,10,normal,9.0,1.0\n")
    assert _lnprior(csv, [999.0]) == -np.inf


def test_a_config_without_the_new_key_keeps_its_behaviour():
    """param_id_info assembled by hand, or from before these columns existed."""
    from param_id.paramID import OpencorParamID

    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = {
        "param_prior_types": np.array(["normal"]),
        "param_mins": np.array([0.0]),
        "param_maxs": np.array([6.0]),
    }
    assert pid.get_lnprior_from_params([4.0]) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# A centre outside the range is refused
#
# Every prior is truncated to [min, max], so a mean outside it describes a peak
# the sampler can never reach: every draw sits on a tail and is pulled to the
# nearer bound. Legal arithmetic, silent, and almost never intended.
# ---------------------------------------------------------------------------
def test_a_mean_outside_the_range_is_rejected():
    with pytest.raises(ValueError, match="must lie within the parameter's range"):
        _info("vessel_name,param_name,min,max,prior,prior_mean\na,k,0,10,normal,20\n")


def test_a_mean_below_the_range_is_rejected():
    with pytest.raises(ValueError, match=r"\[0.0, 10.0\]"):
        _info("vessel_name,param_name,min,max,prior,prior_mean\na,k,0,10,normal,-5\n")


def test_the_offending_row_and_value_are_named():
    with pytest.raises(ValueError, match="row 1"):
        _info(
            "vessel_name,param_name,min,max,prior,prior_mean\n"
            "a,k,0,10,normal,5\n"
            "b,j,0,10,normal,99\n"
        )


@pytest.mark.parametrize("mean", [0, 5, 10])
def test_a_mean_on_or_inside_the_bounds_is_accepted(mean):
    """The bounds themselves are legal centres -- a prior peaked at an endpoint is
    a half-Gaussian, which is a reasonable thing to ask for."""
    info = _info(
        f"vessel_name,param_name,min,max,prior,prior_mean\na,k,0,10,normal,{mean}\n")
    assert info["param_prior_params"][0]["prior_mean"] == float(mean)


def test_a_negative_range_still_admits_a_negative_mean():
    """The check is against the row's own bounds, not against zero."""
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_mean\na,k,-10,-1,normal,-4\n")
    assert info["param_prior_params"][0]["prior_mean"] == -4.0


def test_the_std_is_not_bounds_checked():
    """Only values declared within_bounds are. A std larger than the range is a
    deliberately weak prior, not an error."""
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_std\na,k,0,10,normal,500\n")
    assert info["param_prior_params"][0]["prior_std"] == 500.0


def test_bounds_are_skipped_when_the_caller_cannot_supply_them():
    """normalise_prior_params is also called with a bare dict of fields (a
    downstream editor validating one row); it must still run every other check."""
    assert normalise_prior_params("normal", {"prior_mean": 999.0})["prior_mean"] == 999.0
    with pytest.raises(ValueError, match="greater than zero"):
        normalise_prior_params("normal", {"prior_std": -1.0})


def test_bounds_supplied_in_a_bare_dict_are_honoured():
    """So a downstream editor that does pass them gets the same verdict."""
    with pytest.raises(ValueError, match="must lie within"):
        normalise_prior_params("normal", {"prior_mean": 999.0, "min": 0, "max": 10})


# ---------------------------------------------------------------------------
# Unbounded parameters
#
# min/max are not only the prior's truncation: they are the optimiser's search
# box, the Sobol sampling range, the denominator of the parameter normalisation
# and the fallback FD step. An actually infinite range makes the normalisation
# NaN and every calibration with it, so "unbounded" means the range is derived
# from the prior rather than typed -- wide enough not to bind, and finite.
# ---------------------------------------------------------------------------
from parsers.PrimitiveParsers import (PARAM_UNBOUNDED_COLUMN, UNBOUNDED_SIGMA_SPAN,
                                      derive_bounds_from_prior, prior_supports_unbounded)


def test_the_range_is_derived_from_the_prior():
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_mean,prior_std,unbounded\n"
        "a,k,,,normal,7.0,1.5,true\n"
    )
    half = UNBOUNDED_SIGMA_SPAN * 1.5
    assert info["param_mins"][0] == pytest.approx(7.0 - half)
    assert info["param_maxs"][0] == pytest.approx(7.0 + half)


def test_the_derived_range_is_finite():
    """The whole reason it is derived rather than infinite: an infinite range
    makes the parameter normalisation NaN, and the GA with it."""
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_mean,prior_std,unbounded\n"
        "a,k,,,normal,7.0,1.5,true\n"
    )
    assert np.isfinite(info["param_mins"][0]) and np.isfinite(info["param_maxs"][0])


def test_an_unbounded_prior_is_not_truncated():
    """The derived range exists for the optimiser, not as a bound the user asked
    for, so the prior must not cut off at it."""
    from param_id.paramID import OpencorParamID

    info = _info(
        "vessel_name,param_name,min,max,prior,prior_mean,prior_std,unbounded\n"
        "a,k,,,normal,0.0,1.0,true\n"
    )
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    # Far outside the derived [-5, 5]: finite, and exactly the Gaussian's value.
    assert pid.get_lnprior_from_params([20.0]) == pytest.approx(-200.0)


def test_a_bounded_parameter_is_still_truncated():
    from param_id.paramID import OpencorParamID

    info = _info("vessel_name,param_name,min,max,prior\na,k,0,10,normal\n")
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    assert pid.get_lnprior_from_params([999.0]) == -np.inf


def test_unbounded_needs_a_prior_with_a_centre_and_a_width():
    """A uniform is *defined* by the range, and an exponential has a rate but no
    centre, so neither can stand in for one."""
    # Both priors that carry a location and a scale qualify; a uniform is *defined*
    # by its range and so cannot stand in for one.
    assert prior_supports_unbounded("normal") is True
    assert prior_supports_unbounded("exponential") is True
    assert prior_supports_unbounded("uniform") is False

    with pytest.raises(ValueError, match="no centre and width"):
        _info("vessel_name,param_name,min,max,prior,unbounded\na,k,,,uniform,true\n")


def test_unbounded_requires_the_centre_and_width_to_be_stated():
    """Their usual defaults come *from* the range, so leaving them out here would
    be circular."""
    with pytest.raises(ValueError, match="must be given"):
        _info(
            "vessel_name,param_name,min,max,prior,prior_mean,unbounded\n"
            "a,k,,,normal,7.0,true\n"
        )


def test_a_bounded_row_still_requires_min_and_max():
    with pytest.raises(ValueError, match="required unless"):
        _info("vessel_name,param_name,min,max,prior\na,k,,,normal\n")


@pytest.mark.parametrize("flag,expected_unbounded", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("", False),
])
def test_the_flag_is_read_the_ways_it_is_written(flag, expected_unbounded):
    csv = ("vessel_name,param_name,min,max,prior,prior_mean,prior_std,unbounded\n"
           f"a,k,0,10,normal,7.0,1.5,{flag}\n")
    info = _info(csv)
    assert bool(info["param_unbounded"][0]) is expected_unbounded


def test_an_unreadable_flag_is_an_error_not_a_quiet_false():
    """A cell the user filled in must not be ignored."""
    with pytest.raises(ValueError, match="must be true/false"):
        _info("vessel_name,param_name,min,max,prior,unbounded\na,k,0,10,normal,maybe\n")


def test_no_unbounded_column_leaves_everything_bounded():
    info = _info("vessel_name,param_name,min,max,prior\na,k,0,10,normal\n")
    assert not any(info["param_unbounded"])


def test_derive_bounds_is_span_times_the_scale():
    assert derive_bounds_from_prior("normal", {"prior_mean": 2.0, "prior_std": 0.5}) == (
        2.0 - 0.5 * UNBOUNDED_SIGMA_SPAN, 2.0 + 0.5 * UNBOUNDED_SIGMA_SPAN)


# ---------------------------------------------------------------------------
# Derived defaults are declared once
#
# The formulas used to be written twice: in get_lnprior_from_params, and in
# whatever prose a UI chose to describe them. `default_expr` states each once, CA
# computes from it, and a downstream editor shows the same number in the blank
# field's placeholder -- so the two cannot drift.
# ---------------------------------------------------------------------------
from parsers.PrimitiveParsers import eval_prior_default, prior_param_default


def test_the_normal_defaults_are_the_documented_ones():
    assert prior_param_default("normal", "prior_mean", {"min": 1.0, "max": 2.0}) == 1.5
    assert prior_param_default("normal", "prior_std", {"min": 0.0, "max": 6.0}) == 1.0


def test_a_derived_default_needs_the_bounds_it_names():
    """An unbounded row has no max, so nothing is invented from one."""
    assert prior_param_default("normal", "prior_mean", {"min": 1.0}) is None


def test_the_evaluator_does_arithmetic_and_nothing_else():
    """It reads CA's own schema, but an evaluator that can only add and divide
    cannot grow into something else later."""
    assert eval_prior_default("(min + max) / 2", {"min": 2.0, "max": 4.0}) == 3.0
    for hostile in ("__import__('os')", "open('/etc/passwd')", "[x for x in (1,2)]", "min.__class__"):
        assert eval_prior_default(hostile, {"min": 1.0, "max": 2.0}) is None


def test_a_division_by_zero_yields_no_default():
    assert eval_prior_default("max / prior_lambda", {"max": 1.0}, {"prior_lambda": 0.0}) is None


# ---------------------------------------------------------------------------
# The exponential's scale, and its unbounded form
# ---------------------------------------------------------------------------
def test_the_exponential_is_unchanged_when_nothing_is_stated():
    """origin 0 and scale max/lambda reproduce the original -lambda*x/max exactly."""
    from param_id.paramID import OpencorParamID

    info = _info("vessel_name,param_name,min,max,prior\na,k,0,10,exponential\n")
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    # -1.0 * 5 / 10
    assert pid.get_lnprior_from_params([5.0]) == pytest.approx(-0.5)


def test_a_stated_rate_still_steepens_it():
    from param_id.paramID import OpencorParamID

    info = _info(
        "vessel_name,param_name,min,max,prior,prior_lambda\na,k,0,10,exponential,4.0\n")
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    assert pid.get_lnprior_from_params([5.0]) == pytest.approx(-2.0)


def test_a_stated_scale_is_in_the_parameters_own_units():
    from param_id.paramID import OpencorParamID

    info = _info(
        "vessel_name,param_name,min,max,prior,prior_scale\na,k,0,10,exponential,2.0\n")
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    assert pid.get_lnprior_from_params([4.0]) == pytest.approx(-2.0)


def test_an_unbounded_exponential_derives_a_one_sided_range():
    """It decays away from its origin in one direction, so a range centred on the
    origin would put half the box where the prior has no mass."""
    info = _info(
        "vessel_name,param_name,min,max,prior,prior_origin,prior_scale,unbounded\n"
        "a,k,,,exponential,2.0,3.0,true\n"
    )
    assert info["param_mins"][0] == pytest.approx(2.0)
    assert info["param_maxs"][0] == pytest.approx(2.0 + UNBOUNDED_SIGMA_SPAN * 3.0)


def test_an_unbounded_exponential_needs_a_scale():
    """Its original rate is defined relative to max, and an unbounded parameter
    has no max, so the scale must be given in the parameter's own units."""
    with pytest.raises(ValueError, match="must be given"):
        _info(
            "vessel_name,param_name,min,max,prior,prior_origin,unbounded\n"
            "a,k,,,exponential,0.0,true\n"
        )


def test_an_unbounded_exponential_is_not_truncated():
    from param_id.paramID import OpencorParamID

    info = _info(
        "vessel_name,param_name,min,max,prior,prior_origin,prior_scale,unbounded\n"
        "a,k,,,exponential,0.0,1.0,true\n"
    )
    pid = OpencorParamID.__new__(OpencorParamID)
    pid.param_id_info = info
    # Far past the derived [0, 5]: finite, and the exponential's own value.
    assert pid.get_lnprior_from_params([40.0]) == pytest.approx(-40.0)
