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
    assert set(PARAM_PRIOR_PARAM_NAMES) == {"prior_lambda", "prior_mean", "prior_std"}


def test_each_prior_declares_the_values_it_takes():
    assert [s["name"] for s in PARAM_PRIOR_TYPES["exponential"]["params"]] == ["prior_lambda"]
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
