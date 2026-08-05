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
