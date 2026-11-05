"""An unsupported model_type must say so, not crash on a missing attribute later.

CUFLynx #270 reported `AttributeError: 'CVS0DParamID' object has no attribute 'param_id'`
from inside EmulatorTrainer.init_from_dict. The attribute was never the problem: __init__
only builds `self.param_id` for the model types parameter identification supports, and
`set_output_dir` -- called a few lines later, unconditionally -- then dereferenced it. So
any unsupported model_type surfaced several frames from the config key that caused it, in
terms that name an internal attribute rather than the setting the user chose.

`cpp` is the case that reaches this by being *valid*: a real model_type for generation that
neither param_id nor solver_wrappers can run.
"""
import pytest

from libcuflynx.param_id.paramID import CVS0DParamID, PARAM_ID_MODEL_TYPES


@pytest.mark.unit
def test_cpp_is_valid_to_generate_but_not_to_calibrate():
    """The list is a capability statement, so pin what it excludes and why."""
    from libcuflynx.parsers.PrimitiveParsers import SOLVER_SCHEMA

    assert 'cpp' in SOLVER_SCHEMA['solvers_by_model_type'], 'cpp is a real model type'
    assert 'cpp' not in PARAM_ID_MODEL_TYPES, 'and param-id cannot run one'


@pytest.mark.unit
@pytest.mark.parametrize('model_type', ['cpp', 'not_a_model_type', None])
def test_an_unsupported_model_type_names_itself(model_type):
    with pytest.raises(ValueError) as excinfo:
        CVS0DParamID(model_path='/nonexistent/model.cellml', model_type=model_type,
                     param_id_method='genetic_algorithm')

    message = str(excinfo.value)
    assert 'parameter identification' in message
    assert repr(model_type) in message, 'the message must name what was actually passed'
    for supported in PARAM_ID_MODEL_TYPES:
        assert supported in message, 'and say what it could have been'
    # The symptom this replaces.
    assert 'has no attribute' not in message
