"""The ``cost_kwargs`` contract for user-defined cost functions (issue #84).

Cost funcs used to be called with a fixed positional ``(output, ground_truth, std, weight)`` at
every call site. That forced every cost func to accept a std and a weight whether or not the cost
has any use for them, and gave a user cost no way at all to receive its own parameters. These
tests pin the replacement: what a cost func is offered follows from its signature, and a
data_item may add its own keyword arguments.
"""
import pytest

from libcuflynx.param_id.cost_kwargs import (
    RESERVED_COST_KWARGS, call_cost_func, check_cost_kwargs, framework_kwargs_for,
    get_cost_kwarg_spec, resolve_cost_kwargs, validate_cost_kwargs)


def _full(output, desired_mean, std, weight):
    return (output - desired_mean) ** 2 / (std ** 2) * weight


def _no_std(output, prob_dist_params, weight):
    """Mirrors multimodal_gaussian: a cost with no notion of a standard deviation."""
    return (output - prob_dist_params) * weight


def _with_user_kwarg(output, desired_mean, std, weight, exponent=2):
    return abs(output - desired_mean) ** exponent / std * weight


def _accepts_any(*args, **kwargs):
    return (args, kwargs)


@pytest.mark.unit
def test_a_cost_func_is_only_offered_what_its_signature_can_hold():
    """The point of the contract: a cost with no std must not be handed one.

    Previously every call site passed four positional arguments, so a cost that does not use a
    standard deviation still had to declare the parameter and drop it on the floor.
    """
    assert framework_kwargs_for(_full, std=2.0, weight=3.0) == {'std': 2.0, 'weight': 3.0}
    assert framework_kwargs_for(_no_std, std=2.0, weight=3.0) == {'weight': 3.0}
    # **kwargs takes everything on offer
    assert framework_kwargs_for(_accepts_any, std=2.0, weight=3.0) == {'std': 2.0, 'weight': 3.0}


@pytest.mark.unit
def test_call_cost_func_dispatches_by_signature():
    assert call_cost_func(_full, 3.0, 1.0, std=2.0, weight=1.0) == pytest.approx(1.0)
    # no std parameter -> not passed, and no TypeError
    assert call_cost_func(_no_std, 3.0, 1.0, std=2.0, weight=2.0) == pytest.approx(4.0)


@pytest.mark.unit
def test_a_user_kwarg_reaches_the_cost_func():
    """The other half of #84: a cost's own parameters, supplied per data_item."""
    base = call_cost_func(_with_user_kwarg, 4.0, 1.0, std=1.0, weight=1.0)
    cubed = call_cost_func(_with_user_kwarg, 4.0, 1.0, std=1.0, weight=1.0,
                           cost_kwargs={'exponent': 3})
    assert base == pytest.approx(9.0)
    assert cubed == pytest.approx(27.0)


@pytest.mark.unit
def test_an_unknown_cost_kwarg_is_an_error_not_a_silent_no_op():
    """A stale or misspelled key would otherwise change nothing at all, and a calibration against
    the wrong cost looks exactly like one against the right cost."""
    with pytest.raises(ValueError, match="has no parameter 'expnent'"):
        check_cost_kwargs({'expnent': 3}, _with_user_kwarg, 'custom', 'my_item')


@pytest.mark.unit
@pytest.mark.parametrize('reserved', sorted(RESERVED_COST_KWARGS))
def test_cost_kwargs_may_not_shadow_what_the_framework_supplies(reserved):
    """std and weight come from the data_item's own fields; letting cost_kwargs set them would
    either silently calibrate against the wrong value or raise a bare duplicate-argument error."""
    with pytest.raises(ValueError, match='supplied by circulatory_autogen'):
        check_cost_kwargs({reserved: 1.0}, _full, 'gaussian_MLE', 'my_item')


@pytest.mark.unit
def test_a_positional_argument_cannot_be_given_as_a_cost_kwarg():
    with pytest.raises(ValueError, match='filled positionally'):
        check_cost_kwargs({'output': 1.0}, _full, 'gaussian_MLE', 'my_item')


@pytest.mark.unit
def test_a_non_string_key_is_rejected():
    with pytest.raises(ValueError, match='keys must be strings'):
        check_cost_kwargs({3: 1.0}, _full, 'gaussian_MLE', 'my_item')


@pytest.mark.unit
def test_a_func_accepting_kwargs_takes_any_key():
    """MSE is *args/**kwargs, so validation must not invent a restriction it does not have."""
    check_cost_kwargs({'anything': 1}, _accepts_any, 'MSE', 'my_item')


@pytest.mark.unit
def test_json_numbers_are_coerced_towards_the_default_type():
    """JSON has one number type, so a GUI writes 3.0 where the default is the int 2."""
    assert resolve_cost_kwargs({'exponent': 3.0}, _with_user_kwarg) == {'exponent': 3}
    assert isinstance(resolve_cost_kwargs({'exponent': 3.0}, _with_user_kwarg)['exponent'], int)


@pytest.mark.unit
def test_spec_separates_positional_from_keyword():
    accepted, positional, accepts_any = get_cost_kwarg_spec(_with_user_kwarg)
    assert not accepts_any
    # std/weight are framework-supplied, so they are not counted as positional fill
    assert positional == ['output', 'desired_mean']
    assert 'exponent' in accepted


@pytest.mark.unit
def test_validate_cost_kwargs_checks_every_data_item_up_front():
    """A bad key should fail at setup, not after the first expensive forward solve."""
    obs_info = {'cost_kwargs': [{}, {'expnent': 3}],
                'data_item_names': ['ok_item', 'bad_item']}
    with pytest.raises(ValueError, match='bad_item'):
        validate_cost_kwargs(obs_info, {'custom': _with_user_kwarg}, ['custom', 'custom'])


@pytest.mark.unit
def test_validate_cost_kwargs_is_a_noop_when_nothing_declares_any():
    validate_cost_kwargs({'cost_kwargs': [{}, {}]}, {'custom': _with_user_kwarg},
                         ['custom', 'custom'])
    validate_cost_kwargs({}, {}, [])


@pytest.mark.unit
def test_the_builtin_cost_funcs_still_receive_std_and_weight():
    """Regression guard: the shipped funcs must keep getting what they always got."""
    from libcuflynx.funcs.cost_funcs_user import get_cost_funcs_dict_for_mode

    funcs = get_cost_funcs_dict_for_mode('numpy')
    offered = framework_kwargs_for(funcs['gaussian_MLE'], std=1.0, weight=1.0)
    assert offered == {'std': 1.0, 'weight': 1.0}
