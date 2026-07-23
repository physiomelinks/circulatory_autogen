"""Unit tests for loading user operation/cost funcs from external files (issue #303):
`operation_funcs_external_path` / `cost_funcs_external_path`. No model/MPI needed."""
import os
import textwrap

import pytest

from parsers.PrimitiveParsers import scriptFunctionParser
from param_id.external_funcs import register_funcs_from_file
from param_id.math_backend import make_math_backend


_EXTERNAL_OPS = textwrap.dedent('''
    from param_id.operation_funcs import series_to_constant
    from param_id.differentiable import differentiable
    from param_id.math_backend import make_math_backend
    mb = make_math_backend("numpy")

    @differentiable
    @series_to_constant
    def my_external_op(x, series_output=False):
        if series_output:
            return x
        return mb.mean(x) * 2.0

    def _helper_should_be_skipped(x):   # leading underscore -> not registered
        return x
''')

_EXTERNAL_COSTS = textwrap.dedent('''
    from cost_funcs_user import is_MLE
    from param_id.differentiable import differentiable
    from param_id.math_backend import make_math_backend
    mb = make_math_backend("numpy")

    @differentiable
    @is_MLE
    def my_external_cost(output, desired_mean, std, weight):
        return mb.sum(mb.abs((output - desired_mean) / std)) * weight
''')


def _write(tmp_path, name, content):
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as f:
        f.write(content)
    return p


@pytest.mark.unit
def test_scriptfunctionparser_merges_external_operation_and_cost_funcs(tmp_path):
    op_path = _write(tmp_path, "my_ops.py", _EXTERNAL_OPS)
    cost_path = _write(tmp_path, "my_costs.py", _EXTERNAL_COSTS)

    sfp = scriptFunctionParser(operation_funcs_external_path=op_path,
                               cost_funcs_external_path=cost_path)

    ops = sfp.get_operation_funcs_dict("numpy")
    # external op registered, alongside the built-ins; the private helper is skipped.
    assert "my_external_op" in ops
    assert {"max", "min", "mean"} <= set(ops)          # core built-ins still present
    assert "steady_state_avg" in ops                    # funcs_user built-in still present
    assert "_helper_should_be_skipped" not in ops
    assert getattr(ops["my_external_op"], "series_to_constant", False) is True

    costs = sfp.get_cost_funcs_dict("numpy")
    assert "my_external_cost" in costs and "gaussian_MLE" in costs

    meta = sfp.cost_func_metadata("numpy")
    assert "my_external_cost" in meta
    assert meta["my_external_cost"]["is_MLE"] is True
    assert meta["my_external_cost"]["differentiable"] is True
    assert meta["my_external_cost"]["is_combiner"] is False


@pytest.mark.unit
def test_no_external_paths_is_a_noop():
    sfp = scriptFunctionParser()   # no external paths
    ops = sfp.get_operation_funcs_dict("numpy")
    costs = sfp.get_cost_funcs_dict("numpy")
    assert "gaussian_MLE" in costs and {"max", "min", "mean"} <= set(ops)
    assert "my_external_op" not in ops and "my_external_cost" not in costs


@pytest.mark.unit
def test_missing_external_path_raises(tmp_path):
    sfp = scriptFunctionParser(
        operation_funcs_external_path=os.path.join(str(tmp_path), "does_not_exist.py"))
    with pytest.raises(FileNotFoundError):
        sfp.get_operation_funcs_dict("numpy")


@pytest.mark.unit
def test_init_from_dict_forwards_external_paths():
    """CVS0DParamID.init_from_dict must forward the two config keys to the engine (they are in its
    consumed arg_options), so a config-driven run picks up the external funcs."""
    import inspect
    from param_id.paramID import CVS0DParamID
    src = inspect.getsource(CVS0DParamID.init_from_dict)
    assert "operation_funcs_external_path" in src and "cost_funcs_external_path" in src
    params = inspect.signature(CVS0DParamID.__init__).parameters
    assert "operation_funcs_external_path" in params and "cost_funcs_external_path" in params


@pytest.mark.unit
def test_register_funcs_from_file_binds_backend_and_filters(tmp_path):
    op_path = _write(tmp_path, "ops2.py", _EXTERNAL_OPS)
    registry = {}
    register_funcs_from_file(op_path, registry, make_math_backend("numpy"))
    assert "my_external_op" in registry and "_helper_should_be_skipped" not in registry
    # a falsy path is a no-op (does not raise, leaves the registry untouched)
    register_funcs_from_file(None, registry, make_math_backend("numpy"))
    assert list(registry) == ["my_external_op"]
