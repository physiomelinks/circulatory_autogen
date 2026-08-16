"""Unit tests for loading user operation/cost funcs from external files (issue #303):
`operation_funcs_external_path` / `cost_funcs_external_path`. No model/MPI needed."""
import os
import textwrap

import pytest

from libcuflynx.parsers.PrimitiveParsers import scriptFunctionParser
from libcuflynx.param_id.external_funcs import register_funcs_from_file
from libcuflynx.param_id.math_backend import make_math_backend


_EXTERNAL_OPS = textwrap.dedent('''
    from libcuflynx.param_id.operation_funcs import series_to_constant
    from libcuflynx.param_id.differentiable import differentiable
    from libcuflynx.param_id.math_backend import make_math_backend
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
    from libcuflynx.param_id.differentiable import differentiable
    from libcuflynx.param_id.math_backend import make_math_backend
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
    from libcuflynx.param_id.paramID import CVS0DParamID
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


# --------------------------------------------------------------------------------------------
# Issue #433: the built-ins moved from the repo's funcs_user/ into the package, and are no
# longer imported by bare module name. The merged registry (package built-ins + external file)
# is the contract that had to survive the move.

_EXTERNAL_COSTS_PACKAGE_IMPORT = textwrap.dedent('''
    from libcuflynx.funcs.cost_funcs_user import is_MLE, cost_combiner
    from libcuflynx.param_id.differentiable import differentiable
    from libcuflynx.param_id.math_backend import make_math_backend
    mb = make_math_backend("numpy")

    @differentiable
    @is_MLE
    def pkg_external_cost(output, desired_mean, std, weight):
        return mb.sum(mb.abs((output - desired_mean) / std)) * weight

    @cost_combiner
    def pkg_external_combiner(costs):
        return max(costs)
''')

_EXTERNAL_MODIFIERS = textwrap.dedent('''
    from libcuflynx.funcs.modifier_funcs_user import modifier_func

    @modifier_func(inputs={'reference': 'float'}, description='target = theta + reference')
    def pkg_offset_from(theta, baseline, reference):
        return theta + reference
''')


@pytest.mark.unit
def test_builtin_funcs_are_importable_from_the_package():
    """They are library code now, so they import as ``libcuflynx.funcs.*`` -- not by the bare
    module name that only ever resolved from a source checkout with funcs_user/ on sys.path."""
    from libcuflynx.funcs import cost_funcs_user, operation_funcs_user, modifier_funcs_user

    assert cost_funcs_user.__name__ == 'libcuflynx.funcs.cost_funcs_user'
    assert 'gaussian_MLE' in cost_funcs_user.get_cost_funcs_dict_for_mode('numpy')
    assert callable(operation_funcs_user.steady_state_avg)
    assert callable(modifier_funcs_user.modifier_func)


@pytest.mark.unit
def test_no_bare_name_imports_of_the_moved_modules_remain_in_the_package():
    """The whole point of #433: nothing in the shipped package may reach for
    ``import cost_funcs_user`` / ``operation_funcs_user`` / ``modifier_funcs_user``, because an
    installed package has no funcs_user/ directory to find them in."""
    import re
    import libcuflynx

    pkg_dir = os.path.dirname(os.path.abspath(libcuflynx.__file__))
    bare = re.compile(
        r"^\s*(?:import\s+(?:cost|operation|modifier|protocol)_funcs_user"
        r"|from\s+(?:cost|operation|modifier|protocol)_funcs_user\s+import"
        r"|importlib\.import_module\(\s*['\"](?:cost|operation|modifier|protocol)_funcs_user['\"])",
        re.M)
    offenders = []
    for root, _dirs, files in os.walk(pkg_dir):
        for name in files:
            if not name.endswith('.py'):
                continue
            path = os.path.join(root, name)
            with open(path, encoding='utf-8') as f:
                if bare.search(f.read()):
                    offenders.append(path)
    assert offenders == []


@pytest.mark.unit
def test_merged_cost_registry_is_package_builtins_plus_external_with_flags_intact(tmp_path):
    """One registry, two sources: the costs the package ships and the costs in the file named by
    ``cost_funcs_external_path``, each keeping its is_MLE / is_combiner / differentiable flags."""
    cost_path = _write(tmp_path, "pkg_costs.py", _EXTERNAL_COSTS_PACKAGE_IMPORT)
    meta = scriptFunctionParser(cost_funcs_external_path=cost_path).cost_func_metadata("numpy")

    # built-ins, straight from libcuflynx.funcs.cost_funcs_user
    assert meta["gaussian_MLE"] == {"is_MLE": True, "is_combiner": False, "differentiable": True}
    assert meta["additive"] == {"is_MLE": True, "is_combiner": True, "differentiable": True}
    # external file, in the same registry
    assert meta["pkg_external_cost"] == {"is_MLE": True, "is_combiner": False,
                                         "differentiable": True}
    assert meta["pkg_external_combiner"] == {"is_MLE": False, "is_combiner": True,
                                             "differentiable": False}
    # the decorators the external file imported are not themselves registered as costs
    assert "is_MLE" not in meta and "cost_combiner" not in meta


@pytest.mark.unit
def test_merged_operation_registry_is_package_builtins_plus_external(tmp_path):
    op_path = _write(tmp_path, "pkg_ops.py", _EXTERNAL_OPS)
    ops = scriptFunctionParser(operation_funcs_external_path=op_path).get_operation_funcs_dict()

    assert {"max", "min", "mean"} <= set(ops)          # param_id.operation_funcs core ops
    assert "steady_state_avg" in ops                    # libcuflynx.funcs.operation_funcs_user
    assert "my_external_op" in ops                      # the external file


@pytest.mark.unit
def test_merged_modifier_registry_is_builtins_plus_external(tmp_path):
    """``param_modifiers`` is the same story for modifiers (#383): built-ins from the package,
    user funcs from ``modifier_funcs_external_path``, with the declared inputs intact."""
    from libcuflynx.parsers.PrimitiveParsers import param_modifiers

    mod_path = _write(tmp_path, "pkg_modifiers.py", _EXTERNAL_MODIFIERS)
    records = param_modifiers(mod_path)

    assert {"scale", "remainder"} <= set(records)
    assert records["scale"]["user_defined"] is False
    assert records["pkg_offset_from"]["user_defined"] is True
    assert records["pkg_offset_from"]["inputs"] == {"reference": "float"}


@pytest.mark.unit
def test_pre_433_external_files_still_import_the_decorators_by_bare_name(tmp_path):
    """External funcs files are *user* code CA does not get to rewrite, and every documented
    example told users to write ``from cost_funcs_user import is_MLE``. The bare names survive
    as aliases of the package modules, installed when an external file is loaded."""
    cost_path = _write(tmp_path, "legacy_costs.py", _EXTERNAL_COSTS)
    meta = scriptFunctionParser(cost_funcs_external_path=cost_path).cost_func_metadata("numpy")
    assert meta["my_external_cost"]["is_MLE"] is True

    import sys
    from libcuflynx.funcs import cost_funcs_user
    assert sys.modules["cost_funcs_user"] is cost_funcs_user
