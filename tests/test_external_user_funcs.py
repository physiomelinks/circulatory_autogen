"""Unit tests for loading user operation/cost funcs from external files (issue #303):
`operation_funcs_external_path` / `cost_funcs_external_path`. No model/MPI needed."""
import io
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


# The exact header CUFLynx has been writing into user output directories: two decorators
# imported from the bare module name, in a file that is *itself* called cost_funcs_user.py.
_CUFLYNX_WRITTEN_COST_FILE = textwrap.dedent('''
    from cost_funcs_user import is_MLE, cost_combiner
    from libcuflynx.param_id.differentiable import differentiable
    from libcuflynx.param_id.math_backend import make_math_backend
    mb = make_math_backend("numpy")

    @differentiable
    @is_MLE
    def gui_authored_cost(output, desired_mean, std, weight):
        return mb.sum(mb.abs((output - desired_mean) / std)) * weight

    @cost_combiner
    def gui_authored_combiner(costs):
        return mb.sum(costs)
''')


@pytest.mark.unit
def test_a_cost_file_cuflynx_already_wrote_still_loads(tmp_path):
    """The real downstream consumer, reproduced: CUFLynx writes user-authored funcs to
    ``<outputs>/user_funcs/{operation,cost,modifier}_funcs_user.py`` and points CA at them with
    the ``*_funcs_external_path`` config keys. Files it wrote before #433 begin with a bare
    ``from cost_funcs_user import is_MLE, cost_combiner``, which used to resolve because CA put
    the repo's ``funcs_user/`` on ``sys.path``. That ``sys.path`` surgery is gone; the import now
    resolves only because ``param_id/external_funcs.py::_install_legacy_module_aliases()``
    registers the bare names in ``sys.modules`` before exec'ing an external file.

    So that alias shim is **not** dead compatibility code: every such file already sitting in a
    user's output directory keeps that import forever, and deleting the shim breaks those runs
    with a ``ModuleNotFoundError`` from inside the user's own file. Note the file is itself named
    ``cost_funcs_user.py`` -- the alias must win over any same-named file, which is only true
    because its directory is not on ``sys.path``.
    """
    user_funcs_dir = tmp_path / "outputs" / "user_funcs"
    user_funcs_dir.mkdir(parents=True)
    cost_path = _write(user_funcs_dir, "cost_funcs_user.py", _CUFLYNX_WRITTEN_COST_FILE)

    meta = scriptFunctionParser(cost_funcs_external_path=cost_path).cost_func_metadata("numpy")

    # the user's funcs, with the flags their decorators set
    assert meta["gui_authored_cost"] == {"is_MLE": True, "is_combiner": False,
                                         "differentiable": True}
    assert meta["gui_authored_combiner"] == {"is_MLE": False, "is_combiner": True,
                                            "differentiable": False}
    # ...in the same registry as the built-ins the package ships, which the external file
    # must not displace
    assert meta["gaussian_MLE"] == {"is_MLE": True, "is_combiner": False, "differentiable": True}
    assert meta["additive"] == {"is_MLE": True, "is_combiner": True, "differentiable": True}
    # the imported decorators are not themselves registered as cost funcs
    assert "is_MLE" not in meta and "cost_combiner" not in meta


# ---------------------------------------------------------------------------
# where a *relative* path in user_inputs.yaml resolves from
# ---------------------------------------------------------------------------


def _parse(config):
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    return YamlFileParser().parse_user_inputs_file(dict(config), obs_path_needed=False)


_MINIMAL_CONFIG = {'file_prefix': '3compartment',
                   'input_param_file': '3compartment_parameters.csv'}


@pytest.mark.unit
@pytest.mark.parametrize('key', ['cost_funcs_external_path',
                                 'operation_funcs_external_path',
                                 'modifier_funcs_external_path'])
def test_a_relative_funcs_path_resolves_from_the_user_directory_not_the_cwd(
        key, tmp_path, monkeypatch):
    """funcs_user/README.md, CHANGELOG.md and user_inputs.yaml all show these relative.

    Nothing resolved them, so they reached ``os.path.abspath()`` inside
    ``external_funcs.register_funcs_from_file`` and came out relative to the *cwd*. The
    documented way to start a run is ``cd user_run_files && ./run_param_id.sh 4``, which
    makes the cwd ``user_run_files/`` -- so ``cost_funcs_external_path: funcs_user/my.py``,
    copied verbatim out of the README, looked for ``user_run_files/funcs_user/my.py``.
    """
    user_dir = tmp_path / 'my_study'
    (user_dir / 'funcs_user').mkdir(parents=True)
    (user_dir / 'resources').mkdir()
    written = user_dir / 'funcs_user' / 'my_funcs.py'
    written.write_text('', encoding='utf-8')

    monkeypatch.setenv('CUFLYNX_USER_DIR', str(user_dir))
    # ...and run from somewhere else entirely, as every documented launcher does.
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = dict(_MINIMAL_CONFIG)
    config[key] = 'funcs_user/my_funcs.py'
    parsed = _parse(config)

    assert parsed[key] == str(written)
    assert os.path.isfile(parsed[key])


@pytest.mark.unit
def test_an_absolute_funcs_path_is_left_alone(tmp_path, monkeypatch):
    absolute = _write(tmp_path, 'my_costs.py', _EXTERNAL_COSTS)
    user_dir = tmp_path / 'my_study'
    (user_dir / 'resources').mkdir(parents=True)
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(user_dir))

    config = dict(_MINIMAL_CONFIG)
    config['cost_funcs_external_path'] = absolute
    assert _parse(config)['cost_funcs_external_path'] == absolute


@pytest.mark.unit
def test_an_absent_funcs_path_stays_absent(tmp_path, monkeypatch):
    """Absent or empty is a documented no-op, and must not become the user directory."""
    user_dir = tmp_path / 'my_study'
    (user_dir / 'resources').mkdir(parents=True)
    monkeypatch.setenv('CUFLYNX_USER_DIR', str(user_dir))

    parsed = _parse(dict(_MINIMAL_CONFIG, cost_funcs_external_path=''))
    assert parsed['cost_funcs_external_path'] == ''
    assert 'operation_funcs_external_path' not in parsed


@pytest.mark.unit
def test_a_config_named_by_user_inputs_path_override_resolves_beside_itself(tmp_path):
    """With an override, the base is the overriding config's own directory.

    That is what ``resources_dir``/``generated_models_dir`` already do, and it is the only
    reading that lets a study directory outside the repository be self-contained.
    """
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    import yaml

    study = tmp_path / 'study'
    (study / 'funcs_user').mkdir(parents=True)
    (study / 'resources').mkdir()
    written = study / 'funcs_user' / 'my_funcs.py'
    written.write_text('', encoding='utf-8')

    override = study / 'user_inputs.yaml'
    override.write_text(yaml.dump(dict(_MINIMAL_CONFIG,
                                       cost_funcs_external_path='funcs_user/my_funcs.py',
                                       resources_dir='resources',
                                       generated_models_dir='generated_models')),
                        encoding='utf-8')

    parser = YamlFileParser()
    # parse_user_inputs_file(None) reads user_run_files/user_inputs.yaml first and follows
    # its user_inputs_path_override; stub that first read rather than editing the repo's.
    import libcuflynx.parsers.PrimitiveParsers as pp
    real_open = open

    def _fake_open(path, *args, **kwargs):
        if str(path).endswith('user_run_files/user_inputs.yaml'):
            return io.StringIO(yaml.dump({'user_inputs_path_override': str(override)}))
        return real_open(path, *args, **kwargs)

    pp.open = _fake_open
    try:
        parsed = parser.parse_user_inputs_file(None, obs_path_needed=False)
    finally:
        del pp.open

    assert parsed['cost_funcs_external_path'] == str(written)
