# `funcs_user/` — your own cost, operation and modifier functions

**This directory is yours.** Nothing in it is overwritten when you upgrade libcuflynx.

The cost, operation and modifier functions that libcuflynx *ships* used to live here, in
`cost_funcs_user.py` / `operation_funcs_user.py` / `modifier_funcs_user.py`. They are library
code, so as of issue #433 they live inside the package, at
`src/libcuflynx/funcs/{cost,operation,modifier}_funcs_user.py`. Do not edit them: a `pip install
--upgrade libcuflynx` replaces the whole package.

To add your own function, write it in your own file here and name that file in
`user_run_files/user_inputs.yaml`:

```yaml
cost_funcs_external_path:     funcs_user/my_costs.py
operation_funcs_external_path: funcs_user/my_ops.py
modifier_funcs_external_path:  funcs_user/my_modifiers.py
```

Each key is optional; absent or empty is a no-op. The file's top-level functions are merged
into the *same* registry as the shipped ones — same decorators, same discoverability
(`cost_func_metadata()`, `get_operation_funcs_dict()`, `param_modifiers()`) — and a function
that reuses a shipped name deliberately overrides it.

Start from the templates in this directory:

| Template | Config key | Registers |
|---|---|---|
| `cost_funcs_example.py` | `cost_funcs_external_path` | `cost_type` values for obs_data.json |
| `operation_funcs_example.py` | `operation_funcs_external_path` | `operation` values for obs_data data items |
| `modifier_funcs_example.py` | `modifier_funcs_external_path` | `modifier` values for params_for_id entries |

Copy one, rename it, point the config key at it. Relative paths are resolved from the
repository root — or, if you set `user_inputs_path_override`, from the directory of the
config file it names, so a study directory outside the repository is self-contained. From a
`pip install libcuflynx` with no checkout there is no repository root: set `CUFLYNX_USER_DIR`
to your working directory, or the process's current directory is used. An absolute path is
always taken as-is.

## Upgrading from a version before this change

If you added a cost/operation/modifier function by editing `funcs_user/cost_funcs_user.py`
(or the operation/modifier equivalents) **in place**, move those functions into a file of your
own and set the matching `*_external_path` key. Otherwise they are simply not loaded — the
registry now reads the package copy. See `CHANGELOG.md` at the repository root.

## Also in this directory (a different feature)

`example_model_scipy/`, `example_model_external/` and `heat_fenics/` are worked examples of
`model_type: external_python` — a user-supplied *solver class*, not a funcs file. See
`tutorial/docs/external-python-solvers.md`.
