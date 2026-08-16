# Changelog

Notable changes to libcuflynx (circulatory_autogen). Entries under *Unreleased* ship with the
next release; add to that section as you land a change.

## Unreleased

### Breaking — action required if you edited `funcs_user/*_funcs_user.py`

**The built-in cost, operation and modifier functions moved into the package** (issue #433).

| Was | Is now |
|---|---|
| `funcs_user/cost_funcs_user.py` | `libcuflynx.funcs.cost_funcs_user` |
| `funcs_user/operation_funcs_user.py` | `libcuflynx.funcs.operation_funcs_user` |
| `funcs_user/modifier_funcs_user.py` | `libcuflynx.funcs.modifier_funcs_user` |

They were library code living outside the package and imported by bare module name
(`import cost_funcs_user`), which only resolved because CA appended the repo's `funcs_user/`
directory to `sys.path`. That cannot work from an installed package, and a library must not
ask its users to edit files inside it.

**If you added your own cost / operation / modifier function by editing one of those files in
place, it will silently stop being registered** — the registry now reads the package copy, and
your edited file is no longer loaded at all. Move your functions into a file of your own and
name it in `user_run_files/user_inputs.yaml`:

```yaml
cost_funcs_external_path:      funcs_user/my_costs.py
operation_funcs_external_path: funcs_user/my_ops.py
modifier_funcs_external_path:  funcs_user/my_modifiers.py
```

Nothing else about your functions has to change: the external file's top-level functions are
merged into the *same* registry as the built-ins, with `@is_MLE` / `@cost_combiner` /
`@differentiable` / `@series_to_constant` / `@modifier_func` intact and reported by
`cost_func_metadata()` and `param_modifiers()`; a function reusing a built-in name overrides
it. This mechanism is not new (issues #303, #383) — it is now the only supported way to add
your own.

Templates to copy: `funcs_user/cost_funcs_example.py`, `funcs_user/operation_funcs_example.py`,
`funcs_user/modifier_funcs_example.py`. See `funcs_user/README.md`.

Imports of the *shipped* functions change accordingly — `from funcs_user.cost_funcs_user import
cost_func_metadata` becomes `from libcuflynx.funcs.cost_funcs_user import cost_func_metadata`.
For external funcs files, the old bare-name imports (`from cost_funcs_user import is_MLE`)
still work: CA aliases the three legacy module names onto the package modules when it loads an
external file. New files should import `from libcuflynx.funcs.cost_funcs_user import is_MLE`.

`funcs_user/` itself stays, and is now purely yours: the three templates above, plus the
unrelated `model_type: external_python` examples (`example_model_scipy/`,
`example_model_external/`, `heat_fenics/`).
