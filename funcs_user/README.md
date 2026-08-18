# `funcs_user/` — your own cost, operation and modifier functions

**This directory is yours.** Nothing in it is touched when you upgrade libcuflynx, so it is
where your own code belongs.

Three kinds of function can be added here:

| What you want | Where it is used |
|---|---|
| a **cost** function | `cost_type` on a data item in `obs_data.json` |
| an **operation** | `operation` on a data item — how a trace becomes one number |
| a **modifier** | `modifier` on an entry in `params_for_id` |

## Adding one

Copy the matching template in this directory, rename it, and write your function in it:

| Template | Config key |
|---|---|
| `cost_funcs_example.py` | `cost_funcs_external_path` |
| `operation_funcs_example.py` | `operation_funcs_external_path` |
| `modifier_funcs_example.py` | `modifier_funcs_external_path` |

Then name your file in `user_run_files/user_inputs.yaml`:

```yaml
cost_funcs_external_path:      funcs_user/my_costs.py
operation_funcs_external_path: funcs_user/my_ops.py
modifier_funcs_external_path:  funcs_user/my_modifiers.py
```

Each key is optional, and an empty one changes nothing. Every top-level function in the file
you name is registered, alongside the ones libcuflynx ships — so yours appear in the same
menus and are used the same way. Give a function the same name as a shipped one and yours
replaces it, which is occasionally what you want and otherwise a surprise, so prefer a name
of your own.

**Where relative paths point.** From a checkout, at the repository root. If you set
`user_inputs_path_override`, at the directory holding that config file — so a study kept
outside the repository stays self-contained. From a `pip install libcuflynx` with no
checkout, set `CUFLYNX_USER_DIR` to your working directory; otherwise the directory you run
from is used. An absolute path is always taken as-is.

## The functions libcuflynx ships

These now live inside the installed package rather than in this directory, so that upgrading
libcuflynx keeps them current. **Do not edit them in place** — an upgrade replaces the whole
package, and your edits go with it. Write your own file here instead, as above; if you want to
start from a shipped function, copy it into your file and rename it.

If you previously added your own functions by editing `cost_funcs_user.py`,
`operation_funcs_user.py` or `modifier_funcs_user.py` in this directory, move them into a file
of your own and set the matching `*_external_path` key. Until you do, they are not loaded.

## Also here: example solver classes

`example_model_scipy/`, `example_model_external/` and `heat_fenics/` are a different feature —
worked examples of `model_type: external_python`, where you supply a whole solver rather than a
function. See `tutorial/docs/external-python-solvers.md`.
