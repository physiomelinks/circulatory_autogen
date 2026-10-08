# Writing up a model and its calibration

Two commands turn a model and its calibration into the methods section of a paper:

1. `cuflynx-variable-mapping` lists every variable and parameter of the model with the LaTeX symbol it is written as;
2. `cuflynx-methods-latex` writes a methods section in LaTeX. It has three parts:
    - the model's equations, one subsection per module;
    - a flow chart of the calibration;
    - what was done in each step of the flow chart.

## The variable mapping

```
cuflynx-variable-mapping generated_models/<prefix>/<prefix>_flat.cellml
cuflynx-variable-mapping --workflow <instance dir>/calibration_workflow.json --module-library-dir <library>
```

Each command writes `<prefix>_variable_mapping.csv` beside the model. For a calibration workflow, the file goes in the target supermodule instance's directory instead, as `<instance>_variable_mapping.csv`.

| Column | What it holds |
|---|---|
| `variable_name` | `component/variable`, with `_module` dropped from the component (`mod_C/z`, `parameters/p_mod_A`). These are the names obs_data operands and CUFLynx use. |
| `latex` | The symbol, without `$`. **Edit these.** |
| `kind` | `state`, `parameter`, `constant`, `algebraic` or `time`. |
| `component`, `units` | Where the variable is defined, and its units. |

The default symbols follow a simple rule, meant to be edited:

- **Subscripts:** a name is split on `_`. The first part is the symbol and the rest are its subscripts: `g_leak_Na` → $g_{\mathrm{leak},\mathrm{Na}}$.
- **Greek letters:** the name of a Greek letter becomes that letter: `rho_M` → $\rho_{M}$, `Delta_C` → $\Delta_{C}$.
- **Upright text:** one-character parts stay italic, and longer ones are set upright: `Nai` → $\mathrm{Nai}$.
- **Module subscript:** the module a variable belongs to is added as a last subscript. A generated parameter `<var>_<vessel>` is split into its variable and its vessel. In a model generated alone (vessel `mod`), the `mod_` prefix is dropped: `p_mod_A` → $p_{A}$.

Running the command again keeps every symbol already in the file. It adds new variables with the default rule and drops variables the model no longer has. The command also warns when two variables share a symbol, because they could not be told apart in the equations. CUFLynx edits the same file from its Variables panel.

## The methods section

```
# a calibration workflow (see Parameter Identification -> Calibration workflows)
cuflynx-methods-latex --workflow <instance dir>/calibration_workflow.json \
    --module-library-dir <library> --run-dir workflow_output/<name> -o methods.tex --pdf

# one calibration
cuflynx-methods-latex --model <prefix>_flat.cellml --obs-data <obs>.json \
    --params-for-id <params>.csv --settings user_run_files/user_inputs.yaml \
    --run-dir param_id_output/<method>_<prefix>_<obs> -o methods.tex
```

The document has three parts.

1. **Model.** Each module gets a subsection with:
    - its ODEs and algebraic equations, as written by Myokit's LaTeX writer in the mapping's symbols, with conditionals set as `cases`;
    - its initial values;
    - a table of the parameters it uses (symbol, value, units, name in the model).

    For a workflow, the model is the target supermodule, and each module is labelled with its module type, version and instance from the supermodule config.
2. **Flow chart.** This is a TikZ figure.
    - **For a workflow:** one box per step (its instance, how many parameters and observables, the method). Arrows run from each step to the steps after it: solid for values passed on as fixed values, dashed for priors. A last box shows the merge into the target.
    - **For one calibration:** the model, observables and parameters feed the optimisation, followed by MCMC and identifiability when they were run.
3. **One subsection per step.** Each covers:
    - which instance the step calibrates and against which observations;
    - the values it was given by earlier steps, and the priors it was given;
    - the protocol (experiments, pre-times, segment durations, inputs set per segment);
    - the observables (feature, observed value ± standard deviation, weight, experiment and segment);
    - the calibrated parameters with their ranges and priors;
    - the method, from CA's own description of it, with its settings, the solver, and the cost function as a formula;
    - MCMC settings when the step stored its posterior;
    - with `--run-dir`, the calibrated values and the best cost. A step that used priors reports its posterior medians.

`--body-only` writes the sections without a preamble, to `\input` into a paper. The packages it needs are listed in its first line. `--flowchart-only` writes just the figure.

The full document needs `amsmath`, `amssymb`, `booktabs`, `graphicx` and `tikz`. Compile it twice, or with `latexmk` (`--pdf` does this), so the figure reference resolves.

From Python:

```python
from libcuflynx.reporting import workflow_methods_latex, calibration_methods_latex, update_variable_mapping_file
update_variable_mapping_file(model_path, mapping_path)
text = workflow_methods_latex(workflow_path, run_dir=run_dir, mapping=mapping_path)
```

!!! note
    The output is a starting point for the paper's methods, not the paper. Check the symbols, and reword the text to suit. A very long single equation can run past the margin: LaTeX cannot break an `align` line by itself.
