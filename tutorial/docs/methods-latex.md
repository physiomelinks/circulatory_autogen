# Writing up a model and its calibration

Two commands turn a model and its calibration into the methods section of a paper:

1. `cuflynx-variable-mapping` lists every variable and parameter of the model with the LaTeX symbol it is written as;
2. `cuflynx-methods-latex` writes a methods section in LaTeX. It has four parts:
    - the model's equations, one subsection per module;
    - a table of every parameter with its value and source, citing your references;
    - a flow chart of the calibration;
    - what was done in each step of the flow chart.

To see one without setting anything up, run the workflow example in
[Calibration workflows → Try it](parameter-identification.md#try-it).

## The variable mapping

```
cuflynx-variable-mapping generated_models/<prefix>/<prefix>_flat.cellml
cuflynx-variable-mapping --workflow workflow.json
```

Each command writes `<prefix>_variable_mapping.csv` beside the model. For a calibration workflow, the file goes beside the target's parameters file instead, as `<name>_variable_mapping.csv` (`<name>` is the target module array's prefix, or its parameterisation).

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
cuflynx-methods-latex --workflow workflow.json --run-dir workflow_output/<name> -o methods.tex --pdf

# one calibration
cuflynx-methods-latex --model <prefix>_flat.cellml --obs-data <obs>.json \
    --params-for-id <params>.csv --settings user_run_files/user_inputs.yaml \
    --run-dir param_id_output/<method>_<prefix>_<obs> -o methods.tex
```

The document has four parts.

1. **Model.** Each module gets a subsection with:
    - its ODEs and algebraic equations, as written by Myokit's LaTeX writer in the mapping's symbols, with conditionals set as `cases`;
    - its initial values;
    - a table of the parameters it uses (symbol, value, units, name in the model).

    For a workflow, the model is the target, and each module is labelled with its module type, version and parameterisation, from the target's module array or supermodule config.
2. **Parameters and sources.** One table of every parameter of the model: symbol, value, units and source (see [Citations](#citations)). A parameter that was calibrated names the step that calibrated it, with a link to that step's subsection.
3. **Flow chart.** This is a TikZ figure.
    - **For a workflow:** one box per step (its model, how many parameters and observables, the method). Arrows run from each step to the steps after it: solid for values passed on as fixed values, dashed for priors. A last box shows the merge into the target.
    - **For one calibration:** the model, observables and parameters feed the optimisation, followed by MCMC and identifiability when they were run.
4. **One subsection per step.** Each covers:
    - which model the step calibrates and against which observations;
    - the values it was given by earlier steps, and the priors it was given;
    - the protocol (experiments, pre-times, segment durations, inputs set per segment);
    - the observables (feature, observed value ± standard deviation, weight, experiment and segment);
    - the calibrated parameters with their ranges and priors;
    - the method, from CA's own description of it, with its settings, the solver, and the cost function as a formula;
    - MCMC settings when the step stored its posterior;
    - with `--run-dir`, the calibrated values and the best cost. A step that used priors reports its posterior medians.

### Citations

A parameter's source is its `data_reference` in the generated model's parameters CSV (the workflow's target, or the calibrated model). By convention it starts with the BibTeX key of the source, followed by a note: `martin2023modelling; Methods (p. 16): w_inf k = 10 mV`. Every word in it that is a key of the bibliography becomes `\cite{key}`; other words, such as `definitional` or `user_defined`, stay as text.

- **For a workflow,** the bibliography is the `*_references.bib` (and `*_references_proposed.bib`) of every module version in the target, as the module library keeps them. Pass `--bib` to use other files instead.
- **For one calibration,** pass the `.bib` files with `--bib` (repeatable). The parameters are the `<prefix>_parameters.csv` that generation writes beside the generated model, or `--parameters`.

The command writes the cited entries to a `.bib` beside the `.tex` (`methods.bib` for `-o methods.tex`), and the document ends with `\bibliography{methods}`. `--pdf` compiles it with `latexmk`, which runs BibTeX.

### Options

| Option | Meaning |
|---|---|
| `--workflow` | a `calibration_workflow.json`; with `--module-library-dir` for libraries the file does not list |
| `--model`, `--obs-data`, `--params-for-id` | one calibration instead of a workflow |
| `--settings` | the `user_inputs.yaml` of that calibration (method, optimiser options, solver, UQ) |
| `--run-dir` | a workflow's output directory, or a param_id run directory: adds the results |
| `--mapping` | the variable mapping CSV (default: the one `cuflynx-variable-mapping` writes) |
| `--parameters` | the generated model's parameters CSV (one calibration) |
| `--bib` | a `.bib` file the sources cite (repeatable) |
| `-o` | the `.tex` to write |
| `--title` | the document title |
| `--body-only` | the sections without a preamble, to `\input` into a paper; the packages it needs are listed in its first line |
| `--flowchart-only` | just the figure |
| `--pdf` | also compile it with `latexmk` |

`cuflynx-variable-mapping` writes beside the model by default, or where `-o` says.

The full document needs `amsmath`, `amssymb`, `booktabs`, `longtable`, `graphicx` and `tikz`. Compile it with `latexmk` (`--pdf` does this), or run pdflatex, BibTeX and pdflatex twice, so the references and the figure resolve.

From Python:

```python
from libcuflynx.reporting import workflow_methods_latex, calibration_methods_latex, update_variable_mapping_file
update_variable_mapping_file(model_path, mapping_path)
text = workflow_methods_latex(workflow_path, run_dir=run_dir, mapping=mapping_path)
```

!!! note
    The output is a starting point for the paper's methods, not the paper. Check the symbols, and reword the text to suit. A very long single equation can run past the margin: LaTeX cannot break an `align` line by itself.
