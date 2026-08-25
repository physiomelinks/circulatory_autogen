# Importing other model formats

A study's model is CellML. Two other formats are read *into* CellML on the way in,
so everything downstream -- the generators, the solvers, `params_for_id`'s
`component/variable` naming, the calibration and sensitivity pipelines -- keeps
seeing the CellML it already expects, and no solver or `model_type` has to know
these formats exist.

| Format | Extension | Reader |
| --- | --- | --- |
| Myokit | `.mmt` | `libcuflynx.parsers.MyokitParsers` |
| EasyML (openCARP) | `.model` | `libcuflynx.parsers.EasyMLParsers` |

## Myokit (`.mmt`)

```python
from libcuflynx.parsers.MyokitParsers import cellml_from_myokit, protocol_info_from_mmt

cellml, saved = cellml_from_myokit(data, filename="lr-1991.mmt", out_dir=outputs)
info, notes = protocol_info_from_mmt(data, filename="lr-1991.mmt")
```

The two halves of a `.mmt` are read separately and deliberately. The `[[model]]`
section becomes CellML **without** its protocol: baking Myokit's stimulus into
the exported model would give it two sources of pacing that disagree, since the
protocol belongs in obs_data. The `[[protocol]]` section becomes a one-experiment
`protocol_info` whose events are Myokit's own five fields -- `level`, `start`,
`length`, `period`, `multiplier` -- which are the same five
[`protocol_shapes`](api/protocols.md) uses, so they copy across unchanged.

A periodic Myokit protocol usually runs forever (`multiplier = 0`) while an
experiment here has a finite length, so an indefinite protocol is cut to
`beats` (default 2) and the cut is reported in `notes`.

## EasyML (`.model`)

```python
from libcuflynx.parsers.EasyMLParsers import import_easyml

read = import_easyml(data, filename="Courtemanche.model", out_dir=outputs)
read["cellml"]           # bytes
read["warnings"]         # what the reader had to decide -- show these
read["parameters"]       # the .param() group
read["protocol_info"]    # a default stimulus, offered not applied
```

EasyML is the language openCARP's published ionic models are written in. This is
an independent reader of the file format: no openCARP code is used or vendored,
because openCARP is distributed under the openCARP Academic Public License, which
is neither OSI-approved nor compatible with this package's Apache-2.0 licence.

Three things are implicit in EasyML, supplied by openCARP's own translator and so
supplied here too:

**Gating variables have no equation.** A Hodgkin-Huxley gate is written by giving
either `alpha_X`/`beta_X` (or the short `a_X`/`b_X`) or `tau_X`/`X_inf`; writing
`diff_X` as well is an error in EasyML. The state equation is reconstructed from
whichever pair is present.

**`X_init` is generated when absent**, at the steady state the pair implies. A
gate started at zero instead is a different simulation for the first few beats.

**There is no membrane equation.** A published model declares
`V; .nodal(); .external(Vm);` and `Iion; .nodal(); .external();` -- in openCARP
the tissue solver owns V. Read as written, such a model has no `dV/dt` at all, so
one is synthesised:

```
dot(V) = -(Iion + i_stim)
```

with no capacitance term, because EasyML's currents are in A/F, V is in mV and
time is in ms, and A/F *is* mV/ms. The sign is openCARP's: an inward (negative)
current depolarises, so a depolarising `i_stim` is **negative**. `i_stim` is an
ordinary variable, not a state, so a `protocol_info` can drive it.

### `.method()` groups are read, not executed

An EasyML file says how openCARP would step each state -- `rush_larsen` for
gates, `markov_be` for Markov chains, `cvode` for the rest. Those are
discretisation choices for a fixed-step tissue solver, where per-cell CVODE is
unaffordable across millions of cells and forward Euler is unstable for fast
gates. **The ODE system is the same either way**, and here it is integrated as one
system, which for single-cell work is at least as accurate -- it is the reference
the Rush-Larsen scheme approximates.

Dropping that silently would be the wrong kind of quiet, so every non-CVODE group
comes back in `warnings`, naming the states it covered.

### Units

EasyML records none. Its conventions are V in mV, time in ms, currents in A/F and
concentrations in mM. Variables are left dimensionless rather than annotated from
a guess: a partial annotation makes every equation mixing an annotated and an
unannotated term inconsistent, and libCellML reports those. The convention is
recorded in the model's metadata instead.

### Names

EasyML is flat. The reader puts everything in one component named after the model
(from the file's `name:` header, else its filename), so a variable addresses as
`Decker2009/GNa` -- the `component/variable` form `params_for_id` and obs
`operands` require.
