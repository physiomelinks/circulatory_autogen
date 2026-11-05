# Smoothing a model

`cuflynx-smooth` writes a smooth copy of a flat CellML 2.0 model, for gradient-based solvers and calibrators
and for Operator Inference. Each non-smooth operation (a *site*) is replaced by a form built only from square
roots, reciprocals, products and sums, so the smoothed model lifts exactly to quadratic form.

```bash
cuflynx-smooth model_flat.cellml --runs runs.json        # reference runs give each site's range
cuflynx-smooth model_flat.cellml --ranges ranges.json    # or ranges from an earlier run
```

It writes `<name>_smooth.cellml` and `smoothing_report.json`. In Python: `libcuflynx.utilities.smooth_cellml`
(`probe`, `reference_ranges`, `smooth`, `smooth_cellml_file`).

## The rules

Each record of `smoothing_report.json` names its `rule`. Each site `k` adds named algebraic variables to its
component: `sm<k>_x` (the argument; a guarded division's denominator), a constant `eps_sm<k>` in the units of x,
`sm<k>_w = sqrt(x^2 + eps^2)` and `sm<k>_z = 1/w`, `sm<k>_H = (1 + x z)/2` where a rule needs a step, and
`sm<k>_q = 1/(x^2 + eps^2)` for a division. "Lifted" is the number of variables a site adds to the quadratic lift;
the report sums them as `lifted_estimate`.

| rule | pattern | example | smooth form | lifted (per use) | eps | when it is not smoothed |
|---|---|---|---|---|---|---|
| `abs` | `abs(x)` | `abs(v)` in a valve flow | `w` | w, z: 2 | `rel["abs"]` × range | exempt |
| `maxmin` | `max(a, b)`, `min(a, b)`, a − b changes sign | `max(T + ΔT, T_min)` | form `w` (default): `(a + b ± w)/2`; form `xH`: `b + (a − b)H`, `a − (a − b)H` | w: w, z: 2; xH: z, x z, z², w: about 4 | `rel["maxmin"]` × range | exempt |
| `rectifier` | `max(x, 0)`, `min(x, 0)` | `max(Cai − Cai_NE_base, 0)` | as `maxmin` with b = 0: `(x + w)/2` or `x H` | as `maxmin` | as `maxmin` | exempt |
| `piecewise` | `f if a ≥ b else g`; `and`, `or`, `not`; several pieces | valve opening rates | `g + H(f − g)`, split on one comparison at a time; the weights sum to 1; a comparison repeated either way round (`x ≤ 0.25`, `x > 0.25`) is one H | per comparison z, x z, plus products with the branches | `rel["piecewise"]` × range | exempt; a condition with `eq`, `neq`, `xor` or `floor` |
| `division` | `num/den`, den reaches or crosses 0 in a reference run | the GHK form `V/(1 − exp(−zFV/RT))` | `num den q`, `q = 1/(den² + eps_d²)` | q, den q, q²: about 3 | `rel["division"]` × range of den | den stays away from 0 (`not_needed`); den of constants alone is not probed |
| `exempt` | named in `exempt` | `chi_vfloor_final` | left as written; the report gives its `switch_variables` and whether each is a floor `phase` | 0 | – | – |
| `left_as_written` | `floor`, conditions with `eq`, `neq`, `xor` | `floor(s)` | left as written | 0 | – | – |

**eps.** `rel` (one number, or per kind `{"abs", "maxmin", "piecewise", "division"}`, and per equation
`{"variables": {"[component/]variable": rel}}`) times the range of x in the nominal (first) reference run.
An argument that keeps one sign in every run, such as an `abs(tau)` guard, takes `rel × min(range, smallest |x|)`,
so eps stays below its smallest value. No site is collapsed (replaced by its branch) unless `collapse` asks.

**Report flags.** `floor_upstream`, `branch_domain`, `one_sided`, `nan_in_runs`, `nan_at_init`, `zero_range`, and
`jump_candidate` (a piecewise with a constant branch that differs from another, whose result is only a state
derivative or is used only inside cos or sin: suggested for `exempt`, never applied).
