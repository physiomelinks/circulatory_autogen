# Lifting a model to quadratic form

`cuflynx-lift` lifts a smooth flat CellML 2.0 model to a model that is exactly quadratic in its state, for Operator
Inference with Lift & Learn (Qian, Kramer, Peherstorfer & Willcox 2020). It adds auxiliary states w = T(s) to the
original states s so that the lifted state obeys

    dw/dt = A w + H (w ⊗ w) + Σ_k N_k w u_k + B u + C

with inputs u. The original states are the first n entries of w, so the reverse map is w[:n].

```bash
cuflynx-smooth model_flat.cellml --runs runs.json                 # first remove abs, max, min, piecewise
cuflynx-lift model_flat_smooth.cellml --smoothing-report smoothing_report.json --inputs P_in,Q_out
```

It needs `pip install "libcuflynx[lift]"`. In Python: `libcuflynx.utilities.lift_cellml` (`lift`,
`lift_cellml_file`, `verify`, `structure`).

## What it writes

| file | contents |
|---|---|
| `<name>_lifted.cellml` | the lifted model: one component, every variable dimensionless; each auxiliary starts at T(s(0)) at the model's parameter values; inputs keep their expressions in time and the original states |
| `lifting.json` | `states` (myokit qnames, in myokit's order), `lifted` (the original states, then the auxiliaries), `auxiliaries` (`expr` = T, `jac` = dT/ds, `exact`, `epsilon`, `scale`), `inputs`, `operators` (A dense, H as COO over pairs i ≤ j in `numpy.triu_indices(n_w)` order, N per input feature, B, C; also `symbolic`, in the parameters), `engine`, `verification`, `check`, `structure`, `conditioning` |
| `lifting_functions.py` | vectorised numpy: `T(X)` (n_snap, n) → (n_snap, n_w), `J(X)` → (n_snap, n_w, n), `inputs(t, X)`, `features(U)`, `rhs_original(X, U)`, `rhs_lifted(W, U)`, `operators()`; each takes `p` to override parameters |

## lifting.json (schema version 1)

`lifting.json` is read by Operator Inference and by POD-Galerkin reduction. Its fields are stable: a later version
only adds fields and raises `schema_version`; none is renamed or removed. `check_lifting_json(doc)` lists a document's
problems against this table (`LIFTING_JSON_FIELDS` in the module).

| field | type | always | meaning |
|---|---|---|---|
| `format` | str | yes | always "libcuflynx lift_cellml" |
| `schema_version` | int | yes | the schema version (1); later versions only add fields |
| `parse` | str | yes | how to parse the expression strings and the H pair indices |
| `states` | list | yes | the original states kept, as myokit qnames, in myokit's order: the first n entries of w |
| `state_symbols` | list | yes | their names in the expressions |
| `lifted` | list | yes | the names of w in order: the original states, the held inputs, the smoothing sites, the auxiliaries |
| `reverse` | str | yes | the reverse map, w[:n] |
| `held_inputs` | list | yes | inputs held as states (du/dt = 0 within a chunk): {name, qname, kind, index, expr (a held function), note} |
| `sites` | list | yes | smoothing sites lifted in closed form: {name, kind (w, z, q, or x: a site's argument when it is a sum), k, qname, x, epsilon} |
| `size` | dict | yes | {n, n_held, n_sites, n_w, n_aux, n_inputs, n_features} |
| `symbols` | dict | yes | {expression name: component.variable} |
| `time` | str | yes | the name of time |
| `auxiliaries` | list | yes | each lifted variable after the states: {name, expr (T), jac (dT/d states and held inputs), jac_inputs (dT/du, when it depends on an input), exact, epsilon, rule, scale} |
| `inputs` | list | yes | {name, expr, kind (variable, cut, time, input_function, nonsmooth, derivative, nonpolynomial_term), qname, of, switch_variables} |
| `input_features` | list | yes | the input monomials the B and N operators use, in order |
| `parameters` | dict | yes | {name: value} at which the numeric operators are evaluated |
| `derived_parameters` | dict | yes | {name: expression in the parameters} |
| `operators` | dict | yes | present (which of A, H, N, B, C are non-zero), A (n_w x n_w), H ({rows, pairs, values, shape}: COO over the pairs i <= j in numpy.triu_indices(n_w) order), N (one n_w x n_w per input feature), B (n_w x n_f), C (n_w), sparsity, symbolic (the same in the parameters) |
| `initial` | dict | yes | {w0}: T at the initial state and the parameters |
| `engine` | dict | yes | which engine ran each stage, QBee errors, input_free, notes, sizes of the polynomial system |
| `timings` | dict | yes | seconds of each stage |
| `invariants` | list | yes | g(w) = 0 exactly: {expr, degree, origin (definition, conserved, oscillator), variable, max_reference, relative_reference, max_lifted_run, relative_lifted_run}, or {variable, skipped, origin} where a definition has no polynomial identity |
| `python` | str | no | the generated module: T, J, JU, inputs, features, rhs_original, rhs_lifted, operators |
| `check` | dict | no | the operator form against J f on the model's trajectory: {max_relative_residual, per_equation} |
| `verification` | dict | no | simulated, states, manifold, max_state_error, max_manifold_defect, defect_growth (times past 1e-8, 1e-6, 1e-3 and the rate), jf |
| `structure` | dict | no | clocks, inputs (kind, multiplies_state), blocks (block-triangular pattern), conserved |
| `conditioning` | dict | no | {scales: each variable's min, max, max_abs over the run, ratio, note} |
| `smoothing_sites` | list | no | each smoothing site: rule, smoothness class before and after |
| `outputs` | list | no | named variables lifted with the states: {name, qname, expr (a polynomial in the lifted names and inputs: quadratic where the lifted variables allow), degree, original} |
| `model` | str | no | the model lifted |
| `lifted_model` | str | no | the lifted CellML model |
| `libcellml` | dict | no | Validator and Analyser issues of the lifted CellML model |

Expressions are sympy strings: parse with `sympy.sympify(expr, locals={name: sympy.Symbol(name) for name in symbols})`.

## Options

- `--inputs a,b`: variables kept as inputs u (ports, open boundary conditions).
- `--as-inputs a,b` (or `component/*`): variables cut from the model; their values come from data (kind `cut`). The
  lifted CellML holds them at their value at t = 0, so the verification checks the lifting on the model's own run
  instead of simulating it.
- `--keep comp1,comp2`: keep only the states of these components; every other state they read is cut. Sites and cut
  inputs the kept states do not reach are dropped.
- `--held-inputs`: named and cut inputs become states with du/dt = 0, held within a chunk or beat (`held_inputs`); a
  function of held inputs alone becomes one held variable `h_k` (`kind: held_function`, with its `expr`). With no
  inputs left the system is polynomial without inputs, so an input inside a square root or a tanh lifts.
- `--outputs a,b`: variables lifted with the states (e.g. a drug effect that feeds no rate here): their non-polynomial
  parts become auxiliaries, and `outputs` gives each as a polynomial in the lifted variables.
- `--no-sites`: polynomialize the smoothing sites' square roots instead of lifting the sites in closed form.

## Smoothing sites

The smoothing pass's `sm<k>_w = sqrt(x^2 + eps^2)`, `sm<k>_z = 1/w` and `sm<k>_q = 1/(x^2 + eps^2)` are kept whole:
lifted variables with closed-form rates, dw/dt = x x' z, dz/dt = -x x' z^3, dq/dt = -2 x x' q^2, where x is the site's
argument in the states and other sites. When the argument is a sum, it is a lifted variable too (`sm<k>_x`, kind
`x`), so that x z x' has as many terms as x', not their product. The lifted size then grows with the number of sites, not with the depth of
the curves inside them. `H = (1 + x z)/2` is a polynomial in x and z.

## Invariants

`invariants` lists polynomial identities g(w) = 0 that hold exactly among the lifted variables, so that a data
matrix built from lifted snapshots is known to be collinear along them:

- `definition`: each lifted variable's own: a monomial w - x y; a site's w^2 - x^2 - eps^2, z w - 1, q (x^2 + eps^2) - 1;
  a power g^(a/b) as w^b - g^a (w^b g^(-a) - 1 for a < 0); sin and cos as s^2 + c^2 - 1. An exp, log, tanh or
  symbolic power has none and is recorded as `skipped`.
- `conserved`: linear functionals with no rate, l.w - l.w(0).
- `oscillator`: clock norms c^2 + s^2 - r^2.

Each is checked on T of the model's run and on the lifted run: max |g| and max |g| over the sum of |its terms|.

## How

1. **Read.** The MathML becomes sympy, one symbol per connected set of variables. Constants are parameters;
   algebraic equations are substituted into the state right-hand sides.
2. **Inputs.** Named variables (`--inputs`: ports, open boundary conditions) stay symbols. A subexpression of time
   alone (a pacing waveform) becomes an input, as does a non-smooth site left by the smoothing pass (an exempt
   site, a `floor`): the lifted model computes it from the original states, as the original does. A non-monomial
   expression of parameters alone becomes a derived parameter.
3. **Polynomialize.** QBee's `polynomialize`, in a subprocess killed after `--time-budget` seconds and checked
   against the chain rule. If it fails, a rule table: each non-polynomial f(g) becomes w = f(g) with
   dw/dt = f'(g) dg/dt rewritten in the auxiliaries — exp(g): w g′; ln g: g′ z, z = 1/g; g^(a/b): w = g^(1/b),
   z = g^(−1/b); 1/D: −w² D′; sin and cos as a pair; tanh: (1 − w²) g′.
4. **Quadratize.** QBee's `quadratize(input_free=True)` within the budget, otherwise a greedy search that adds the
   monomial removing the most terms of degree > 2. A term with an input may carry one state factor (an N term).
   Auxiliaries whose T is constant or equal to another's are then removed.

The search works on the coefficients' values at two parameter points, so a model's parameter expressions are never
multiplied out; the symbolic operators are formed once, for the variables kept. A polynomial system of more than
`--max-terms` terms (20000) is not quadratized: the lift stops with its size (e.g. nested square roots of a smoothed
dissociation curve give terms of degree 24).

Input-free: the auxiliaries depend on the states only. A non-polynomial term that mixes states and inputs takes
the inputs' derivatives as further inputs (`d<u>_dt`) when they are known functions of time, and `engine.input_free`
is false; such auxiliaries depend on the inputs, so `T`, `J` and `JU` (dT/du) take `t` or `U` (`NEEDS_INPUTS`) and
`auxiliaries[].jac_inputs` gives dT/du. Some systems have no finite input-affine quadratic lift (u x² needs x³, then u x⁴, ...); the search then
stops at `--max-aux` and says so.

## Verification

`verify` runs the lifted and the original model with CVODE_myokit (rtol 1e-10) from t = 0:

- `states`: the largest error of each original state, over its scale max(range, |mean|, atol, 1e-8 x the largest
  state);
- `manifold`: max |w − T(s)| of each auxiliary along the lifted run, over its scale;
- `defect_growth`: when the defect and the state error first pass 1e-8, 1e-6 and 1e-3, and the defect's exponential
  rate. The manifold w = T(s) is invariant but need not attract: for FitzHugh–Nagumo with w = V², solver error grows
  off it at about 2c(1 − 2V²/3), so a long lifted run leaves the original trajectory while J f still matches;
- `jf`: J(s) f(s) against `rhs_lifted(T(s))` at sampled states, relative to the size of the terms that cancel
  (|J| |f|) or |w|, whichever is larger;
- `check` (top level): the operator form against J f on the reference trajectory, per equation.

## Structure

`structure` in lifting.json: `clocks` (state pairs with c² + s² conserved, or 1 on the unit circle, their rate and
whether it is constant, a known function of time or state-dependent), each input's kind (held, known function of
time, state-dependent) and whether it multiplies the state (N ≠ 0), `blocks` (strongly connected blocks of the
lifted system in order, which feeds which, and whether through A or H), and `conserved` linear functionals ℓ with
ℓᵀ(A w + H(w ⊗ w) + N w u) = 0, their rate ℓᵀ(B u + C) and their drift along the lifted run. With
`--smoothing-report`, `smoothing_sites` gives each site's smoothness class before and after smoothing.
