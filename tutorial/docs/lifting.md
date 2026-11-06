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

Expressions are sympy strings: parse with `sympy.sympify(expr, locals={name: sympy.Symbol(name) for name in symbols})`.

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
