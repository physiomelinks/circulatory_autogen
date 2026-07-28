# Where calibration time actually goes, and where speedups could come from

Findings from profiling the benchmark suite with `benchmarks/profile_benchmark.py`. Every number
below is measured, not estimated; reproduce with the commands in `PROFILING.md`. Ordered by
expected payoff.

## The headline: calibration is integrator-bound

3compartment, GA leg, 745 cost evaluations, 201.2 s total (single rank):

| layer | cumulative | share |
|---|---|---|
| `myokit_helper.run()` (the integrator) | 189.8 s | **94.4%** |
| `set_param_vals()` | 5.6 s | 2.8% |
| `reset_states()` | 5.0 s | 2.5% |
| everything else (cost assembly, observables, MPI, GA bookkeeping) | ~1 s | <0.5% |

FitzHugh-Nagumo agrees: `casadi._casadi.Function_call` is 39.0 s of 45.9 s (85%).

**Consequence: optimising Python glue cannot win more than ~6%.** The large wins are in doing less
integration, or in choosing an optimiser that needs fewer evaluations. Items 1-2 attack the
integration itself; items 3-4 are the worthwhile Python-side wins; item 5 is the algorithmic one.

## 1. 91% of every integration is warmup that is re-run every evaluation

`three_compartment_config` sets `pre_time: 20` and `sim_time: 2`. Every one of the ~19000 cost
evaluations in a full GA run integrates 22 s of model time, of which 20 s is unlogged warmup
discarded before the observables are computed.

CA already has the mechanism for this: `offline_pre_time` reaches a generic steady state once and
reuses it (`run_offline_pre_and_set_default_state`; see the obs_data docs). It can only absorb the
*parameter-independent* part of the settling — the calibrated parameters here (`q_lv_init`,
`aortic_root/C`, `E_lv_A`, `E_lv_B`) all move the steady state, so some per-evaluation warmup
remains. But even partial coverage attacks the dominant 91%.

Worth measuring: how much warmup is genuinely parameter-dependent. If the model settles in, say,
5 s from a generic steady state rather than 20 s from the CSV initial conditions, that alone is a
~3x cut to the whole benchmark.

## 2. `MaximumStep: 0.001` with no tolerances set

The base 3compartment config sets `solver_info: {MaximumStep: 0.001, MaximumNumberOfSteps: 5000}`
and no `rtol`/`atol`. Over 22 s of model time a 1 ms step cap forces **at least 22000 integrator
steps per evaluation** as a floor, regardless of how easy the dynamics are.

The FSA variant in the *same* benchmark uses `MaximumStep: 0.005` with `rtol/atol: 1e-9`, and
commit `9fe78d8` already documents that MaximumStep is a **cap, not an accuracy control** — rtol
and atol govern accuracy. The base config therefore pays for a tight cap without getting the
accuracy guarantee that tolerances would give.

Worth testing: raise `MaximumStep` and set explicit `rtol`/`atol`, then check the cost surface is
unchanged. This is a config change, not a code change, and it applies to the GA and CMA-ES legs
which currently carry the tightest cap.

## 3. Initial values are re-derived twice per cost evaluation (~5%)

`set_param_vals()` ends with `self.default_states = list(model.initial_values(as_floats=True))`,
and `protocol_executor` then calls `reset_states()`, which derives them again. Measured cost:
5.6 s + 5.0 s = **10.6 s of 201 s (5.3%)**, and it is where the Myokit model-validation traffic in
the profile comes from — `_validate_solvability` is called 1500 times for 745 evaluations, i.e.
exactly twice each, along with `create_unique_names` (1501) and `validate` (256500).

The second derivation is redundant: nothing changes the model between the two calls. Removing it
is a ~5% win on 3compartment and proportionally **much larger on cheap models**, where this fixed
per-evaluation overhead is a bigger share of a smaller total.

Care needed: `reset_states()` also applies the offline warm-up state and state overrides, so the
two are not blindly interchangeable — see the ordering discussion in the `change_states` work.

## 4. CasADi rebuilds its integrator on every cost evaluation

`casadi_python_solver_helper.run()` calls `ca.integrator("F", ...)` and `.mapaccum(total_steps)`
inside the run path (lines ~498-499). The profile shows exactly one construction per evaluation:

```
746  0.878s  casadi._casadi.Function_mapaccum      <- 746 cost evaluations
746  0.513s  casadi._casadi.integrator
746  0.172s  casadi._casadi.Function_map
2242 0.794s  casadi._casadi.new_Function
```

That is symbolic graph construction, not integration, and the structure depends only on the model,
`dt` and the integrator options — none of which vary between evaluations of the same calibration.
Caching the built `Function` and rebuilding only when the timeline or options change recovers
~2.4 s of 45.9 s (**5%**) on FitzHugh-Nagumo, and more on shorter simulations where construction is
a larger fraction. Present in both the old and current trees, so this is long-standing.

## 5. The algorithmic win: fewer evaluations, not faster ones

From the benchmark tables, on models with a known ground truth the multi-start gradient methods
reach the true parameters while the population methods do not, in less time:

| Goodwin | best cost | time | max param err |
|---|---|---|---|
| `genetic_algorithm` | 1.0099e-03 | 41.1 s | 1.4934 |
| `CMA-ES` | 5.4266e-04 | 88.0 s | 1.7682 |
| `multi_start (FD)` | 9.1452e-15 | 15.0 s | 0.0000 |

Since cost is ~94% integration and integration count scales with evaluations, the evaluation
budget *is* the runtime. A gradient method converging in hundreds of evaluations beats tuning the
integrator for a GA that spends 19344 of them. This is a recommendation about how to calibrate,
not a code change.

## Not worth pursuing

Micro-optimising observable extraction, cost assembly or the GA's bookkeeping: together they are
under 0.5% of runtime. The profile is unambiguous that there is nothing to win there.
