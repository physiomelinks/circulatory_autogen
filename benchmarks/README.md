# Optimiser benchmarks

Wall-clock + accuracy comparisons of the parameter-identification optimisers (gradient-free
global search vs multi-start L-BFGS-B driven by different gradient backends).

## What's here

| File | Purpose |
|---|---|
| `benchmark_specs.py` | The benchmark definitions: `run_*` (execute a comparison → `BenchmarkResult`) and `assert_*` (regression checks). Shared by the tests and the runner. |
| `compare_optimisers.py` | `OptimiserComparison` — the harness that runs each method through `run_param_id` and collects costs/runtimes. |
| `docs_results.py` | Format `BenchmarkResult`s as Markdown and splice them into the docs. |
| `run_benchmarks.py` | CLI runner. |
| `run_benchmarks.sh` | Local wrapper: runs `run_benchmarks.py` under the OpenCOR Python + MPI. |

## Benchmarks

| Name | Model | Needs OpenCOR? | Runs in CI? |
|---|---|---|---|
| `fitzhugh_nagumo` | FitzHugh-Nagumo (non-stiff, multi-modal) | no | yes |
| `three_compartment` | 3compartment cardiovascular (stiff, 20 s warmup) | no (Myokit/CasADi) | yes (slow, ~20+ min) |

Both benchmarks run on Myokit and CasADi, so neither needs OpenCOR and both run in CI. AADC is
recorded as a skipped row on the 3compartment benchmark: its tape cost can only represent
observables whose operand is a state with a reimplemented op (max/min/mean), so 3compartment's
algebraic-variable observables (`aortic_root/u`) and its `max_minus_min` are dropped — AADC
would optimise a reduced cost, not the full one the other methods use. It stays off until it
can replicate the same cost (upstream issue #258).

The FitzHugh-Nagumo benchmark is also a normal pytest test
(`tests/test_param_id.py::test_compare_optimisers_on_fitzhugh_nagumo`) — the test and the
runner call the same `run_fitzhugh_nagumo` in `benchmark_specs.py`.

## Running

Everything, locally under the OpenCOR Python + MPI (the wrapper just gives a consistent
environment; the benchmarks themselves do not require OpenCOR):

```bash
./benchmarks/run_benchmarks.sh                 # 1 MPI rank
./benchmarks/run_benchmarks.sh -n 8            # 8 MPI ranks
./benchmarks/run_benchmarks.sh --update-docs   # and splice results into the docs
```

Or with any Python that has the deps installed (this is what CI does):

```bash
python benchmarks/run_benchmarks.py --set ci --update-docs
```

A single benchmark, with the regression assertions:

```bash
python benchmarks/run_benchmarks.py --benchmark fitzhugh_nagumo --assert
```

Options: `--set {all,ci}`, `--benchmark NAME`, `--update-docs`, `--assert`, `--num-calls N`.

## Parallel-scaling study (cores per column)

`--scaling` runs each benchmark once at each of several core counts and builds a table with one
**wall-clock column per core count** (default `1, 2, 4, 8, 16`), so you can see how each optimiser
speeds up with cores. Multi-start uses 16 starts and early-stopping is disabled, so every core
count runs the *same* work — the best cost is therefore core-independent (reported once) and the
per-core columns are pure wall-clock:

```bash
./benchmarks/run_benchmarks.sh --scaling                       # 1,2,4,8,16 cores, all benchmarks
./benchmarks/run_benchmarks.sh --scaling --update-docs         # and splice tables into the docs
./benchmarks/run_benchmarks.sh --cores 1,4,16 --benchmark fitzhugh_nagumo
```

How it works: in this mode `run_benchmarks.py` is an **orchestrator** — it launches its own
`mpiexec -n C` child per core count (each child runs the benchmark and hands its numbers back as
JSON), so it must *not* itself be under `mpiexec`. `run_benchmarks.sh` detects `--scaling`/`--cores`
and drops the outer `mpiexec` automatically (and exports `BENCH_PYTHON`, since the OpenCOR
`pythonshell` leaves `sys.executable` empty). Each child's numbers are cached under
`benchmarks/_results/<name>/scaling_<C>core.json`; `--from-cache` rebuilds the tables (and can
`--update-docs`) from those cached JSONs without re-running anything — handy when the benchmarks
were run separately.

Extra options: `--scaling`, `--cores C1,C2,...`, `--from-cache`.

AADC variants run only if AADC is installed **and** licensed; otherwise they are reported as
skipped. (The licence check must precede `import mpi4py`, which `run_benchmarks.py` handles.)

## Publishing to the docs

`--update-docs` rewrites the region between `<!-- BENCHMARK_RESULTS_START -->` and
`<!-- BENCHMARK_RESULTS_END -->` in `tutorial/docs/parameter-identification.md`. The
**Benchmarks** GitHub Actions workflow (`.github/workflows/benchmarks.yml`) does this
automatically on a weekly schedule (and on manual dispatch) for the CI-safe set, then — on the
default branch — **opens a PR** with the refreshed table (`benchmarks/refresh-results`).
Merging that PR lands the numbers in source and triggers the Docs Release workflow, which
republishes the tutorial. A direct push/deploy is not used because `master` is branch-protected
(the runner push is rejected) and a one-off `gh-deploy` would be overwritten by the next docs
deploy from source. Because the numbers depend on the hardware and MPI rank count, treat the
published table as indicative, not exact.
