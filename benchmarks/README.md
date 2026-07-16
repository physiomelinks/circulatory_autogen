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
| `three_compartment` | 3compartment cardiovascular (stiff, 20 s warmup) | yes | no (local only) |

The FitzHugh-Nagumo benchmark is also a normal pytest test
(`tests/test_param_id.py::test_compare_optimisers_on_fitzhugh_nagumo`) — the test and the
runner call the same `run_fitzhugh_nagumo` in `benchmark_specs.py`.

## Running

Everything, locally (needs the OpenCOR Python for the stiff 3compartment benchmark):

```bash
./benchmarks/run_benchmarks.sh                 # 1 MPI rank
./benchmarks/run_benchmarks.sh -n 8            # 8 MPI ranks
./benchmarks/run_benchmarks.sh --update-docs   # and splice results into the docs
```

Just the CI-safe set (no OpenCOR — plain Python with the pip-installed deps):

```bash
python benchmarks/run_benchmarks.py --set ci --update-docs
```

A single benchmark, with the regression assertions:

```bash
python benchmarks/run_benchmarks.py --benchmark fitzhugh_nagumo --assert
```

Options: `--set {all,ci}`, `--benchmark NAME`, `--update-docs`, `--assert`, `--num-calls N`.

AADC variants run only if AADC is installed **and** licensed; otherwise they are reported as
skipped. (The licence check must precede `import mpi4py`, which `run_benchmarks.py` handles.)

## Publishing to the docs

`--update-docs` rewrites the region between `<!-- BENCHMARK_RESULTS_START -->` and
`<!-- BENCHMARK_RESULTS_END -->` in `tutorial/docs/parameter-identification.md`. The
**Benchmarks** GitHub Actions workflow (`.github/workflows/benchmarks.yml`) does this
automatically on a weekly schedule (and on manual dispatch) for the CI-safe set, then commits
and deploys the docs. Because the numbers depend on the hardware and MPI rank count, treat the
published table as indicative, not exact.
