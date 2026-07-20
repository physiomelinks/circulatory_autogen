"""Run the optimiser benchmarks and (optionally) publish their results to the docs.

Examples
--------
    # everything, under the OpenCOR Python + MPI wrapper
    ./benchmarks/run_benchmarks.sh --update-docs

    # the CI-safe set (no OpenCOR needed), and splice the results into the docs
    python benchmarks/run_benchmarks.py --set ci --update-docs

    # one benchmark, with regression assertions
    python benchmarks/run_benchmarks.py --benchmark fitzhugh_nagumo --assert

``--set ci`` selects the benchmarks that run without OpenCOR (currently all of them); that is
what the GitHub Actions workflow runs. ``--update-docs`` rewrites the marker-delimited results
region of tutorial/docs/parameter-identification.md.
"""
# AADC's licence check must run before mpi4py is imported (importing MPI first breaks it),
# so probe it here at the very top, before any other project import pulls in mpi4py.
_AADC_OK = False
try:  # pragma: no cover - depends on optional licensed dependency
    import aadc as _aadc
    import numpy as _np_probe
    _f = _aadc.Functions(); _f.start_recording()
    _x = _aadc.idouble(1.0); _xa = _x.mark_as_input()
    _y = _x * _x; _yr = _y.mark_as_output(); _f.stop_recording()
    _aadc.evaluate(_f, {_yr: [_xa]}, {_xa: _np_probe.array([2.0])}, _aadc.ThreadPool(1))
    _AADC_OK = True
except Exception:
    _AADC_OK = False

import argparse
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from mpi4py import MPI

from benchmarks import benchmark_specs as specs
from benchmarks.docs_results import results_to_markdown, update_docs_section

RESULTS_DIR = os.path.join(ROOT, "benchmarks", "_results")
DOCS_PATH = os.path.join(ROOT, "tutorial", "docs", "parameter-identification.md")


def _load_base_config():
    """Base user-inputs dict with path overrides stripped (matches the test fixture)."""
    path = os.path.join(ROOT, "user_run_files", "user_inputs.yaml")
    with open(path) as f:
        inp = yaml.load(f, Loader=yaml.FullLoader)
    for key in ("user_inputs_path_override", "resources_dir", "generated_models_dir",
                "param_id_output_dir", "param_id_obs_path"):
        inp.pop(key, None)
    return inp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["all", "ci"], default="all",
                        help="'ci' = only benchmarks that run without OpenCOR (default: all)")
    parser.add_argument("--benchmark", default=None,
                        help="run a single benchmark by name (overrides --set)")
    parser.add_argument("--update-docs", action="store_true",
                        help="splice the results into tutorial/docs/parameter-identification.md")
    parser.add_argument("--assert", dest="do_assert", action="store_true",
                        help="also run the regression assertions for each benchmark")
    parser.add_argument("--num-calls", type=int, default=None,
                        help="override the population-method evaluation budget")
    parser.add_argument("--results-out", default=None,
                        help="also write the results table (markdown) to this file, e.g. for CI "
                             "to attach/email")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.benchmark:
        selected = [args.benchmark]
    elif args.set == "ci":
        selected = [name for name, spec in specs.BENCHMARKS.items() if spec["ci"]]
    else:
        selected = list(specs.BENCHMARKS.keys())

    base_config = _load_base_config()
    resources_dir = os.path.join(ROOT, "resources")

    results = []
    failures = []
    for name in selected:
        spec = specs.BENCHMARKS.get(name)
        if spec is None:
            if rank == 0:
                print(f"[benchmarks] unknown benchmark '{name}', skipping")
            continue
        output_dir = os.path.join(RESULTS_DIR, name, "output")
        gen_dir = os.path.join(RESULTS_DIR, name, "generated")
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(gen_dir, exist_ok=True)
            print(f"\n[benchmarks] running '{name}' ...")
        comm.Barrier()

        run_kwargs = {}
        if name == "fitzhugh_nagumo":
            run_kwargs["include_aadc"] = _AADC_OK
        if args.num_calls is not None:
            run_kwargs["num_calls"] = args.num_calls
        try:
            result = spec["run"](base_config, resources_dir, output_dir, gen_dir, comm,
                                 **run_kwargs)
            results.append(result)
            if args.do_assert:
                spec["assert"](result, comm)
        except Exception as exc:  # keep going so one failure doesn't lose the rest
            if rank == 0:
                print(f"[benchmarks] '{name}' FAILED: {exc}")
                traceback.print_exc()
            failures.append(name)
        comm.Barrier()

    if rank == 0:
        for result in results:
            print("\n" + "=" * 84)
            print(result.title + "  --  " + result.env_note)
            for r in result.rows:
                if r.skipped_reason:
                    print(f"  {r.method:<26} skipped: {r.skipped_reason}")
                else:
                    err = "" if r.param_err is None else f"  max_param_err={r.param_err:.4f}"
                    print(f"  {r.method:<26} cost={r.cost:.4e}  time={r.time_s:7.1f}s{err}")
        print("=" * 84)

        if results and (args.update_docs or args.results_out):
            note = ("Generated by `benchmarks/run_benchmarks.py`; numbers depend on the "
                    "hardware and MPI rank count used.")
            md = results_to_markdown(results, generated_note=note)
            if args.update_docs:
                changed = update_docs_section(md, DOCS_PATH)
                print(f"[benchmarks] docs {'updated' if changed else 'unchanged'}: {DOCS_PATH}")
            if args.results_out:
                with open(args.results_out, "w") as f:
                    f.write(md)
                print(f"[benchmarks] results written: {args.results_out}")

    comm.Barrier()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
