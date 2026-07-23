"""Run the optimiser benchmarks and (optionally) publish their results to the docs.

Examples
--------
    # everything, under the OpenCOR Python + MPI wrapper (fixed rank count)
    ./benchmarks/run_benchmarks.sh --update-docs

    # the CI-safe set (no OpenCOR needed), and splice the results into the docs
    python benchmarks/run_benchmarks.py --set ci --update-docs

    # one benchmark, with regression assertions
    python benchmarks/run_benchmarks.py --benchmark fitzhugh_nagumo --assert

    # PARALLEL SCALING: run each benchmark at several core counts and build a table with one
    # wall-clock column per core count (this process orchestrates its own `mpiexec` children, so
    # do NOT wrap it in mpiexec -- run_benchmarks.sh handles that automatically):
    ./benchmarks/run_benchmarks.sh --scaling --update-docs
    python benchmarks/run_benchmarks.py --cores 1,2,4,8 --set ci

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
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

# docs_results and registry are deliberately MPI-free, so the scaling orchestrator can build
# tables and pick benchmarks WITHOUT importing mpi4py in this (parent) process -- importing MPI
# here would make it an MPI singleton and break the nested `mpiexec` child launches.
from benchmarks.docs_results import (
    results_to_markdown, scaling_results_to_markdown, update_docs_section,
    benchmark_result_to_dict, ScalingBenchmarkResult, ScalingRow)
from benchmarks.registry import BENCHMARK_CI

RESULTS_DIR = os.path.join(ROOT, "benchmarks", "_results")
DOCS_PATH = os.path.join(ROOT, "tutorial", "docs", "parameter-identification.md")

DEFAULT_SCALING_CORES = [1, 2, 4, 8]


def _load_base_config():
    """Base user-inputs dict with path overrides stripped (matches the test fixture)."""
    path = os.path.join(ROOT, "user_run_files", "user_inputs.yaml")
    with open(path) as f:
        inp = yaml.load(f, Loader=yaml.FullLoader)
    for key in ("user_inputs_path_override", "resources_dir", "generated_models_dir",
                "param_id_output_dir", "param_id_obs_path"):
        inp.pop(key, None)
    return inp


def _select_benchmarks(args, names):
    if args.benchmark:
        return [args.benchmark]
    if args.set == "ci":
        return [n for n in names if BENCHMARK_CI.get(n, False)]
    return list(names)


# --------------------------------------------------------------------------------------------
# Child mode: run the selected benchmarks once at the current mpiexec rank count.
# --------------------------------------------------------------------------------------------

def run_child(args):
    # Imported here (not at module top) so the scaling orchestrator never pulls in mpi4py.
    from mpi4py import MPI
    from benchmarks import benchmark_specs as specs

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    selected = _select_benchmarks(args, list(specs.BENCHMARKS.keys()))
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

        # Scaling orchestrator child: hand the results back as JSON.
        if args.emit_json and results:
            payload = {"num_ranks": comm.Get_size(),
                       "result": benchmark_result_to_dict(results[0])}
            with open(args.emit_json, "w") as f:
                json.dump(payload, f)
            print(f"[benchmarks] emitted results json: {args.emit_json}")

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


# --------------------------------------------------------------------------------------------
# Scaling orchestrator: run each benchmark once per core count via `mpiexec -n C` children.
# --------------------------------------------------------------------------------------------

def _mpi_launcher():
    """(mpiexec, extra_args, env) for launching child benchmark runs. Mirrors run_benchmarks.sh:
    silence OpenMPI's abort-on-non-zero. Does not oversubscribe -- see the comment below."""
    mpiexec = os.environ.get("MPIEXEC_BIN") or shutil.which("mpiexec") or "mpiexec"
    extra = []
    env = os.environ.copy()
    is_openmpi = False
    try:
        out = subprocess.run([mpiexec, "--version"], capture_output=True, text=True)
        ver = (out.stdout + out.stderr).lower()
        # This launcher reports itself as "OpenRTE" (Open MPI's runtime) rather than "Open MPI",
        # and points at open-mpi.org -- match any of these so oversubscription is actually enabled.
        is_openmpi = any(s in ver for s in ("open mpi", "openrte", "open-rte", "open-mpi"))
    except Exception:
        pass
    if is_openmpi:
        extra += ["--mca", "orte_abort_on_non_zero_status", "0"]
        env["OMPI_MCA_orte_abort_on_non_zero_status"] = "0"
        # Deliberately NOT --oversubscribe: OpenMPI counts *physical* cores as slots, so a core
        # count above the machine's physical cores is rejected ("not enough slots"). The
        # orchestrator then logs that leg as failed and simply omits its column, rather than
        # reporting a wall-clock measured under hyperthread contention (which is misleading -- an
        # oversubscribed point is not a real speedup). Only clean, non-oversubscribed core counts
        # are measured.
    return mpiexec, extra, env


def _build_scaling_result(name, cores, per_core):
    """Combine per-core child result dicts into one ScalingBenchmarkResult.

    ``per_core`` maps a core count to the child's ``benchmark_result_to_dict`` payload. Best cost
    and max param err are taken from the smallest core count present (they are core-independent
    because every core count runs the same work); the times form the per-core columns.
    """
    ref_core = min(per_core)
    ref = per_core[ref_core]
    methods = [row["method"] for row in ref["rows"]]

    rows = []
    for m in methods:
        times = {}
        cost = perr = skipped = None
        for c in cores:
            data = per_core.get(c)
            if not data:
                continue
            row = next((r for r in data["rows"] if r["method"] == m), None)
            if row is None:
                continue
            if row.get("skipped_reason"):
                skipped = row["skipped_reason"]
                continue
            times[c] = row["time_s"]
            if cost is None:
                cost, perr = row["cost"], row.get("param_err")
        if not times and skipped is not None:
            rows.append(ScalingRow(method=m, skipped_reason=skipped))
        else:
            rows.append(ScalingRow(method=m, cost=cost, param_err=perr, times_by_core=times))

    core_list = ", ".join(str(c) for c in cores)
    env_note = (f"cores: {core_list}; wall-clock seconds per core count; best cost / max param "
                f"err from the {ref_core}-core run (same work is run at every core count)")
    return ScalingBenchmarkResult(
        name=ref["name"], title=ref["title"], description=ref["description"], cores=cores,
        env_note=env_note, rows=rows, true_params=ref.get("true_params"),
        param_labels=ref.get("param_labels"))


def _child_python():
    """Interpreter for the mpiexec children. The OpenCOR ``pythonshell`` leaves ``sys.executable``
    empty, so prefer the path the wrapper exports (BENCH_PYTHON), then a couple of fallbacks."""
    for cand in (os.environ.get("BENCH_PYTHON"), sys.executable,
                 getattr(sys, "_base_executable", None)):
        if cand and os.path.exists(cand):
            return cand
    return shutil.which("python3") or shutil.which("python")


def _load_cached(jdir):
    """Read all cached ``scaling_<C>core.json`` files in ``jdir`` into a {cores: result} dict, so
    the scaling tables can be re-rendered (e.g. into the docs) without re-running the benchmarks."""
    per_core = {}
    if not os.path.isdir(jdir):
        return per_core
    for fname in os.listdir(jdir):
        if fname.startswith("scaling_") and fname.endswith("core.json"):
            try:
                c = int(fname[len("scaling_"):-len("core.json")])
                with open(os.path.join(jdir, fname)) as f:
                    per_core[c] = json.load(f)["result"]
            except (ValueError, KeyError, OSError):
                continue
    return per_core


def orchestrate(args):
    """Run each selected benchmark once per core count and build the scaling tables. This process
    must NOT be under mpiexec (it launches its own children)."""
    cores = [int(c) for c in str(args.cores).split(",") if str(c).strip()] if args.cores else []
    if not cores and not args.from_cache:
        print("[scaling] no valid core counts given")
        return 1
    selected = _select_benchmarks(args, list(BENCHMARK_CI.keys()))
    child_python = mpiexec = extra = env = None
    if not args.from_cache:
        child_python = _child_python()
        if not child_python:
            print("[scaling] could not resolve the Python interpreter for the mpiexec children; "
                  "set BENCH_PYTHON to the interpreter path (run_benchmarks.sh does this).")
            return 1
        mpiexec, extra, env = _mpi_launcher()

    scaling_results = []
    failures = []
    for name in selected:
        jdir = os.path.join(RESULTS_DIR, name)
        if args.from_cache:
            per_core = _load_cached(jdir)
            if per_core:
                print(f"[scaling] {name}: loaded cores {sorted(per_core)} from cache")
            else:
                print(f"[scaling] {name}: no cached results in {jdir}")
            if per_core:
                scaling_results.append(_build_scaling_result(name, sorted(per_core), per_core))
            continue
        per_core = {}
        for c in cores:
            os.makedirs(jdir, exist_ok=True)
            jpath = os.path.join(jdir, f"scaling_{c}core.json")
            if os.path.exists(jpath):
                os.remove(jpath)
            cmd = [mpiexec, *extra, "-n", str(c), child_python, os.path.abspath(__file__),
                   "--benchmark", name, "--emit-json", jpath]
            if args.num_calls is not None:
                cmd += ["--num-calls", str(args.num_calls)]
            print(f"\n[scaling] === {name} @ {c} core(s) ===", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, env=env)
            wall = time.time() - t0
            if proc.returncode != 0 or not os.path.exists(jpath):
                print(f"[scaling] {name} @ {c} core(s) FAILED (exit {proc.returncode}) "
                      f"after {wall:.1f}s")
                failures.append(f"{name}@{c}")
                continue
            with open(jpath) as f:
                per_core[c] = json.load(f)["result"]
            print(f"[scaling] {name} @ {c} core(s) done in {wall:.1f}s")
        if per_core:
            scaling_results.append(_build_scaling_result(name, cores, per_core))

    for result in scaling_results:
        print("\n" + "=" * 90)
        print(result.title + "  --  " + result.env_note)
        core_cols = "  ".join(f"{c:>7}c" for c in result.cores)
        print(f"  {'method':<26}{'cost':>12}   {core_cols}")
        for r in result.rows:
            if r.skipped_reason:
                print(f"  {r.method:<26} skipped: {r.skipped_reason}")
                continue
            tcols = "  ".join(
                (f"{r.times_by_core[c]:>7.1f} " if c in r.times_by_core else f"{'-':>8}")
                for c in result.cores)
            print(f"  {r.method:<26}{r.cost:>12.4e}   {tcols}")
    print("=" * 90)

    if scaling_results and (args.update_docs or args.results_out):
        note = ("Generated by `benchmarks/run_benchmarks.py --scaling`; wall-clock times depend "
                "on the hardware.")
        md = scaling_results_to_markdown(scaling_results, generated_note=note)
        if args.update_docs:
            changed = update_docs_section(md, DOCS_PATH)
            print(f"[scaling] docs {'updated' if changed else 'unchanged'}: {DOCS_PATH}")
        if args.results_out:
            with open(args.results_out, "w") as f:
                f.write(md)
            print(f"[scaling] results written: {args.results_out}")

    return 1 if failures else 0


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
                        help="also write the results table (markdown) to this file")
    parser.add_argument("--scaling", action="store_true",
                        help=f"parallel-scaling study over the default core counts "
                             f"({','.join(map(str, DEFAULT_SCALING_CORES))}); shorthand for --cores")
    parser.add_argument("--cores", default=None,
                        help="comma-separated core counts for a scaling study, e.g. 1,2,4,8 "
                             "(runs each benchmark once per count via mpiexec children); a count "
                             "above the machine's physical cores is skipped, not oversubscribed")
    parser.add_argument("--emit-json", default=None,
                        help="internal: a scaling child dumps its results here as JSON")
    parser.add_argument("--from-cache", action="store_true",
                        help="rebuild the scaling tables from the cached per-core JSON results "
                             "(no benchmarks are run); useful to (re)splice the docs")
    args = parser.parse_args(argv)

    if args.scaling and not args.cores:
        args.cores = ",".join(map(str, DEFAULT_SCALING_CORES))

    if args.cores or args.from_cache:
        return orchestrate(args)
    return run_child(args)


if __name__ == "__main__":
    sys.exit(main())
