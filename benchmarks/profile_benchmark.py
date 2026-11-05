"""Profile a benchmark's calibration to find where the time actually goes.

Runs one optimiser leg of one benchmark under ``cProfile`` and writes both a ``.prof`` file (for
``snakeviz`` / ``pstats``) and a plain-text summary. Two uses:

* **Regression hunting** -- run the same command against two checkouts of the repo and diff the
  summaries. ``--root`` points the import path at an arbitrary source tree, so an older commit
  in a second worktree can be profiled with an identical harness::

      python benchmarks/profile_benchmark.py --root /path/to/old/worktree --tag old
      python benchmarks/profile_benchmark.py --root /path/to/new/worktree --tag new

* **Speedup hunting** -- read the cumulative-time table to see which layer (integrator, observable
  extraction, cost assembly, MPI) dominates a calibration.

Times are **CPU time** by default (``time.process_time``), so unrelated load on the machine
does not inflate them; pass ``--wall-clock`` for the old behaviour. CPU time does not normalise
for core speed, so on a heterogeneous CPU it is still perturbed -- for questions that must be
immune to that, compare call counts (see PROFILING.md).

Deliberately defaults to ``--num-calls`` far below a real benchmark budget: a profile only needs
enough cost evaluations for the hot path to dominate the fixed setup cost, and cProfile adds
noticeable overhead per call.

Run under a single MPI rank (``mpiexec -n 1`` or no launcher at all) unless you specifically want
to profile the parallel path -- with several ranks each writes its own profile and rank 0's is
distorted by the time it spends waiting on the others.
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time


def _add_root_to_path(root):
    """Import a checkout at ``root``: its ``src/`` (which holds ``libcuflynx``) plus the
    repo root itself, since this profiler is pointed at an arbitrary tree rather than
    at whatever is installed."""
    for path in (os.path.join(root, "src"), root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_base_config(root):
    """The user-inputs defaults, with machine-specific path overrides stripped."""
    import yaml
    with open(os.path.join(root, "user_run_files", "user_inputs.yaml")) as f:
        inp = yaml.load(f, Loader=yaml.FullLoader)
    for key in ("user_inputs_path_override", "resources_dir", "generated_models_dir",
                "param_id_output_dir", "param_id_obs_path"):
        inp.pop(key, None)
    return inp


# Benchmarks whose config builder takes the standard (base, resources, out, generated) signature.
_CONFIG_BUILDERS = {
    "three_compartment": "three_compartment_config",
    "fitzhugh_nagumo": "fitzhugh_nagumo_config",
    "goodwin": "goodwin_config",
    "teusink": "teusink_config",
}


def build_comparison(root, benchmark, method, num_calls, work_dir, comm):
    """Generate the model (rank 0) and return an OptimiserComparison ready to run ``method``."""
    from benchmarks import benchmark_specs as specs
    from benchmarks.compare_optimisers import OptimiserComparison
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture

    builder_name = _CONFIG_BUILDERS.get(benchmark)
    if builder_name is None or not hasattr(specs, builder_name):
        raise SystemExit(f"no config builder for benchmark '{benchmark}' in this tree; "
                         f"known: {sorted(_CONFIG_BUILDERS)}")
    builder = getattr(specs, builder_name)

    out_dir = os.path.join(work_dir, "output")
    gen_dir = os.path.join(work_dir, "generated", "myokit")
    if comm is None or comm.Get_rank() == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    base = _load_base_config(root)
    resources = os.path.join(root, "resources")
    try:
        config = builder(base, resources, out_dir, gen_dir)
    except TypeError:
        # goodwin_config takes a trailing param_id_method argument.
        config = builder(base, resources, out_dir, gen_dir, method)

    if comm is None or comm.Get_rank() == 0:
        if not generate_with_new_architecture(False, config):
            raise SystemExit(f"model generation failed for '{benchmark}'")
    if comm is not None:
        comm.Barrier()

    return OptimiserComparison(config, methods=[method], num_calls=num_calls)


def summarise(profiler, out_prefix, tag, wall, cpu, meta, top=40, unit="CPU"):
    """Write ``<prefix>.prof`` and ``<prefix>.txt``; return the text summary."""
    profiler.dump_stats(out_prefix + ".prof")

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    header = [f"# profile tag={tag}", f"# {meta}",
              f"# table times are {unit} time",
              f"# total wall: {wall:.1f}s   total CPU: {cpu:.1f}s", ""]

    buf.write("\n".join(header))
    buf.write(f"\n=== top {top} by CUMULATIVE time ===\n")
    stats.sort_stats("cumulative").print_stats(top)
    buf.write(f"\n=== top {top} by INTERNAL (tottime) time ===\n")
    stats.sort_stats("tottime").print_stats(top)

    text = buf.getvalue()
    with open(out_prefix + ".txt", "w") as f:
        f.write(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None,
                        help="source tree to profile (default: the tree holding this script)")
    parser.add_argument("--benchmark", default="three_compartment",
                        help="benchmark to profile (default: %(default)s)")
    parser.add_argument("--method", default="genetic_algorithm",
                        help="optimiser leg to profile (default: %(default)s)")
    parser.add_argument("--num-calls", type=int, default=744,
                        help="cost-evaluation budget; keep small, cProfile is not free "
                             "(default: %(default)s = one 3compartment GA generation)")
    parser.add_argument("--tag", default=None,
                        help="label for the output files (default: the root's basename)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write <tag>.prof/.txt (default: alongside the work dir)")
    parser.add_argument("--work-dir", default=None,
                        help="scratch dir for the generated model and param-id output")
    parser.add_argument("--top", type=int, default=40, help="rows per table (default: %(default)s)")
    parser.add_argument("--wall-clock", action="store_true",
                        help="measure wall-clock instead of CPU time (default is CPU time, which "
                             "is not inflated by unrelated load on the machine)")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tag = args.tag or os.path.basename(root)
    _add_root_to_path(root)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None
    rank = 0 if comm is None else comm.Get_rank()

    work_dir = args.work_dir or os.path.join("/tmp", f"profile_{tag}_{args.benchmark}")
    out_dir = args.out_dir or work_dir
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    comparison = build_comparison(root, args.benchmark, args.method, args.num_calls,
                                  work_dir, comm)

    # CPU time by default: time.process_time counts only cycles this process was actually ON a
    # CPU, so unrelated load on the machine no longer inflates the numbers. It does NOT normalise
    # for core speed -- the same work on a slower core costs more CPU-seconds, so a heterogeneous
    # CPU still perturbs it -- and under MPI it would count busy-wait spinning at barriers, which
    # is why this harness profiles a single rank. For questions that must be immune to all of
    # that, compare call counts (see PROFILING.md).
    if args.wall_clock:
        profiler = cProfile.Profile()
    else:
        profiler = cProfile.Profile(timer=time.process_time_ns, timeunit=1e-9)
    t0, c0 = time.time(), time.process_time()
    profiler.enable()
    comparison.run_method(args.method)
    profiler.disable()
    wall, cpu = time.time() - t0, time.process_time() - c0

    if rank != 0:
        return 0

    cost = comparison.results.get(args.method, {}).get("cost")
    meta = (f"root={root} benchmark={args.benchmark} method={args.method} "
            f"num_calls={args.num_calls} ranks={1 if comm is None else comm.Get_size()} "
            f"cost={cost}")
    prefix = os.path.join(out_dir, f"{tag}_{args.benchmark}_{args.method.replace(' ', '_')}")
    unit = "wall-clock" if args.wall_clock else "CPU"
    text = summarise(profiler, prefix, tag, wall, cpu, meta, top=args.top, unit=unit)

    print(f"PROFILE_DONE tag={tag} wall={wall:.1f}s cpu={cpu:.1f}s cost={cost} "
          f"out={prefix}.txt")
    # Echo just the cumulative table so a driver script can capture it from stdout.
    print(text.split("=== top")[1][:4000] if "=== top" in text else text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
