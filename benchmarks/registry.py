"""MPI-free registry of benchmark names and their CI eligibility.

The scaling orchestrator in ``run_benchmarks.py`` must NOT import ``mpi4py`` before it launches
its per-core ``mpiexec`` children (importing MPI in the parent would make it an MPI singleton and
break the nested launches). ``benchmark_specs`` pulls in the MPI-using run harness, so the
orchestrator selects benchmarks from this lightweight map instead. ``benchmark_specs.BENCHMARKS``
is built from the same keys, and ``tests/test_benchmarks.py`` asserts the two stay in sync.
"""

# name -> whether the benchmark runs in CI (must not need OpenCOR)
BENCHMARK_CI = {
    'fitzhugh_nagumo': True,
    'three_compartment': True,
    'goodwin': True,
}
