#!/bin/bash
# Run the optimiser benchmarks with MPI, passing through any run_benchmarks.py arguments. The
# benchmarks only need Myokit/CasADi (no OpenCOR), so they run under a plain Python venv.
#
#   ./benchmarks/run_benchmarks.sh                 # all benchmarks, 1 rank
#   ./benchmarks/run_benchmarks.sh -n 8            # all benchmarks, 8 MPI ranks
#   ./benchmarks/run_benchmarks.sh --update-docs   # and splice results into the docs
#   ./benchmarks/run_benchmarks.sh --set ci        # only the CI-safe (non-OpenCOR) set
#   ./benchmarks/run_benchmarks.sh --scaling       # core-scaling study (1,2,4,8,16 cores)
#
# In --scaling / --cores mode run_benchmarks.py is the orchestrator and launches its own
# `mpiexec -n C` children per core count, so this wrapper runs it WITHOUT an outer mpiexec.
#
# Interpreter selection (in order): $BENCH_PYTHON, then /venv/bin/python, then -- only as a
# fallback -- OpenCOR's bundled pythonshell (deprecated: it is heavier, ~2 GB of RSS per MPI rank,
# and on its way out; prefer a venv with `pip install -e ".[dev]"` + myokit/casadi).
#
# The CI workflow does NOT use this wrapper: it runs `python benchmarks/run_benchmarks.py
# --set ci` under its own Python with the pip-installed deps.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

# Resolve the benchmark Python interpreter. Prefer a normal venv over OpenCOR's pythonshell.
if [[ -n "${BENCH_PYTHON}" ]]; then
    PY="${BENCH_PYTHON}"
elif [[ -x /venv/bin/python ]]; then
    PY=/venv/bin/python
else
    # Fallback: OpenCOR's bundled pythonshell (deprecated; see header).
    source user_run_files/python_path.sh
    PY="${python_path}"
fi
if [[ ! -x "${PY}" ]] && ! command -v "${PY}" >/dev/null 2>&1; then
    echo "Benchmark Python interpreter not found: '${PY}'."
    echo "Set BENCH_PYTHON to a Python that has the deps installed (mpi4py, myokit, casadi, numpy,"
    echo "scipy, pyyaml), create /venv, or fix user_run_files/python_path.sh."
    exit 1
fi
# Export it so the scaling orchestrator launches its mpiexec children with the same interpreter
# (also needed because the OpenCOR pythonshell leaves sys.executable empty).
export BENCH_PYTHON="${PY}"

# Parse -n as the MPI rank count (default 1); everything else passes to run_benchmarks.py.
# Detect --scaling / --cores: in that mode run_benchmarks.py orchestrates its own mpiexec
# children, so we must NOT wrap it in an outer mpiexec.
BENCH_ARGS=()
NUM_PROCS=1
SCALING=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            if [[ $# -gt 0 ]]; then NUM_PROCS="$1"; shift
            elif command -v nproc >/dev/null 2>&1; then NUM_PROCS=$(nproc)
            else NUM_PROCS=4; fi
            ;;
        --scaling|--cores|--from-cache)
            SCALING=1; BENCH_ARGS+=("$1"); shift
            ;;
        *)
            BENCH_ARGS+=("$1"); shift
            ;;
    esac
done

if ! command -v mpiexec >/dev/null 2>&1; then
    echo "mpiexec not found in PATH. Please install OpenMPI/MPICH or add mpiexec to PATH."
    exit 1
fi

MPIEXEC_BIN="${MPIEXEC_BIN:-mpiexec}"
export MPIEXEC_BIN

echo "Using benchmark Python interpreter: ${PY}"

if [[ "${SCALING}" -eq 1 ]]; then
    # Orchestrator mode: run_benchmarks.py launches its own mpiexec children per core count.
    echo "Running core-scaling benchmarks (orchestrator launches its own mpiexec children)"
    "${PY}" benchmarks/run_benchmarks.py "${BENCH_ARGS[@]}"
    exit "$?"
fi

MPIEXEC_ARGS=()
if "${MPIEXEC_BIN}" --version 2>/dev/null | grep -qi "Open MPI"; then
    MPIEXEC_ARGS+=(--mca orte_abort_on_non_zero_status 0)
    export OMPI_MCA_orte_abort_on_non_zero_status=0
fi

echo "Running benchmarks with ${NUM_PROCS} MPI rank(s) via ${MPIEXEC_BIN}"
"${MPIEXEC_BIN}" "${MPIEXEC_ARGS[@]}" -n "${NUM_PROCS}" "${PY}" \
    benchmarks/run_benchmarks.py "${BENCH_ARGS[@]}"
exit "${PIPESTATUS[0]}"
