#!/bin/bash
# Run the optimiser benchmarks under the OpenCOR Python shell with MPI, passing through any
# run_benchmarks.py arguments. The benchmarks do not require OpenCOR (they run on
# Myokit/CasADi); this wrapper just gives a consistent local environment with MPI.
#
#   ./benchmarks/run_benchmarks.sh                 # all benchmarks, 1 rank
#   ./benchmarks/run_benchmarks.sh -n 8            # all benchmarks, 8 MPI ranks
#   ./benchmarks/run_benchmarks.sh --update-docs   # and splice results into the docs
#   ./benchmarks/run_benchmarks.sh --set ci        # only the CI-safe (non-OpenCOR) set
#
# The CI workflow does NOT use this wrapper: it runs `python benchmarks/run_benchmarks.py
# --set ci` under a plain Python with the pip-installed deps (no OpenCOR).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

source user_run_files/python_path.sh

# Parse -n as the MPI rank count (default 1); everything else passes to run_benchmarks.py.
BENCH_ARGS=()
NUM_PROCS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            if [[ $# -gt 0 ]]; then NUM_PROCS="$1"; shift
            elif command -v nproc >/dev/null 2>&1; then NUM_PROCS=$(nproc)
            else NUM_PROCS=4; fi
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
MPIEXEC_ARGS=()
if "${MPIEXEC_BIN}" --version 2>/dev/null | grep -qi "Open MPI"; then
    MPIEXEC_ARGS+=(--mca orte_abort_on_non_zero_status 0)
    export OMPI_MCA_orte_abort_on_non_zero_status=0
fi

echo "Running benchmarks with ${NUM_PROCS} MPI rank(s) via ${MPIEXEC_BIN}"
"${MPIEXEC_BIN}" "${MPIEXEC_ARGS[@]}" -n "${NUM_PROCS}" "${python_path}" \
    benchmarks/run_benchmarks.py "${BENCH_ARGS[@]}"
exit "${PIPESTATUS[0]}"
