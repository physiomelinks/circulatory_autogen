#!/bin/bash
# Script to run pytest using OpenCOR's Python shell
# This ensures tests run with the correct Python environment that includes OpenCOR

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Source the OpenCOR Python shell path
source user_run_files/opencor_pythonshell_path.sh

# Check if pytest is installed in the OpenCOR Python environment
if ! ${opencor_pythonshell_path} -m pytest --version > /dev/null 2>&1; then
    echo "pytest is not installed in the OpenCOR Python environment."
    echo "Installing pytest and related packages from pyproject.toml..."
    # Try to install from pyproject.toml dev dependencies first
    if [ -f "pyproject.toml" ]; then
        ${opencor_pythonshell_path} -m pip install -e ".[dev]"
    else
        # Fallback to direct installation
        ${opencor_pythonshell_path} -m pip install pytest pytest-cov pytest-mpi pytest-xdist
    fi
fi

# Parse arguments and handle -n as the MPI process count (default 1)
PYTEST_ARGS=()
NUM_PROCS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            if [[ $# -gt 0 ]]; then
                NUM_PROCS="$1"
                shift
            else
                if command -v nproc >/dev/null 2>&1; then
                    NUM_PROCS=$(nproc)
                else
                    NUM_PROCS=4
                fi
            fi
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Always disable pytest-xdist workers when running under mpiexec to avoid
# spawning additional worker processes alongside MPI ranks.
PYTEST_ARGS+=("-p" "no:xdist")

# Ensure mpiexec is available
if ! command -v mpiexec >/dev/null 2>&1; then
    echo "mpiexec not found in PATH. Please install OpenMPI/MPICH or add mpiexec to PATH."
    exit 1
fi

MPIEXEC_BIN="${MPIEXEC_BIN:-mpiexec}"
MPIEXEC_ARGS=()
if [ -n "${MPIEXEC_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    MPIEXEC_ARGS+=(${MPIEXEC_EXTRA_ARGS})
fi
if "${MPIEXEC_BIN}" --version 2>/dev/null | grep -qi "Open MPI"; then
    MPIEXEC_ARGS+=(--mca orte_abort_on_non_zero_status 0)
    export OMPI_MCA_orte_abort_on_non_zero_status=0
fi

# Run pytest with the OpenCOR Python shell
# Pass all arguments to pytest and tee output to a log file
LOG_FILE="${SCRIPT_DIR}/log_test.log"
touch "$LOG_FILE"
echo "===== pytest run $(date -Iseconds) =====" | tee -a "$LOG_FILE"
echo "Running pytest with ${NUM_PROCS} MPI rank(s) via ${MPIEXEC_BIN}" | tee -a "$LOG_FILE"
"${MPIEXEC_BIN}" "${MPIEXEC_ARGS[@]}" -n "${NUM_PROCS}" "${opencor_pythonshell_path}" -m pytest "${PYTEST_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
exit "${PIPESTATUS[0]}"

