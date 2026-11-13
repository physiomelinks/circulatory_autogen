#!/usr/bin/env bash
# The checks CI cannot run, required before tagging a libcuflynx release (CONTRIBUTING.md,
# "Making a release"). CI has no AADC licence, no libOpenCOR before Python 3.12 and no MSVC
# SUNDIALS, so here every backend is required: a test that skips for want of one fails
# (CUFLYNX_REQUIRED_BACKENDS, tests/_required_backends.py).
#
# Usage, from the repository root:
#   PYTHON=/path/to/venv/bin/python tests/run_release_checks.sh            # every check
#   PYTHON=... tests/run_release_checks.sh aadc libopencor                  # only these
#   tests/run_release_checks.sh --list                                      # the check names
#
# The venv needs `pip install -e ".[dev,aadc,uq,emulation,libopencor]"`, a Matlogica licence
# for AADC, and SUNDIALS_DIR for the C++ and batched CVODES builds. Each check is one pytest
# session of under 10 minutes; a check that runs longer is stopped and counted as failed.
# Ends with a pass/fail table, and exits non-zero unless every check passed.
set -u

PYTHON="${PYTHON:-python}"
RANKS="${RANKS:-2}"
LIMIT="${CHECK_SECONDS:-600}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${RELEASE_CHECK_LOGS:-${TMPDIR:-/tmp}/cuflynx-release-checks}"
cd "$ROOT" || exit 2
mkdir -p "$LOGS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ALL_BACKENDS="myokit,casadi,mpi4py,cmake,sundials,aadc,libopencor"

PYTEST=("$PYTHON" -m pytest -p no:xdist --tb=short -q -p no:cacheprovider)
MPI_PYTEST=(mpiexec -n "$RANKS" "$PYTHON" -m pytest -p no:xdist --tb=short -q -p no:cacheprovider --with-mpi)
MPI_REST_IGNORES=(--ignore=tests/test_param_id.py --ignore=tests/test_sensitivity_analysis.py
                  --ignore=tests/test_profile_likelihood.py --ignore=tests/executable)

# name | backends that must not skip | command
check_command() {
    case "$1" in
        backends)       echo "$ALL_BACKENDS|$PYTHON tests/_backend_info.py --require $ALL_BACKENDS" ;;
        aadc)           echo "aadc|${PYTEST[*]} -m 'not mpi' tests/test_aadc_solvers.py tests/test_aadc_conditionals.py tests/test_aadc_coupled_pipeline.py tests/test_aadc_vs_casadi_3compartment.py tests/test_bdf_kernel.py tests/test_param_id.py tests/test_solvers.py tests/test_experiment_time.py tests/test_solver_matrix.py -k aadc" ;;
        libopencor)     echo "libopencor|${PYTEST[*]} -m 'not mpi' tests/ -k 'libopencor'" ;;
        solver-matrix)  echo "$ALL_BACKENDS|${PYTEST[*]} tests/test_solver_matrix.py" ;;
        main-*)         echo "myokit,casadi,mpi4py,cmake,sundials|${PYTEST[*]} -m 'not mpi' tests/ --ci-shard main:${1#main-}" ;;
        mpi-rest-*)     echo "myokit,casadi,mpi4py|${MPI_PYTEST[*]} -m 'mpi and not full_scale_model' tests/ ${MPI_REST_IGNORES[*]} --ci-shard mpi-rest:${1#mpi-rest-}" ;;
        param-id-*)     echo "myokit,mpi4py|${MPI_PYTEST[*]} -m 'mpi and not full_scale_model' tests/test_param_id.py tests/test_sensitivity_analysis.py tests/test_profile_likelihood.py --ci-shard param-id:${1#param-id-}" ;;
        notebooks)      echo "myokit|${PYTEST[*]} tests/interactive_tests" ;;
        *)              return 1 ;;
    esac
}

ALL_CHECKS=(backends aadc libopencor solver-matrix
            main-rest main-autogeneration main-module-arrays main-simulation main-cpp main-misc
            mpi-rest-rest mpi-rest-emulators param-id-rest param-id-heavy notebooks)

if [ "${1:-}" = "--list" ]; then
    printf '%s\n' "${ALL_CHECKS[@]}"
    exit 0
fi
CHECKS=("${ALL_CHECKS[@]}")
[ $# -gt 0 ] && CHECKS=("$@")

"$PYTHON" tests/_backend_info.py
echo

declare -A RESULT SECONDS_TAKEN
for name in "${CHECKS[@]}"; do
    line="$(check_command "$name")" || { echo "unknown check: $name (see --list)"; exit 2; }
    required="${line%%|*}"
    command="${line#*|}"
    log="$LOGS/$name.log"
    echo "== $name: $command"
    start=$(date +%s)
    CUFLYNX_REQUIRED_BACKENDS="$required" timeout "$LIMIT" bash -c "$command" > "$log" 2>&1
    status=$?
    SECONDS_TAKEN[$name]=$(( $(date +%s) - start ))
    if [ $status -eq 0 ]; then
        RESULT[$name]=pass
    elif [ $status -eq 124 ]; then
        RESULT[$name]="FAIL (over ${LIMIT}s)"
    else
        RESULT[$name]="FAIL (exit $status)"
        tail -n 15 "$log"
    fi
    echo "   ${RESULT[$name]} in ${SECONDS_TAKEN[$name]}s, log $log"
done

echo
echo "Release checks"
printf '  %-22s %-22s %s\n' check result seconds
failed=0
for name in "${CHECKS[@]}"; do
    printf '  %-22s %-22s %s\n' "$name" "${RESULT[$name]}" "${SECONDS_TAKEN[$name]}"
    [ "${RESULT[$name]}" = pass ] || failed=1
done
if [ $failed -eq 0 ]; then
    echo "ALL PASSED"
else
    echo "FAILED: fix these before tagging (logs in $LOGS)"
fi
exit $failed
