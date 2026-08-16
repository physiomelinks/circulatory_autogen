#!/bin/bash
# Resolve a libcuflynx console command, or explain how to get one.
#
# Sourced by the run_*.sh launchers. They used to invoke a file path
# (`${python_path} ../src/libcuflynx/scripts/param_id_run_script.py`), which meant every
# launcher had to know where the package lives -- a checkout layout that stops being true
# the moment libcuflynx is pip-installed. `pyproject.toml` declares a console entry point
# per stage instead, so the launcher just names the command and PATH resolves it.
#
# Usage:
#     source "${BASH_SOURCE[0]%/*}/cuflynx_entry_point.sh"
#     require_cuflynx cuflynx-param-id libcuflynx.scripts.param_id_run_script || exit 1
#     mpiexec -n "$1" "${cuflynx_cmd[@]}"
#
# On success `cuflynx_cmd` is an array holding the command to run. On failure it prints
# what to install and returns 1 -- deliberately before mpiexec is reached, because the
# alternative is an ImportError raised inside rank 3 of an MPI job, which is a much worse
# way to be told that `pip install -e .` was never run.
#
# Note that `python_path.sh` is not sourced here. It exists only for the OpenCOR route
# (see the deprecation note at the top of it); someone who pip-installed libcuflynx has no
# reason to have it configured. If you do want a specific interpreter -- OpenCOR's
# `pythonshell`, or a venv you have not activated -- set CUFLYNX_PYTHON to it and the
# stage runs as `$CUFLYNX_PYTHON -m <module>`, which needs no entry point on PATH.

require_cuflynx() {
    local entry_point="$1"
    local module="$2"

    # printf and command are builtins, and ${0##*/} is parameter expansion rather than
    # basename: everything below has to work in the shell whose PATH is the problem.
    if [ -n "${CUFLYNX_PYTHON:-}" ]; then
        if ! "${CUFLYNX_PYTHON}" -c 'import libcuflynx' >/dev/null 2>&1; then
            printf '%s\n' \
                "ERROR: CUFLYNX_PYTHON is set to '${CUFLYNX_PYTHON}', but that interpreter" \
                "       cannot import libcuflynx. Install it there with:" \
                "" \
                "           ${CUFLYNX_PYTHON} -m pip install -e ." \
                "" \
                "       (run from the repository root), or unset CUFLYNX_PYTHON to use the" \
                "       ${entry_point} command from your PATH." >&2
            return 1
        fi
        cuflynx_cmd=("${CUFLYNX_PYTHON}" -m "${module}")
        return 0
    fi

    if command -v "${entry_point}" >/dev/null 2>&1; then
        cuflynx_cmd=("${entry_point}")
        return 0
    fi

    printf '%s\n' \
        "ERROR: '${entry_point}' is not on your PATH, so libcuflynx is not installed in the" \
        "       Python environment active in this shell." \
        "" \
        "       From the root of this repository, once:" \
        "" \
        "           python3 -m venv venv" \
        "           source venv/bin/activate" \
        "           pip install -e ." \
        "" \
        "       Then re-run this script. In a new shell, activate the environment first." \
        "" \
        "       Already have an interpreter with libcuflynx installed (OpenCOR's pythonshell," \
        "       a conda env, a venv you would rather not activate)? Point CUFLYNX_PYTHON at it:" \
        "" \
        "           CUFLYNX_PYTHON=/path/to/python ./${0##*/} ..." >&2
    return 1
}
