#!/bin/bash
# Staged ("sequential") calibration.
#
# NOTE: this stage is not currently implemented -- the SequentialParamID class the script
# drives is not part of libcuflynx. The command below says so and exits non-zero. Use
# ./run_param_id.sh for ordinary calibration.
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_sequential_param_id.sh num_processors'
    exit 1
fi

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-sequential-param-id libcuflynx.scripts.sequential_param_id_run_script || exit 1

"${_cuflynx_dir}/run_autogeneration.sh"

mpiexec -n $1 "${cuflynx_cmd[@]}"
