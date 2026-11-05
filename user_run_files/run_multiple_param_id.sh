#!/bin/bash
# Batch calibration over several observation files. NOTE: the script this drives still has
# one study's paths hardcoded in it and a TODO saying it needs porting to the user_inputs
# parser -- read it before relying on it.
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_multiple_param_id.sh num_processors'
    exit 1
fi

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx_module libcuflynx.scripts.run_multiple_param_id || exit 1

mpiexec -n "$1" "${cuflynx_cmd[@]}"
