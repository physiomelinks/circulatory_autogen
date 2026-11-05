#!/bin/bash
# Set every parameter of a parameters file (nested JSON) in the one named by input_param_file
# in user_inputs.yaml, in place.
# Usage: ./read_and_insert_parameters.sh parameters_to_add.json
#    or: ./read_and_insert_parameters.sh parameters.json parameters_to_add.json
# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx_module libcuflynx.scripts.read_and_insert_parameters || exit 1

"${cuflynx_cmd[@]}" "$@"
