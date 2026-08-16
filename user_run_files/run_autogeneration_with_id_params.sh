#!/bin/bash
# Regenerate the model using the parameter values a previous calibration fitted.
# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-generate libcuflynx.scripts.script_generate_with_new_architecture || exit 1

"${cuflynx_cmd[@]}" True
