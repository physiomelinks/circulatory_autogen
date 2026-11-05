#!/bin/bash
# Identifiability analysis around a previous calibration's best-fit parameters.
# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-identifiability libcuflynx.scripts.identifiability_run_script || exit 1

echo "Running identifiability analysis with 1 processor"

"${cuflynx_cmd[@]}"
