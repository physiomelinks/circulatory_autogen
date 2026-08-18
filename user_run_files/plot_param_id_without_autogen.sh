#!/bin/bash
# Plot the results of a calibration without regenerating the model first (that is the
# `False`). Configuration is in user_inputs.yaml.
# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-plot libcuflynx.scripts.plot_param_id_script || exit 1

"${cuflynx_cmd[@]}" False
