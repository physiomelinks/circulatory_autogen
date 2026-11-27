#!/bin/bash
# Generate the model, then calibrate it against the observables in user_inputs.yaml.
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_param_id.sh num_processors'
    exit 1
fi

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-param-id libcuflynx.scripts.param_id_run_script || exit 1

# cuflynx-generate exits non-zero when generation or the generated model's validation fails:
# there is then no model to calibrate.
if ! "${_cuflynx_dir}/run_autogeneration.sh"; then
  echo "Error: model generation failed (see the messages above). Not calibrating."
  exit 1
fi
echo "Autogeneration completed successfully."

mpiexec -n "$1" "${cuflynx_cmd[@]}"
