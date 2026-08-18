#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_sensitivity_analysis.sh num_processors'
    exit 1
fi

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-sensitivity libcuflynx.scripts.sensitivity_analysis_run_script || exit 1

echo "Running sensitivity analysis with $1 processors"

# Run the autogeneration script
"${_cuflynx_dir}/run_autogeneration.sh"

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."

  # If successful, proceed with the mpirun command
  mpiexec -n "$1" "${cuflynx_cmd[@]}"

else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
