#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_emulator_training.sh num_processors'
    exit 1
fi

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx cuflynx-train-emulator libcuflynx.scripts.train_emulator_run_script || exit 1

echo "Training an emulator with $1 processors"

# Make sure the model the emulator is trained against is up to date
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
