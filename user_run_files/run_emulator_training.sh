#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_emulator_training.sh num_processors'
    exit 1
fi

# Source the path
source python_path.sh

echo "Training an emulator with $1 processors"

# Make sure the model the emulator is trained against is up to date
./run_autogeneration.sh

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."

  # If successful, proceed with the mpirun command
  mpiexec -n "$1" "${python_path}" ../src/scripts/train_emulator_run_script.py

else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
