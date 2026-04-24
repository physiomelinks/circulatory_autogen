#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_sensitivity_analysis.sh num_processors'
    exit 1
fi

# Source the path
source python_path.sh

echo "Running sensitivity analysis with $1 processors"

# Run the autogeneration script
./run_autogeneration.sh

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."
  
  # If successful, proceed with the mpirun command
  mpiexec -n "$1" "${python_path}" ../src/scripts/sensitivity_analysis_run_script.py
  
else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
