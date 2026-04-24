if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_param_id.sh num_processors'
    exit 1
fi
source python_path.sh
./run_autogeneration.sh

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."

  mpiexec -n $1 ${python_path} ../src/scripts/param_id_run_script.py

else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
