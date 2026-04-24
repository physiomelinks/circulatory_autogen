if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_param_id.sh num_processors'
    exit 1
fi
source python_path.sh
mpiexec -n $1 ${python_path} ../src/scripts/run_multiple_param_id.py

