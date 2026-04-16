if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_test_sensitivity_analysis.sh num_processors'
    exit 1
fi

source python_path.sh

mpiexec -n "$1" "${python_path}" -m pytest ../tests/test_sensitivity_analysis.py -v
