if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_test_sensitivity_analysis.sh num_processors'
    exit 1
fi

source opencor_pythonshell_path.sh

mpiexec -n "$1" "${opencor_pythonshell_path}" -m pytest ../tests/test_sensitivity_analysis.py -v
