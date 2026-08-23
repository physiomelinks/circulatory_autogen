#!/bin/bash

### Build the generated 0D C++ model and the 1D-0D coupler, then run the coupler.
###
### Usage: ./run_coupler1d0d.bash [generated-cpp-model-dir]
###
### The model directory is the `<generated_model_subdir>_cpp` that CVSCppGenerator wrote -- it
### holds the generated sources, the Makefiles, and the coupler_config.json the coupler reads.
### It is resolved to an absolute path before anything cd's, because this script changes
### directory twice and the config path is used after the second one.
###
### It used to be hardwired to `../../generated_models/<one tutorial model>_cpp`, relative to
### this script. That broke twice over: the model name only ever suited one tutorial, and when
### `src/coupler` became `src/libcuflynx/coupler` the extra path element made `../..` land on
### `src/` instead of the repo root.

set -u

FOLDERcoupler="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DEFAULTcpp="$FOLDERcoupler/../../../generated_models/cvs_model_with_arm_hybrid_cpp"
FOLDERcpp="${1:-$DEFAULTcpp}"

if [[ $# -eq 0 ]]; then
    echo "No model directory given; falling back to $DEFAULTcpp" >&2
    echo "Pass the generated <model>_cpp directory explicitly -- the default only fits one tutorial model." >&2
fi

if [[ ! -d "$FOLDERcpp" ]]; then
    echo "Generated C++ model directory not found: $FOLDERcpp" >&2
    echo "Generate it first (model_type: cpp, couple_to_1d: true), then pass its path to this script." >&2
    exit 1
fi
FOLDERcpp="$( cd -- "$FOLDERcpp" &> /dev/null && pwd )"

FILEconfig="coupler_config.json"

if [[ ! -f "$FOLDERcpp/$FILEconfig" ]]; then
    echo "No $FILEconfig in $FOLDERcpp." >&2
    echo "The notebook/driver writes it next to the generated model before running the coupler." >&2
    exit 1
fi

USE_PETSC=1
# USE_PETSC=0

cd "$FOLDERcpp" || exit 1

if [[ "$USE_PETSC" -eq 1 ]]; then
    make -f MakefilePETSC clean
    make -f MakefilePETSC
else
    make -f Makefile clean
    make -f Makefile
    export LD_LIBRARY_PATH=$(spack location -i sundials)/lib:$LD_LIBRARY_PATH
fi

cd "$FOLDERcoupler" || exit 1

make -f Makefile clean
make -f Makefile clean_pipe
make -f Makefile

echo "*** RUNNING THE COUPLER NOW ***"

./coupler "$FOLDERcpp/$FILEconfig"
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt &
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"
