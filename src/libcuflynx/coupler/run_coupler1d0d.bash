#!/bin/bash

### Build the generated 0D C++ model and the 1D-0D coupler, then run the coupler.
###
### Usage: ./run_coupler1d0d.bash [generated-cpp-model-dir]
###
### The model directory is the `<generated_model_subdir>_cpp` that the C++ generator wrote -- it
### holds the generated sources, their CMakeLists.txt, and the coupler_config.json the coupler
### reads. It is resolved to an absolute path before anything cd's.
###
### Both are built with CMake:
###   - the model in <model-dir>/build; main0d is also copied to <model-dir>/main0d, the path
###     coupler_config.json's solver0d_path usually points at;
###   - the coupler in <coupler-dir>/build (nlohmann/json is downloaded if not installed).
###
### Environment:
###   SUNDIALS_DIR   SUNDIALS install prefix, for models generated with the CVODE solver, when
###                  CMake can't find it on its own (e.g. a from-source SUNDIALS 7 build).
###   CMAKE_ARGS     extra arguments passed to both CMake configure steps.

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
    echo "The C++ generator writes it when the model is coupled to 1D (couple_to_1d): regenerate the model." >&2
    exit 1
fi
if [[ ! -f "$FOLDERcpp/CMakeLists.txt" ]]; then
    echo "No CMakeLists.txt in $FOLDERcpp: regenerate the model with this version of libcuflynx." >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "${SUNDIALS_DIR:-}" ]]; then
    EXTRA_ARGS+=("-DSUNDIALS_DIR=$SUNDIALS_DIR")
fi
# shellcheck disable=SC2206
EXTRA_ARGS+=(${CMAKE_ARGS:-})

echo "*** BUILDING THE 0D MODEL ***"
cmake -S "$FOLDERcpp" -B "$FOLDERcpp/build" "${EXTRA_ARGS[@]}" || exit 1
cmake --build "$FOLDERcpp/build" -j || exit 1
cp "$FOLDERcpp/build/main0d" "$FOLDERcpp/main0d" || exit 1

echo "*** BUILDING THE COUPLER ***"
cmake -S "$FOLDERcoupler" -B "$FOLDERcoupler/build" "${EXTRA_ARGS[@]}" || exit 1
cmake --build "$FOLDERcoupler/build" -j || exit 1

# Stale FIFOs from an interrupted run would be reused by the next one.
PIPE_DIR=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('tmp_pipe_path',''))" "$FOLDERcpp/$FILEconfig" 2>/dev/null)
if [[ -n "$PIPE_DIR" ]]; then
    mkdir -p "$PIPE_DIR" || exit 1
    rm -f "$PIPE_DIR"/zero_to_* "$PIPE_DIR"/one_to_* "$PIPE_DIR"/parent_to_*
fi

echo "*** RUNNING THE COUPLER NOW ***"

"$FOLDERcoupler/build/coupler" "$FOLDERcpp/$FILEconfig" || exit 1

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"
