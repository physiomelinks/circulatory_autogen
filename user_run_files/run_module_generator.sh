#!/bin/bash
# Convert a hand-written CellML model into a libcuflynx module: writes
# <prefix>_module_config.json and <prefix>_modules.cellml into module_config_user/, and a
# resources/ + user_inputs.yaml pair into OUTPUT_DIR.
#
# Every setting below is a variable you are expected to edit, or to override in the
# environment: `INPUT_MODEL=... OUTPUT_DIR=... ./run_module_generator.sh`. They used to be
# one contributor's absolute paths, which meant a fresh clone ran the generator against a
# directory that exists on exactly one machine (the same leak commit 2de2a6f removed
# elsewhere), and the script sourced an opencor_pythonshell_path.sh that is in neither tree
# -- so the interpreter it invoked expanded to the empty string.
INPUT_MODEL="${INPUT_MODEL:?set INPUT_MODEL to the .cellml model to convert}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the directory the generated files go in}"
FILE_PREFIX="${FILE_PREFIX:-my_module}"
VESSEL_NAME="${VESSEL_NAME:-${FILE_PREFIX}}"
DATA_REFERENCE="${DATA_REFERENCE:-unknown}"
TIME_VARIABLE="${TIME_VARIABLE:-t}"
COMPONENT_NAME="${COMPONENT_NAME:-main}"

# This script's own directory, using builtins only: dirname needs a working PATH, and a
# broken PATH is precisely the situation the helper below exists to report on.
_cuflynx_dir="${BASH_SOURCE[0]%/*}"
[ "${_cuflynx_dir}" = "${BASH_SOURCE[0]}" ] && _cuflynx_dir="."
source "${_cuflynx_dir}/cuflynx_entry_point.sh"
require_cuflynx_module libcuflynx.scripts.generate_modules_files || exit 1

"${cuflynx_cmd[@]}" \
    -i "${INPUT_MODEL}" \
    -o "${OUTPUT_DIR}" \
    --file-prefix "${FILE_PREFIX}" \
    --vessel-name "${VESSEL_NAME}" \
    --data-reference "${DATA_REFERENCE}" \
    --time-variable "${TIME_VARIABLE}" \
    --component-name "${COMPONENT_NAME}"
