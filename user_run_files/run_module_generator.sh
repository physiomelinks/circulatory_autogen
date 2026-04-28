
# Load OpenCOR Python shell path
source opencor_pythonshell_path.sh

# Set variables for arguments
INPUT_MODEL="/home/hsma807/Animus/RA/ANS-BGCVS/Code/CA_user_ANS_BGCVS/cellML Models/BloodVolumeControl.cellml"
OUTPUT_DIR="/home/hsma807/Animus/RA/ANS-BGCVS/Code/CA_user_ANS_BGCVS"
FILE_PREFIX="BG_volume_control"
VESSEL_NAME="blood_volume_control"
DATA_REFERENCE="Peter_Hunter"
TIME_VARIABLE="t"
COMPONENT_NAME="main"

# Run the script with all arguments
${opencor_pythonshell_path} ../src/scripts/generate_modules_files.py \
    -i "${INPUT_MODEL}" \
    -o "${OUTPUT_DIR}" \
    --file-prefix "${FILE_PREFIX}" \
    --vessel-name "${VESSEL_NAME}" \
    --data-reference "${DATA_REFERENCE}" \
    --time-variable "${TIME_VARIABLE}" \
    --component-name "${COMPONENT_NAME}"