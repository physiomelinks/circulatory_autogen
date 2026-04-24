#!/bin/bash
source python_path.sh

echo "Running identifiability analysis with 1 processor"

${python_path} ../src/scripts/identifiability_run_script.py
