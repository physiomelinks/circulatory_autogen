#!/bin/bash 

# spack load sundials
make clean
make

# export LD_LIBRARY_PATH=$(spack location -i sundials)/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(spack location -i sundials~mpi)/lib:$LD_LIBRARY_PATH

./cvs_model_with_arm_0d
# ./cvs_model_with_arm_0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./cvs_model_with_arm_0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt & 
# ./cvs_model_with_arm_0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"
