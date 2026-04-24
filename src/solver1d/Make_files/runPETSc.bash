#!/bin/bash 

make -f MakefilePETSC clean
make -f MakefilePETSC

mpirun -np 1 ./cvs_model_with_arm_0d -log_view
# mpirun -np 1 ./cvs_model_with_arm_0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# mpirun -np 1 ./cvs_model_with_arm_0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt & 
# mpirun -np 1 ./cvs_model_with_arm_0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"
