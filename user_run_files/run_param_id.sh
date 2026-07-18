if [[ $# -eq 0 ]] ; then
    echo 'usage is ./run_param_id.sh num_processors'
    exit 1
fi
source python_path.sh
./run_autogeneration.sh

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
  echo "Autogeneration completed successfully."

  # Pick an MPI launcher that matches the mpi4py runtime. OpenCOR bundles an
  # MPICH-based mpi4py, but the system default `mpiexec` is often Open MPI, whose
  # PMIx launcher aborts MPICH ranks at MPI_Init with
  # "unsupported PMI version PMIx". When mpi4py is MPICH, prefer an MPICH/Hydra
  # launcher so the launcher's PMI matches the runtime.
  MPIEXEC=mpiexec
  mpi_vendor=$(${python_path} -c "from mpi4py import MPI; print(MPI.get_vendor()[0])" 2>/dev/null)
  if [ "${mpi_vendor}" = "MPICH" ]; then
    if command -v mpiexec.mpich >/dev/null 2>&1; then
      MPIEXEC=mpiexec.mpich
    elif command -v mpiexec.hydra >/dev/null 2>&1; then
      MPIEXEC=mpiexec.hydra
    elif mpiexec --version 2>&1 | grep -qiE 'open.?mpi|openrte'; then
      echo "WARNING: mpi4py is MPICH but the default 'mpiexec' is Open MPI; MPI ranks may" >&2
      echo "         abort with 'unsupported PMI version PMIx'. Install MPICH (e.g." >&2
      echo "         'sudo apt-get install mpich') to get a matching mpiexec.hydra." >&2
    fi
  fi

  ${MPIEXEC} -n $1 ${python_path} ../src/scripts/param_id_run_script.py

else
  echo "Error: Autogeneration failed. Aborting."
  exit 1
fi
