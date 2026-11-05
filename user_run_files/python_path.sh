#_______python paths _________
# Users should modify python_path to the path of their own python
#
# DEPRECATED: pointing python_path at OpenCOR's bundled `pythonshell` is deprecated.
# It will be replaced by a plain `pip install libopencor` into a normal Python
# environment once libOpenCOR is published to PyPI, after which this file (and
# opencor_pythonshell_path.sh) will be removed. Prefer a standard venv +
# `pip install -e .` and point python_path at that interpreter.
#
# Why it matters: OpenCOR ships a dual-ABI mpi4py (both MPI.mpich.*.so and
# MPI.openmpi.*.so). The variant selected at import time can differ from the MPI
# that the system `mpiexec` belongs to, which aborts MPI runs with
# "unsupported PMI version PMIx". Pin it to match your launcher, e.g.
#   export MPI4PY_MPIABI=openmpi   # or: mpich
# A pip-installed mpi4py links the single system MPI and avoids the ambiguity.

# the below for my hpc
# python_path=/hpc/farg967/OpenCOR-0-8-1-Linux/pythonshell
# python_path=/hpc/farg967/OpenCOR-0-8-3-Linux/pythonshell
# python_path=/hpc/bghi639/Software/OpenCOR-0-7-1-Linux/pythonshell
# python_path=/hpc/bghi639/Software/OpenCOR-0-8-1-Linux/pythonshell

# the below for my local
python_path=/home/farg967/software/OpenCOR-0-8-3-Linux/pythonshell
# python_path=/home/bghi639/Software/OpenCOR-0-7-1-Linux/pythonshell
# python_path=/home/bghi639/Software/OpenCOR-0-8-1-Linux/pythonshell
# python_path=/home/bghi639/Software/OpenCOR-0-8-3-Linux/pythonshell

