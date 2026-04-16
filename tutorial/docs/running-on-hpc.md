# Running on HPC

## ABI HPC Standard

If running on the ABI HPC, you can use a pre-installed (with all necessary libraries) OpenCOR pythonshell. To do this, clone the circulatory_autogen repo into a directory of your choice, then change the `python_path` in `python_path.sh` to:

`python_path=/hpc/farg967/OpenCOR-0-8-3-Linux/pythonshell`

To run in parallel you need to load MPI. Do the following from the `{project_dir}/user_run_files` dir:

`. load_mpi.sh`

Then you should be able to run as normal from the `user_run_files` dir (e.g. `./run_param_id.sh <NUM_CORES>` or `./run_sensitivity_analysis.sh <NUM_CORES>`).

## ABI HPC Extra

If you need to install specific Python libraries that aren't installed in the above OpenCOR Python version, install a separate OpenCOR and then install the libraries. See [Getting Started](getting-started.md) for how to install Python libraries; the process is the same on the HPC.

!!! warning
    Before installing mpi4py in your separate OpenCOR, make sure you load mpi with
    
    `module load mpi/mpich-x86_64 && echo "succesfully loaded mpi/mpich-x86_64"`
