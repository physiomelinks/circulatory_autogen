# Running on HPC

## ABI HPC Standard

Clone the circulatory_autogen repo into a directory of your choice, create a virtual environment and install the project into it (`python3 -m venv .venv && .venv/bin/pip install -e .`, see [Getting Started](getting-started.md)), then set `python_path` in `python_path.sh` to that venv's Python:

`python_path={project_dir}/.venv/bin/python`

!!! warning "Deprecated: the shared OpenCOR pythonshell"
    Earlier versions of this page pointed `python_path` at a pre-installed OpenCOR pythonshell
    on the ABI HPC (`/hpc/farg967/OpenCOR-0-8-3-Linux/pythonshell`). **Do not use it for new
    work.** OpenCOR's bundled interpreter is deprecated and will be replaced by
    `pip install libopencor` into a normal environment; it also ships a dual-ABI `mpi4py` that
    can mismatch the cluster's `mpiexec` and abort every rank with
    `unsupported PMI version PMIx`. A venv whose `mpi4py` is built against the module you load
    below has no such ambiguity.

To run in parallel you need to load MPI **before installing `mpi4py`** and before each run. Do the following from the `{project_dir}/user_run_files` dir:

`. load_mpi.sh`

Then you should be able to run as normal from the `user_run_files` dir (e.g. `./run_param_id.sh <NUM_CORES>` or `./run_sensitivity_analysis.sh <NUM_CORES>`).

## Scratch space: `TMPDIR`

libCUFLynx writes a flattened copy of each CellML model to the system temp
directory before Myokit compiles it, and caches it there between runs. That
directory is `/tmp` by default, which on a compute node is often small,
node-local, purged mid-job, or not writable by you at all.

You do not need `sudo` or a code change — set `TMPDIR` before launching and
everything follows it:

```bash
export TMPDIR="$SCRATCH/tmp"   # or any directory you can write to
mkdir -p "$TMPDIR"
```

Put this in your job script, before the `run_*.sh` call, so every rank inherits
it. Python's `tempfile` reads `TMPDIR` (then `TEMP`, then `TMP`), so this covers
every temp path the engine uses, not only the CellML cache.

Two notes for multi-rank jobs:

- **Node-local scratch is the better choice** where you have it. The cache is a
  pure function of the input model, so every rank flattens the same file to the
  same name; writes are atomic (staged then `os.replace`d), which makes sharing
  one directory safe but pointless work.
- **Do not point `TMPDIR` at a directory that is purged while the job runs.** The
  cache is read on every helper construction, not just the first.

## ABI HPC Extra

Any extra Python libraries go into your own venv with `pip install <packagename>` — the process is the same as on a local machine, see [Getting Started](getting-started.md). Because the venv is yours, nothing has to be shared with or re-installed alongside another user's environment.

!!! warning
    Before installing mpi4py into the venv, make sure you load mpi with

    `module load mpi/mpich-x86_64 && echo "succesfully loaded mpi/mpich-x86_64"`

    `mpi4py` compiles against whichever MPI is loaded at install time, so loading it afterwards is too late.
