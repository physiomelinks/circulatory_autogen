# Getting Started

## Prerequisites

- **Python** 3.9 or newer recommended (the package requires Python ≥3.7 per `pyproject.toml`; use the same interpreter for the whole project).
- **Git**, to clone the repository.
- **pip** (usually bundled with Python).
- **MPI** (optional): needed only if you run parameter identification or sensitivity analysis with multiple processes. See [MPI and system libraries](#mpi-and-system-libraries) below.
- **Compiler / SUNDIALS** (optional on Linux): Myokit’s CVODE backend may need `build-essential` and `libsundials-dev` (or your OS equivalent) so extensions can compile when first used.

## Clone the repository

Clone the Circulatory Autogen project from the [GitHub repository](https://github.com/FinbarArgus/circulatory_autogen).

!!! note
    If you have not used Git before: install Git, open a terminal, go to the folder where you want the code, then run:

    - `git clone https://github.com/FinbarArgus/circulatory_autogen`

    To contribute via a fork:

    - `git clone https://github.com/<YourUsername>/circulatory_autogen`

    - `git remote add upstream https://github.com/physiomelinks/circulatory_autogen`

## Directory layout

**`[project_dir]`** is the folder where the repository was cloned, for example:

`[project_dir]: ~/Documents/git_projects/circulatory_autogen`

The rest of this page assumes commands are run from a terminal and that `[project_dir]` is your current directory when installing.

## Install Python libraries from `pyproject.toml`

Use the **same Python** you will use to run scripts and tests (check with `python --version` or `python3 --version`).

**1. Create a virtual environment (recommended)**

=== "Linux / macOS"
    ```
    cd [project_dir]
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows (cmd)"
    ```
    cd [project_dir]
    py -3 -m venv .venv
    .venv\Scripts\activate.bat
    ```

=== "Windows (PowerShell)"
    ```
    cd [project_dir]
    py -3 -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

If you prefer not to use a venv, skip the steps above and use `python -m pip` / `python3 -m pip` for the installs below.

**2. Upgrade pip (recommended)**

```
python -m pip install --upgrade pip
```

**3. Install the project and dependencies**

Dependencies are listed in `pyproject.toml`. Installing the package in editable mode pulls them in automatically.

- **Runtime only** (autogeneration, parameter ID, solvers such as Myokit, etc.):

    ```
    cd [project_dir]
    python -m pip install -e .
    ```

    When you choose `solver: CVODE` for CellML models, the project defaults that to the `CVODE_myokit` backend. Use `CVODE_opencor` explicitly only when you want the OpenCOR backend.

- **With development tools** (pytest, formatters, linters):

    ```
    cd [project_dir]
    python -m pip install -e ".[dev]"
    ```

The authoritative lists are `[project.dependencies]` and `[project.optional-dependencies]` in `pyproject.toml`. Highlights:

- Autogeneration: `pandas`, `pyyaml`, `rdflib`, `libcellml`, `pint`, etc.
- Parameter identification: `mpi4py`, `nevergrad` (CMA-ES), `emcee`, `numdifftools`, and related scientific stack.
- Sensitivity analysis: `SALib`, `seaborn`.
- **Development**: the `dev` extra (e.g. `pytest`, `pytest-mpi`).

**4. Run scripts**

With the venv activated (if you use one), run Python from `[project_dir]` as usual, for example:

```
python src/scripts/<script_name>.py
```

The shell helpers under `user_run_files/*.sh` may still assume a particular interpreter; if a script fails, run the matching Python module or script directly with your venv’s `python`.

For a notebook-oriented walkthrough, see `tutorial/interactive/generate_and_calibrate.ipynb`.

## MPI and system libraries

!!! warning
    **mpi4py** needs an MPI implementation on the machine. On Linux you may need:

    ```
    sudo apt install libopenmpi-dev openmpi-bin
    ```

    On macOS, for example: `brew install openmpi`. On Windows, install [MS MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467) and the SDK, and ensure the MPI `bin` paths are on your `PATH` (see Microsoft’s documentation).

    **Myokit (CVODE)** on Linux often needs a compiler and SUNDIALS headers, for example:

    ```
    sudo apt install build-essential libsundials-dev
    ```

## Expected outcome

You should now have:

- A clone of the repository at `[project_dir]`.
- A Python environment (venv or global) with dependencies installed from `pyproject.toml` via `pip install -e .` or `pip install -e ".[dev]"`.
- The ability to run `python` and import the project after `cd [project_dir]` with your chosen interpreter.

---

## Deprecated: OpenCOR-based setup

This section documents the **older workflow** that used **OpenCOR’s bundled Python** and `python_path.sh`. It is **not required** for the default Myokit-based path described above. Keep it only if you maintain legacy scripts or environments that still call OpenCOR’s interpreter.

### Install OpenCOR (legacy)

Download OpenCOR (e.g. version 0.8.1) from the [OpenCOR downloads page](https://opencor.ws/downloads/index.html). A zip/tarball install in a directory you control (e.g. `~/Desktop`) is typical.

!!! note
    New to OpenCOR? See the [OpenCOR Tutorial (PDF)](https://tutorial-on-cellml-opencor-and-pmr.readthedocs.io/en/latest/_downloads/d271cfcef7e288704c61320e64d77e2d/OpenCOR-Tutorial-v17.pdf).

### Legacy directory names

- **`[OpenCOR_dir]`**: folder where OpenCOR is installed, e.g. `~/Desktop/OpenCOR-0-8-1-Linux/`.

!!! info
    On some HPC systems an OpenCOR tree may already exist (example path used historically: `/hpc/farg967/OpenCOR-0-8-1-Linux/`). Use your site’s path if applicable. See also [running on hpc](running-on-hpc.md).

### Installing packages with OpenCOR’s pip (legacy)

From `[OpenCOR_dir]`, use OpenCOR’s pip interface instead of system Python:

!!! Note
    === "Linux"
        ```
        ./pip install <packagename>
        ```
    === "Mac"
        ```
        ./pythonshell -m pip install <packagename>
        ```
    === "Windows"
        ```
        ./pythonshell.bat -m pip install <packagename>
        ```

To install this project from `pyproject.toml` **into OpenCOR’s Python** (editable):

```
cd [project_dir]
[OpenCOR_dir]/pip install -e .
```

(Use `./pythonshell -m pip` / `./pythonshell.bat -m pip` on Mac/Windows as above.) For development tools: `pip install -e ".[dev]"`.

### `python_path.sh` (legacy)

Shell scripts under `user_run_files/` may read `[project_dir]/user_run_files/python_path.sh`. Set `python_path` to your OpenCOR pythonshell:

!!! Note
    === "Linux and Mac"
        ```
        python_path=<OpenCOR_dir>/pythonshell
        ```
    === "Windows"
        ```
        python_path=C:\<OpenCOR_dir>\pythonshell.bat
        ```
        Use Windows path conventions (`C:\`, backslashes).

This tutorial’s **primary** path no longer depends on this file if you use a normal venv and `python -m pip install -e .`.

### OpenCOR versions before 0.8 and SSL (legacy)

!!! warning
    In **OpenCOR versions before 0.8** you may need:

    ```
    cd [OpenCOR_dir]/python/bin
    ./python -m pip install <packagename>
    ```

!!! warning
    If you see **SSL errors** with OpenCOR before version 0.8 on Linux:

    ```
    cd [OpenCOR_dir]/python/bin
    export LD_LIBRARY_PATH=[OpenCOR_dir]/lib
    ```

    Then retry pip so `libcrypto` can be found.
