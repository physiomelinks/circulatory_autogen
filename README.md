# Overview

[![Tests](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml/badge.svg?branch=master&event=push)](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml)

This project allows the generation and calibration of cellml (and soon to be more) circulatory system models from an array of module/vessel names and connections.

**The repository is `circulatory_autogen`; the package it installs is `libcuflynx`.** They are the
same project under two names: papers, issues and this repository say *circulatory_autogen*, while
PyPI, `pip install` and every `import` say *libcuflynx*. Searching for either name should find you
this page.

> **Note:** Test results and pass percentage are displayed in the [GitHub Actions workflow summary](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml). The badge above shows the overall test status (passing/failing) for `master` of this repository, which is where pull requests are merged. 

# Installing, and what each extra costs

The package is `libcuflynx`. A plain install gives you generation, simulation, calibration,
sensitivity analysis and the built-in (emcee) MCMC:

```
pip install libcuflynx
```

Everything beyond that is an extra, because the optional parts are not small. These are
installed sizes on disk, not wheel sizes — measured by installing into a fresh venv on Linux
and taking `du` of `site-packages`, so they include the shared libraries that show up as
companion directories (`scipy.libs`, `numpy.libs`) and every transitive dependency:

| install | on disk | what it buys |
|---|---|---|
| `libcuflynx` | **≈ 540 MB** | generation, simulation, calibration, Sobol SA, emcee MCMC, Laplace identifiability |
| `libcuflynx[mpi]` | **+5 MB**, and a system MPI toolchain | multi-rank runs under `mpiexec` |
| `libcuflynx[casadi]` | **+221 MB** | `model_type: casadi_python`, `solver: casadi_integrator`, symbolic AD gradients |
| `libcuflynx[uq]` | **+65 MB** | the pyMC sampler (`UQ_options: library: pymc`) |
| `libcuflynx[emulation]` | **+750 MB or more** | training and using surrogate models (`do_emulation` / `use_emulator`) |
| `libcuflynx[cpp]` | +0 MB | `model_type: cpp` — needs a C++ toolchain, which pip cannot install |
| `libcuflynx[all]` | **≈ 1.6 GB** | all of the above |

`[emulation]` is the one to think about before typing: `autoemulate` pulls in **torch**, which
is 734 MB on its own — more than the whole default install — and requires Python >=3.10,<3.13.
On Linux, pip's default torch wheel also drags in the bundled NVIDIA CUDA libraries, which can
take it past 2 GB; install a CPU-only torch first if you do not want them. Because `[all]`
includes `[emulation]`, `[all]` inherits that Python range too.

The per-package numbers below do not add up to the total: the long tail of small transitive
dependencies (packaging, dateutil, six, typing-extensions, cycler, joblib, threadpoolctl, ...)
accounts for the rest.

What makes up the default install: scipy 109 (+27 in `scipy.libs`), pandas 65, statsmodels 49,
scikit-learn 46, numpy 41 (+27 in `numpy.libs`), matplotlib 28 with fontTools 25 / pillow 21,
myokit 13, lxml 12, libcellml 8, kiwisolver 7, nevergrad 5, SALib 5, rdflib 5, libcuflynx
itself 4, pint 3, seaborn 3, and emcee / corner / tqdm / numdifftools at about 1 MB each.

The two that are larger than they look are **statsmodels (49 MB)** and **scikit-learn (46 MB)**.
Neither is optional today: statsmodels supplies the Geweke diagnostic run after every emcee
MCMC chain, and scikit-learn fits the quadratic used by the Laplace identifiability analysis.
Both are on paths a plain install is expected to reach.

**`[mpi]` is an extra, and that is the deliberate part.** `mpi4py` itself is only 5 MB, but it
compiles against a system MPI toolchain (`libopenmpi-dev`, `mpich`) at install time, and that
is the commonest `pip install` failure on macOS and Windows. A serial calibration needs none of
it: `libcuflynx.utilities.mpi_utils` answers rank/size and supplies the one-rank collectives
without ever importing `mpi4py`. Install the toolchain first, then the extra:

```
sudo apt install libopenmpi-dev          # or: brew install open-mpi
pip install "libcuflynx[mpi]"
```

Ask for a model type, solver or analysis whose extra is missing and the error names the extra
to install rather than reporting a bare missing module.

Developing on a checkout instead: `pip install -e ".[dev]"`, which adds the test and lint
tooling plus `mpi4py` (`tests/conftest.py` imports it at module scope, so the suite will not
even collect without it) and `casadi` (whose tests otherwise `importorskip` themselves into
silence).

# Quickstart

```
pip install libcuflynx
```

No import path setup and no checkout are required — the package is importable, and its
commands runnable, from any directory:

```python
from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict
from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper_from_inp_data_dict
from libcuflynx.param_id.paramID import CVS0DParamID

inp = get_default_inp_data_dict(file_prefix, input_param_file, resources_dir)
generate_with_new_architecture(inp_data_dict=inp)     # CSV arrays -> CellML
sim = get_simulation_helper_from_inp_data_dict(inp)   # simulate it
sim.run()
pid = CVS0DParamID.init_from_dict(inp)                # calibrate it
```

Each pipeline stage also has a console command, all configured from
`user_run_files/user_inputs.yaml` and all taking `--help`:

| Command | Stage |
|---|---|
| `cuflynx-generate` | generate a model from the CSV arrays |
| `cuflynx-param-id` | generate + calibrate (`mpiexec -n N` for parallel) |
| `cuflynx-sequential-param-id` | staged calibration — declared, not yet implemented |
| `cuflynx-sensitivity` | Sobol sensitivity analysis |
| `cuflynx-identifiability` | Laplace / profile-likelihood identifiability |
| `cuflynx-train-emulator` | train a surrogate of the obs features |
| `cuflynx-plot` | plot calibration results |

### `CUFLYNX_USER_DIR` — where an installed libcuflynx reads and writes

Every stage needs a directory holding `user_run_files/user_inputs.yaml`, and (unless the
config overrides them) `resources/`, `module_config_user/`, `funcs_user/`,
`generated_models/` and `param_id_output/`. It is `$CUFLYNX_USER_DIR` if set; otherwise the
circulatory_autogen checkout being run from, if this is one — so a clone or a
`pip install -e .` needs no configuration; otherwise the current working directory.

After a plain `pip install libcuflynx` there is no checkout, so either run from your working
directory or point the variable at it:

```bash
export CUFLYNX_USER_DIR=/path/to/my_study
cuflynx-param-id
```

Nothing is ever written inside the installed package.

**Imports without the `libcuflynx.` prefix are deprecated.** `from param_id.paramID import
CVS0DParamID` still works in 0.4.0 and emits a `DeprecationWarning`; the shims are **removed in
0.5.0**. See `CHANGELOG.md` for the migration, including the one for anyone who edited
`funcs_user/*_funcs_user.py` in place.

# Tutorial

Follow the instructions in the tutorial to run the project: https://physiomelinks.github.io/circulatory_autogen/

# AI-generated interactive tutorial

BETA MODE: This AI-generated tutorial can be used to further understand the code base: https://deepwiki.com/FinbarArgus/circulatory_autogen/1-overview

# The `CVODE_opencor` solver requires OpenCOR

Every solver works from a plain install **except** `CVODE_opencor`. That backend needs the
`opencor` Python module, which is supplied by an [OpenCOR](https://opencor.ws) installation
and is not published on PyPI — so it cannot be shipped in a wheel and no `pip install` can
provide it. Asking for it without OpenCOR raises an error naming the alternative rather than
a bare `ModuleNotFoundError`.

- **Use `solver: CVODE_myokit` instead.** It is a drop-in replacement: the same CellML model,
  integrated by CVODE, with no OpenCOR involved. It is what `user_inputs.yaml` ships with,
  and nothing else in the project needs OpenCOR.
- **Or run inside OpenCOR**, whose bundled interpreter provides `opencor`. That route is
  deprecated — see
  [Deprecated: OpenCOR-based setup](https://physiomelinks.github.io/circulatory_autogen/getting-started/#deprecated-opencor-based-setup).

This is expected to be replaced by a plain `pip install libopencor` once
[libOpenCOR](https://opencor.ws/libopencor/) reaches PyPI; the bundled-interpreter route and
the scripts that support it will be removed then.

The handful of tests that exercise this backend are marked `need_opencor`. They have **no
auto-skip**, so without OpenCOR they fail rather than skip — deselect them with
`-m "not need_opencor"`, which is what CI does.

# Releasing

Releases are cut from a `v*` tag and published to PyPI by
[`.github/workflows/release.yml`](.github/workflows/release.yml) using trusted publishing
(OIDC) — no API token is stored anywhere. The procedure, the release-notes checklist, and the
rules that cannot be undone (the tag must match `version` in `pyproject.toml`; a PyPI version
can never be re-uploaded) are in [CONTRIBUTING.md](CONTRIBUTING.md#making-a-release).

# Citing this work

Cite the project as **circulatory_autogen** — that is the name used by the publications, the
repository and any archived DOI. The PyPI distribution `libcuflynx` is the *same* software under
its packaging name, so a paper citing circulatory_autogen and an environment listing
`libcuflynx==0.4.0` refer to one artifact, not two. When it helps reproducibility, record both:
the citation for the project and the exact package version installed, e.g.
"circulatory_autogen (PyPI package `libcuflynx`, version 0.4.0)".

# License
circulatory_autogen is fully open source and distributed under the very permissive Apache License 2.0. See LICENSE for more information.

## Optional third-party backends (not part of circulatory_autogen)

circulatory_autogen is complete and fully open source on its own. Every feature works
without installing any proprietary software.

Separately, the project ships optional *adapters* that let users who already hold a licence
for certain third-party products plug them in. Those products are **not part of
circulatory_autogen**, are **not bundled or installed with it**, are **not covered by the
Apache-2.0 licence above**, and are **not required by any feature**.

- See [Optional third-party backends](https://physiomelinks.github.io/circulatory_autogen/getting-started/) in the tutorial for more info.
