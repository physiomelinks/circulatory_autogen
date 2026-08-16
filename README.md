# Overview

[![Tests](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml/badge.svg?branch=master&event=push)](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml)

This project allows the generation and calibration of cellml (and soon to be more) circulatory system models from an array of module/vessel names and connections.

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
| `libcuflynx` | **≈ 520 MB** | generation, simulation, calibration, Sobol SA, emcee MCMC, Laplace identifiability |
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

# Tutorial

Follow the instructions in the tutorial to run the project: https://physiomelinks.github.io/circulatory_autogen/

# AI-generated interactive tutorial

BETA MODE: This AI-generated tutorial can be used to further understand the code base: https://deepwiki.com/FinbarArgus/circulatory_autogen/1-overview

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
