# Overview

[![Tests](https://github.com/FinbarArgus/circulatory_autogen/workflows/Tests/badge.svg)](https://github.com/FinbarArgus/circulatory_autogen/actions/workflows/tests.yml)

This project allows the generation and calibration of cellml (and soon to be more) circulatory system models from an array of module/vessel names and connections.

> **Note:** Test results and pass percentage are displayed in the [GitHub Actions workflow summary](https://github.com/FinbarArgus/circulatory_autogen/actions/workflows/tests.yml). The badge above shows the overall test status (passing/failing). 

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

- **AADC (Matlogica)** — proprietary, academic/non-commercial use only, and **not open
  source**. It is an *optional* alternative automatic-differentiation backend
  (`model_type: aadc_python`). You must obtain a licence directly from Matlogica and accept
  their terms; the circulatory_autogen maintainers neither supply nor support AADC licences.
  **The default, supported, fully open-source AD backend is CasADi** (`model_type:
  casadi_python`) — you never need AADC. See
  [Optional third-party backends](https://physiomelinks.github.io/circulatory_autogen/getting-started/)
  in the tutorial.
