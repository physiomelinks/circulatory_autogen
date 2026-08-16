# Overview

[![Tests](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml/badge.svg?branch=master&event=push)](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml)

This project allows the generation and calibration of cellml (and soon to be more) circulatory system models from an array of module/vessel names and connections.

> **Note:** Test results and pass percentage are displayed in the [GitHub Actions workflow summary](https://github.com/physiomelinks/circulatory_autogen/actions/workflows/tests.yml). The badge above shows the overall test status (passing/failing) for `master` of this repository, which is where pull requests are merged. 

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
