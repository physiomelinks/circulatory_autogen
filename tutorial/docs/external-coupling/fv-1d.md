# 1D finite-volume coupling

Some vessels of a model can be simulated with the finite-volume 1D blood-flow solver shipped with
libcuflynx (`libcuflynx/solver1d`), while the rest stay 0D modules in generated C++. The
two run as separate programs. The **coupler** (`libcuflynx/coupler`) launches both and relays
their messages over named pipes, and they exchange flow and pressure at every 0D right-hand-side
evaluation.

## The modules

| Module (`coupling_modules_config.json`) | `api` | What it is |
|---|---|---|
| `FV1D_vessel` (`vp`, `pv`, `vv`, `nn`) | `role: consumer`, `transport: named_pipe`, `process: FV1D_solver` | a 1D vessel. Its `calls` say what each 0D end sends and receives, and when |
| `FV1D_volume_sum` | `role: consumer`, `process: FV1D_solver` | the blood volume held in the 1D vessels, received each step |
| `FV1D_solver` | `role: process`, `coordinator: coupler` | the 1D solver itself: what to launch (`program`) and the pipes (`channels`, `message_length`) |

The pipe names are written down once, in `FV1D_solver`; the consumers name it. The coupler
(C++) and the 1D solver (Python) use the same names, and
`tests/test_cpp_template_generator.py` checks that they still match.

The api roles in brief:
- **consumer:** each entry of `calls` has a `when` (`init`, `step_start`, `rhs_start`, `rhs` or `step_end`).
  - It `send`s some of: `$voi`, `$dt`, `port.flow`, `port.pressure`, `control.<port_type>|default`, or numbers.
  - It `recv`s some of: `port.input`, a model variable, `$dt` or `$ignore`.
- **process:** `program` is what to launch, `{"interpreter": "python", "package": "libcuflynx.solver1d", "script": "main1D.py"}` (or a `path` instead of a `package`). `coordinator: coupler` names who launches it.

## Workflow

**1. Start from a 0D model** whose vessels you want in 1D. For example, `resources/aortic_bif_0d`
is an aortic bifurcation of `split_junction` and `arterial` vessels with nonlinear tube laws.

**2. Convert vessels to 1D.**

```python
from libcuflynx.scripts.convert_0d_to_1d import convert_0d_to_1d
convert_0d_to_1d('aortic_bif', 'resources', 'aortic_bif_0d_parameters.csv', 'resources',
                 ['parent', 'daughter_1', 'daughter_2'])
```

This writes `aortic_bif_hybrid_module_array.csv` and `aortic_bif_hybrid_parameters.csv`:
- the listed vessels become `FV1D_vessel` rows, and their `K_tube_*` rows go;
- an `FV1D_solver` row is added;
- the boundary-condition parameters of the 0D modules now fed by the 1D vessels are marked
  `WONT_BE_USED`.

**3. Generate.**

```yaml
file_prefix: aortic_bif_hybrid
model_type: cpp
couple_to_1d: true
generate_1d: true
solver_1d_type: py
cpp_generated_models_dir: generated_models/aortic_bif_hybrid_cpp
cpp_1d_model_config_path: generated_models/aortic_bif_hybrid_1d/run000/input.ini
coupler_pipe_dir: /tmp/my_pipes        # optional
pre_time: 0.0
sim_time: 3.3                          # 3 periods of T = 1.1 s
dt: 0.01
solver_info: {solver: CVODE, dt_solver: 0.0001}
```

Generation writes:
- the 0D C++;
- `aortic_bif_hybrid_coupler1d0d.json` (which 0D port meets which 1D vessel end);
- the 1D solver's input files;
- `coupler_config.json`.

**4. Run.**

```bash
src/libcuflynx/coupler/run_coupler1d0d.bash generated_models/aortic_bif_hybrid_cpp   # builds both, runs
# or, with main0d and the coupler already built:
<coupler build>/coupler generated_models/aortic_bif_hybrid_cpp/coupler_config.json
```

## `coupler_config.json`

| Key | Value |
|---|---|
| `T0` | the global parameter `T` (seconds): the heart period, or the inflow period of an open-loop model. The 1D model's `input.ini` takes it from the same parameter. |
| `nCC` | the number of whole periods covering `pre_time + sim_time`. The coupled run ends at `nCC·T0`; `main0d` and the 1D solver save from `(nCC − 2)·T0`. |
| `tmp_pipe_path` | user input `coupler_pipe_dir`, default `cuflynx_pipes/<model>/` in the system temp folder (`$TMPDIR` when set). The coupler creates it. |
| `python_path` | the Python that ran the generation |
| `solver1d_path` | the `program` of `FV1D_solver` (the installed `main1D.py`) |
| `solver0d_path` | `<cpp_generated_models_dir>/build/main0d`, as built with CMake |
| `initFile_sim1d_path` | user input `cpp_1d_model_config_path` |
| `inputFold`, `networkName`, `ODEsolver` | the generated model's folder, name and C++ solver |

## How close is the hybrid model to the 0D one?

`tests/test_cpp_template_generator.py` (`test_coupled_fv1d_model_stays_close_to_the_0d_model`)
runs `aortic_bif_0d` and its hybrid version through the coupler. It compares them over the last
of three periods:
- mean terminal pressures and flows are equal (mass is conserved);
- pulse amplitudes agree to 0.4 %;
- the largest pointwise difference is 3.6 % of the cycle range, from wave travel in the 1D vessels.
