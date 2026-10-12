# Couple your own Python model

This page couples a Python model to a generated C++ 0D model. The example is the one
libcuflynx tests with, in `tests/data/external_coupling/`. Two capillaries (`capillary` +
`GE_capillary`) exchange O2 with tissue, and the tissue is a numpy class with one well-mixed
volume per capillary. The same steps apply to a FEniCS model; see
[tissue O2 in FEniCS](fenics-tissue-o2.md).

## 1. Write the class

```python
import numpy as np
from libcuflynx.coupling import ExternalModel   # optional: any class with these methods works


class WellMixedTissue(ExternalModel):
    def setup(self):                           # called once; self.params, self.neighbours, self.comm, self.info
        n = len(self.neighbours['C_t'])        # one value per connected capillary
        self.C = np.full(n, self.params['C_init'])

    def initial_outputs(self):                 # what it sets at t = 0
        return {'C_t': self.C}

    def step(self, t, dt, inputs):             # advance t -> t + dt with the 0D values at t
        J = inputs['J_c']                      # O2 flux from each capillary [mol/s]
        V, M, k = self.params['V'], self.params['M'], self.params['k_reduce']
        self.C = self.C + dt * (J / V + M * (1 - np.exp(-k * self.C)))   # (the test uses RK4 substeps)
        return {'C_t': self.C}
```

What the runner passes:

| Argument | What it is |
|---|---|
| `params` | the module's constants and global constants, with values from the parameters file (`V_tissue`, `M_tissue`, … for a row named `tissue`) |
| `neighbours` | `{port variable: [connected 0D modules]}`, so `self.neighbours['C_t'] == ['capillary_GE_0', 'capillary_GE_1']` |
| `comm` | the MPI communicator, or `None` |
| `info` | `units`, `directions`, `coupling_dt`, `output_dir`, `row` |

`step` receives every variable it reads and returns every variable it sets, as numpy arrays,
one value per connected module. Optional methods:
- `write(t)`, called at every output time;
- `close()`;
- `snapshot()` and `restore()`, needed only for `subiterations`.

## 2. Describe it as a module

Put a module config next to the file: `well_mixed_tissue_module_config.json`.

```json
[{
  "vessel_type": "well_mixed_tissue", "BC_type": "nn",
  "module_format": "external_api", "module_file": "", "module_type": "well_mixed_tissue",
  "entrance_ports": [{"port_type": "capillary_to_flux_port", "variables": ["C_t", "J_c"]}],
  "exit_ports": [], "general_ports": [],
  "variables_and_units": [
    ["C_t", "mol_per_m3", "access", "variable"],
    ["J_c", "mol_per_s", "access", "variable"],
    ["V", "m3", "access", "constant"],
    ["M", "millimolar_per_s", "access", "constant"],
    ["k_reduce", "per_millimolar", "access", "constant"],
    ["C_init", "millimolar", "access", "constant"]
  ],
  "api": {
    "name": "well_mixed_tissue", "role": "provider", "transport": "python",
    "python": {"file": "well_mixed_tissue.py", "class": "WellMixedTissue"},
    "coupling_dt": 0.01
  }
}]
```

The **port** decides what is exchanged. `GE_capillary` has an exit port
`capillary_to_flux_port [ub_O2_t, flux_O2_c]`, so this module's entrance port of the same type
pairs `C_t` with `ub_O2_t` and `J_c` with `flux_O2_c`. Matching an existing CellML module's port
(here the tissue cell's: `tissue_diffusion` in the built-in modules,
`tissue_diffusion_volume` in the module library) means a model can swap between the CellML module and yours by
changing one row. The **directions** come from the CellML side:
- `ub_O2_t` is a boundary condition of `GE_capillary`, so the class sets `C_t`;
- `flux_O2_c` is computed, so the class receives `J_c`.

## 3. Add a row to the module array

```
name,BC_type,vessel_type,inp_vessels,out_vessels
cap_0,pp,capillary,,capillary_GE_0
cap_1,pp,capillary,,capillary_GE_1
capillary_GE_0,nn,GE_capillary,cap_0,tissue
capillary_GE_1,nn,GE_capillary,cap_1,tissue
tissue,nn,well_mixed_tissue,capillary_GE_0 capillary_GE_1,
```

The parameters file needs:
- the external module's constants as `<name>_<row>` (`V_tissue`, `M_tissue`, `k_reduce_tissue`,
  `C_init_tissue`);
- the starting values of the boundary conditions the class sets (`ub_O2_t_capillary_GE_0`, …).

Generation lists anything missing in `<prefix>_parameters_unfinished.csv`, as usual.

## 4. Generate as C++

In `user_inputs.yaml`, use the folder holding the module config as an `external_modules_dir`
(or a `module_library_dirs` entry):

```yaml
file_prefix: microvasc_O2_ext
model_type: cpp
external_modules_dir: [/path/to/my_modules]
pre_time: 0.0
sim_time: 20.0
dt: 0.1                          # output step
solver_info: {solver: CVODE, dt_solver: 0.001}
```

```bash
cuflynx-generate --user-inputs user_inputs.yaml
```

Besides the usual C++ (see [Model Generation](../model-generation-simulation.md)), the model
folder gets:
- `model0d_capi.cpp`, a C interface built as the shared library `model0d_capi`;
- `external_models.json`, which lists what runs and what it exchanges.

## 5. Run it

```bash
cuflynx-couple generated_models/microvasc_O2_ext --check   # what is exchanged, in which direction
cuflynx-couple generated_models/microvasc_O2_ext           # build if needed, then run
mpiexec -n 4 cuflynx-couple generated_models/microvasc_O2_ext   # an MPI-parallel external model
```

Or from Python:

```python
from libcuflynx.coupling import run_coupled
result = run_coupled('generated_models/microvasc_O2_ext', record=['capillary_GE_0/ub_O2_c'])
result.times, result.exchange['tissue/C_t'], result.records, result.timings
```

What a run produces:
- the 0D model writes its usual `sol0D_states.txt` / `sol0D_variables.txt` every `dt` from `pre_time` on;
- `coupling_exchange.csv` holds the exchanged values at the same times;
- `result.timings` splits the wall-clock time between the 0D model, the external model and setup.

**Building:** the shared library is built with CMake on first use, and again whenever the
generated sources change. In a conda environment, have `cmake cxx-compiler sundials` installed
there, so the library uses the same C++ runtime as the Python that loads it (see
[Troubleshooting](troubleshooting.md)).

**MPI:** the 0D model runs on every rank. The class gets the communicator and must return the
same values on every rank; reduce with `comm.allreduce`, as the FEniCS examples do. Only
rank 0 writes the 0D files.

## Checking it against CellML

When an all-CellML version of the external side exists, compare the two. The test
`tests/test_external_coupling.py` runs this example against the same model with a CellML
well-mixed volume per capillary (`well_mixed_volume`, the same equations, kept with the tests), and they agree to 0.06 % at a coupling step of 0.01 s. The FEniCS
examples do the same against finite-volume CellML grids.
