# Norepinephrine around a sympathetic varicosity, in FEniCS

A sympathetic neuron releases norepinephrine (NE) into the extracellular space at its varicosity
with each action potential. Here the neuron is generated C++ from CellML modules:
- a periodic current stimulus;
- the soma;
- the axon;
- the varicosity.

The NE diffusing around the varicosity is the same FEniCS model as in the
[tissue O2 example](fenics-tissue-o2.md), with another instance of parameters. It is also
compared with a finite-volume grid of CellML cells.

The files are in the module library
[circulatory-autogen-modules](https://github.com/physiomelinks/circulatory-autogen-modules):

| Path | What it is |
|---|---|
| `modules/cell/neuron/varicosity/versions/NEexchange_v01/` | the varicosity, with its NE exchanged with the outside |
| `modules/transport/tissue_diffusion_FEniCS/versions/box_v01/` | the FEniCS model; instance `NE_extracellular` |
| `system_models/coupled/SN_NE_FEniCS/` | neuron + FEniCS extracellular space |
| `system_models/coupled/SN_NE_FV/` | neuron + a CellML grid of `tissue_diffusion_volume` cells |
| `system_models/coupled/SN_NE_single_volume/`, `SN_NE_original/` | the check that the exchange is right (below) |

## Opening the varicosity to the outside

In the original varicosity (`sympathetic_monolithic_v01`), NE is an internal state: release
driven by Ca above a threshold, and reuptake by the NE transporter (NET):

```
dNE/dt = k_NE max(Cai − Cai_NE_base, 0) − k_NET (NE − NE_init)
```

`NEexchange_v01` has the same Ca and membrane equations, but sends the **net flux** out and
receives the **extracellular NE at the varicosity**, `NE_ext`, back:

```
J_NE = Vol (k_NE max(Cai − Cai_NE_base, 0) − k_NET (NE_ext − NE_init))      [mol/s]
```

It does this through `capillary_to_flux_port [NE_ext, J_NE]`, the port of the finite-volume cell
`tissue_diffusion_volume` and of the FEniCS model. Connected to one well-mixed volume equal to
`Vol`, it is exactly the original varicosity. The system model `SN_NE_single_volume` checks this
against `SN_NE_original`: NE agrees to 2e-7 of its peak, the solver tolerance.

## The FEniCS side

The `NE_extracellular` instance of `tissue_diffusion_FEniCS` is a 5 µm cube of 1 µm cells, each
the varicosity's volume, around the varicosity. Its settings:
- an effective diffusivity of 3e-10 m²/s (a small monoamine's free diffusivity divided by the
  squared tortuosity of the extracellular space);
- no clearance in the box (NET in the varicosity takes NE back up);
- a coupling step of 0.1 ms, set by the instance parameter `coupling_dt` (action potentials last
  about 1 ms).

The system models use a 3 µm cube. Their parameters file sets the module's `Lx_tissue`, `grid_nx_tissue`
and so on, as for any module.

The neuron row lists the FEniCS row in its outputs. The varicosity's `NE_ext` is a boundary
condition and `J_NE` is computed, so the FEniCS model receives `J_s` and sets `C_t`. Nothing
else is configured.

## FEniCS against the CellML grid

`tests/test_coupled_systems.py` runs both over 0.2 s of a 20 Hz train, four action potentials:

| Extracellular model | NE at the varicosity: largest difference (share of peak) | NE peak | Run time |
|---|---|---|---|
| CellML grid (27 cells, 54 faces), C++ through `main0d` | (reference) | 3.142e-4 mM | 0.38 s, after 33 s of generation and 3 s of build |
| FEniCS, DG0 (the same scheme), coupled with `subiterations` | 1.1 % | 3.141e-4 mM (−0.02 %) | 5.1 s (0D 1.2 s, FEniCS 0.8 s) |
| FEniCS, Q1 elements refined twice, coupled | 28 % | 2.76e-4 mM | 4.6 s (0D 0.4 s, FEniCS 0.9 s) |

- **DG0** is the same discrete model as the CellML grid, and the two agree to the coupling and
  time stepping. NE is released in pulses of about 1 ms, a few coupling steps each, so the
  comparison iterates each step (`subiterations: 3`); the explicit default, first order, roughly
  doubles the difference.
- **Q1** differs because the 1 µm CellML cells can't resolve the steep gradient around a release
  this size. Refining the elements from 2 to 4 per cell moved the FEniCS peak by 2 % (measured before
  the varicosity had a PMCA). The coarse CellML grid's peak is 14 % higher, so here the FEniCS
  model is the better-resolved of the two.
- **Time:** on this grid the CellML model runs faster but takes half a minute to generate. Generation
  grows quickly with the number of cells, which the FEniCS model avoids.

## Running it

As for the [tissue O2 example](fenics-tissue-o2.md#running-it), with `file_prefix: SN_NE_FEniCS`,
`sim_time: 0.2`, `dt: 1.0e-4` and `solver_info: {solver: CVODE, dt_solver: 1.0e-4, rtol: 1.0e-8, atol: 1.0e-12}`.
