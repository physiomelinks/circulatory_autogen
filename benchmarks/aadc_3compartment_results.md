# AADC 3compartment Benchmark Results

Date: 2026-07-23
Branch: feat/aadc-algebraic-obs

## Fair Comparison (same problem, same optimizer)

3compartment cardiovascular model, 27 states, 4 params, pre_time=20s, sim_time=2s.
Optimizer: multi-start L-BFGS-B, 1 start, seed=0, 200 max calls.

### Results

| Method | Solver | Gradient | Cost | Time | Notes |
|--------|--------|----------|------|------|-------|
| **CasADi BDF** | CasADi symbolic BDF2, dt_internal=0.001 | CasADi reverse AD | **0.0228** | **3.5 min** | Reference |
| **AADC bdf_newton** | Newton BDF1 + VFJ, dt_internal=0.001 | AADC IFT | **0.0797** | **10 min** | Our approach |
| bdf_newton + FD | Newton BDF1 + VFJ, dt_internal=0.001 | Finite differences | 0.161 | 25 min | Same solver, FD |

### Analysis

- **AADC vs FD (same solver)**: AADC **2.5× faster**, **2× better cost**. Exact gradient → better convergence.
- **CasADi vs AADC**: CasADi **2.8× faster**, **3.5× better cost**.
  - BDF2 (2nd order) vs BDF1 (1st order) → CasADi more accurate at same dt
  - CasADi symbolic `mapaccum` vs Python loop → no Python overhead per step
  - CasADi rootfinder optimized, not generic Newton

### Key technical achievements

1. **Forward solve correct**: AADC BDF v[22]=0.000160 matches CVODE (verified)
2. **Gradient correct**: AD/FD = 1.000 for all 4 params (verified independently)
3. **`_aadc_passive` fix**: `math.floor(_aadc_passive(x))` → `aadc.math.floor(x)` in code generator — fixes tape replay for models with cardiac valve logic
4. **VFJ.func optimization**: 35× faster than Python compute_rates → forward from 50s to 9.3s
5. **IFT sensitivity only during sim_time**: skip warmup → saves 91% of dfdp evaluations
6. **Architecture**: one small kernel (compute_rates) replayed per step — no huge tape

### FHN Benchmark (non-stiff, for comparison)

| Method | Cost | Time |
|--------|------|------|
| FD | 2.66e-03 | 50s |
| CasADi AD | 5.33e-05 | 44s |
| **AADC AD** | **6.06e-06** | **2.4s** |

AADC **16-20× faster** than both FD and CasADi on non-stiff models.

### Next steps to close the gap with CasADi on stiff models

1. **BDF2**: implement 2nd order BDF (x_{n+1} = 4/3 x_n - 1/3 x_{n-1} - 2/3 dt f(x_{n+1}))
2. **Batch VFJ**: vectorize kernel replay across sub-steps
3. **C++ kernel**: move Newton loop to C++ to eliminate Python overhead
