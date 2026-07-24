# AADC 3compartment Benchmark Results

Date: 2026-07-24 (updated)
Branch: feat/aadc-algebraic-obs

## Fair Comparison (same problem, same optimizer)

3compartment cardiovascular model, 27 states, 4 params, pre_time=20s, sim_time=2s.
Optimizer: L-BFGS-B, 1 start, seed=0, 200 max calls.

### Results (2026-07-24, with all gradient fixes)

| Method | Solver | Gradient | Cost | Time | Notes |
|--------|--------|----------|------|------|-------|
| **CasADi BDF** | CasADi symbolic BDF2, dt=0.001 | CasADi reverse AD | **0.0228** | **3.5 min** | Reference |
| **AADC bdf_newton v2** | Newton BDF2 + VFJ, dt=0.001 | AADC IFT (full) | **0.0506** | **34 min** | Correct gradient, all 6 obs |
| AADC bdf_newton v1 | Newton BDF1 + VFJ, dt=0.001 | AADC IFT (sim only) | 0.0797 | 10 min | Wrong gradient (warmup skipped) |
| bdf_newton + FD | Newton BDF1 + VFJ, dt=0.001 | Finite differences | 0.161 | 25 min | Same solver, FD |

### Gradient verification (2026-07-24)

All 4 parameters verified at nominal values with central FD (h=1e-6):

| Param | Name | AD/FD ratio |
|-------|------|-------------|
| 0 | q_lv_init | **1.0000** |
| 1 | C (aortic) | **1.0006** |
| 2 | E_lv_A | **0.9995** |
| 3 | E_lv_B | **0.9991** |

### Bugs found and fixed (2026-07-24)

1. **Cost=0 (indentation bug)**: operation/cost block was inside `else: continue` — dead code after `continue`. Fix: de-indent one level.

2. **Warmup sensitivity skipped**: `S` was reset to zero at start of sim_time, ignoring that warmup changes the steady state in a parameter-dependent way. Fix: compute IFT sensitivity from step 0.

3. **BDF2 sensitivity formula wrong**: used `S_n` instead of `(4/3)*S_n - (1/3)*S_{n-1}`. Fix: track `S_prev` and use correct BDF2 IFT formula.

4. **Algebraic variable gradient missing**: var observables had `S_series = zeros`. Fix: chain rule `du/dp = (du/dx)*S + du/dp_direct` with FD of `compute_variables`.

5. **Initial state param gradient = 0**: `q_lv_init` sets state `q_lv` at t=0, but `S(0) = 0`. Fix: initialize `S[state_idx, param_idx] = 1` for `*_init` parameters.

### Analysis

- **v2 vs v1**: correct gradient → better convergence (cost 0.0506 vs 0.0797), but 3.4× slower (warmup sensitivity = 20000 extra dfdp evals)
- **CasADi vs AADC v2**: CasADi **2.2× better cost**, **10× faster**
  - BDF2 forward accuracy identical (both use BDF2 now)
  - Speed gap: CasADi symbolic `mapaccum` vs Python loop + AADC evaluate per step
  - Cost gap: CasADi gradient is exact (symbolic reverse AD), AADC gradient has small FD error in var obs chain rule

### Key technical achievements

1. **Forward solve correct**: AADC BDF v[22]=0.000160 matches CVODE (verified)
2. **Gradient correct**: AD/FD = 1.000 for all 4 params (verified independently)
3. **`_aadc_passive` fix**: `math.floor(_aadc_passive(x))` → `aadc.math.floor(x)` in code generator
4. **VFJ.func optimization**: 35× faster than Python compute_rates
5. **Full IFT sensitivity**: warmup + BDF2 formula + algebraic vars + initial state params
6. **Architecture**: one small kernel (compute_rates) replayed per step — no huge tape

### FHN Benchmark (non-stiff, for comparison)

| Method | Cost | Time |
|--------|------|------|
| FD | 2.66e-03 | 50s |
| CasADi AD | 5.33e-05 | 44s |
| **AADC AD** | **6.06e-06** | **2.4s** |

AADC **16-20× faster** than both FD and CasADi on non-stiff models.

### Tape approach: semi-implicit on AADC tape (2026-07-24, proof-of-concept)

**Key insight**: record entire BDF integration as idouble operations on one AADC tape.
Forward+reverse replay in C++ eliminates Python loop overhead.

| Metric | Python IFT | **Tape semi-implicit** | CasADi |
|--------|-----------|----------------------|--------|
| Per eval | 46s | **2.2s** | 2s |
| Gradient AD/FD | 1.000 | **1.000** | 1.000 |
| Forward error | 0% | 0.97% | 0% |
| Recording | 0 | 10 min (one-time) | 0 |

**Profiling breakdown** (Python IFT bottleneck):
- aadc.evaluate × 22000: 19s (41%)
- dfdp dict→numpy: 5.5s (12%)
- Python loop overhead: 20s (43%)

**Tape approach eliminates all three** — entire integration replays in compiled C++.

**Sparsity**: Jacobian 10% non-zero, 6 colors → 4.5× FD speedup.
Semi-implicit (diagonal J only) avoids GE and LU entirely.

### Next steps

1. **Integrate tape semi-implicit into optimizer pipeline** — record once, replay per call
2. **Full Newton on tape** — needs pivoted GE with aadc.iif (complex but possible)
3. **AADC tape for compute_variables**: replace FD chain rule with AD for var obs
