| model | Fisher count | variant | rms | median | rms_converged | median_converged |
|---|---|---|---|---|---|---|
| fitzhugh_nagumo | 5 | centre_central_1e-2 | 5 | 5 | - | - |
| fitzhugh_nagumo | 5 | K8_forward_1e-2 | 5 | 5 | - | - |
| fitzhugh_nagumo | 5 | K8_central_1e-2 | 5 | 5 | - | - |
| fitzhugh_nagumo | 5 | K8_central_1e-2_checked | 5 | 5 | 5 | 5 |
| fitzhugh_nagumo | 5 | K8_central_1e-3_checked | 5 | 5 | 5 | 5 |
| fitzhugh_nagumo | 5 | K16_central_1e-3_checked | 5 | 5 | 5 | 5 |
| goodwin | 4 | centre_central_1e-2 | 4 | 4 | - | - |
| goodwin | 4 | K8_forward_1e-2 | 5 | 5 | - | - |
| goodwin | 4 | K8_central_1e-2 | 5 | 5 | - | - |
| goodwin | 4 | K8_central_1e-2_checked | 5 | 5 | 5 (3 unconv.) | 4 (3 unconv.) |
| goodwin | 4 | K8_central_1e-3_checked | 5 | 4 | 4 (2 unconv.) | 4 (2 unconv.) |
| goodwin | 4 | K16_central_1e-3_checked | 5 | 4 | 4 (2 unconv.) | 4 (2 unconv.) |
| teusink | 6 | centre_central_1e-2 | 5 | 5 | - | - |
| teusink | 6 | K8_forward_1e-2 | 5 | 5 | - | - |
| teusink | 6 | K8_central_1e-2 | 5 | 5 | - | - |
| teusink | 6 | K8_central_1e-2_checked | 5 | 5 | 5 | 5 |
| teusink | 6 | K8_central_1e-3_checked | 5 | 5 | 5 | 5 |
| teusink | 6 | K16_central_1e-3_checked | 5 | 5 | 5 | 5 |

How the rows were produced: `make_inputs.py` and `run_study.py <model> <out> --stage sa` for each model (the Fisher counts are `compare.py`'s `compare/summary.json`), then `compare_redundancy.py <study> <out>` on 8 ranks, which writes this table as `<out>/redundancy_comparison.md`. Run on 2026-11-21. Reducing the same Sobol runs with the selection rule that kept the first of two redundant parameters, and with the rule that keeps the one the Sobol indices rank highest, gives the same counts in every row.
