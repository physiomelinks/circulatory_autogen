# Batched CellML solvers: scaling on colonies of fermenter cells

`python benchmarks/batched_solver_scaling.py --jobs 4` at commit 0b0cb786 (BE_batched_c with step-size hysteresis), 2026-10-12, i7-12700H (6 P + 8 E cores), Linux. Other jobs were running (load average 3-4), so single small timings vary by up to ~50% (8 cells, 1 thread). Colonies up to 256 cells had been generated in earlier runs (their CA generation is cached); the 512-cell colony took 50 minutes to generate.

300 simulated s; run wall time (x real time). x86_64, 20 logical CPUs; load average before timing (4.1083984375, 3.6650390625, 3.3369140625).

| solver | threads | 2 cells | 4 cells | 8 cells | 16 cells | 32 cells | 64 cells | 128 cells | 256 cells | 512 cells |
|---|---|---|---|---|---|---|---|---|---|---|
| BE_batched_c, TR-BDF2, rtol 1e-4 | 1 | 0.0019 s (157998x) | 0.00353 s (84939x) | 0.0177 s (16939x) | 0.0123 s (24310x) | 0.0245 s (12238x) | 0.0496 s (6050x) | 0.0978 s (3068x) | 0.196 s (1532x) | 0.411 s (729x) |
| BE_batched_c, TR-BDF2, rtol 1e-4 | 8 | 0.00116 s (257763x) | 0.00119 s (251089x) | 0.00507 s (59225x) | 0.00335 s (89465x) | 0.00518 s (57957x) | 0.00937 s (32013x) | 0.0186 s (16108x) | 0.0366 s (8193x) | 0.0802 s (3741x) |
| BE_batched_c, TR-BDF2, rtol 1e-6 | 1 | 0.00512 s (58572x) | 0.00986 s (30432x) | 0.0179 s (16745x) | 0.034 s (8828x) | 0.0674 s (4454x) | 0.137 s (2189x) | 0.271 s (1107x) | 0.544 s (552x) | 1.18 s (255x) |
| BE_batched_c, TR-BDF2, rtol 1e-6 | 8 | 0.00274 s (109635x) | 0.00393 s (76339x) | 0.00449 s (66769x) | 0.00874 s (34333x) | 0.013 s (23163x) | 0.0239 s (12559x) | 0.0467 s (6419x) | 0.0945 s (3174x) | 0.271 s (1109x) |
| CVODES_batched_c, rtol 1e-6 | 1 | 0.00177 s (169442x) | 0.00339 s (88619x) | 0.00583 s (51475x) | 0.0111 s (27096x) | 0.0227 s (13216x) | 0.045 s (6671x) | 0.0874 s (3433x) | 0.18 s (1668x) | 0.388 s (773x) |
| CVODES_batched_c, rtol 1e-6 | 8 | 0.00105 s (285439x) | 0.00164 s (182641x) | 0.00157 s (190671x) | 0.00298 s (100522x) | 0.00463 s (64775x) | 0.00876 s (34237x) | 0.0164 s (18246x) | 0.0346 s (8669x) | 0.097 s (3091x) |
| CVODE_myokit, rtol 1e-6 | 1 | 0.055 s (5458x) | 0.114 s (2623x) | 0.269 s (1114x) |  |  |  |  |  |  |

Setup, the first time: CA generates the CellML from the module array; each batched solver generates C and builds it (about 0.1 s once cached; built 4 at a time); CVODE_myokit compiles on every load.

| cells | BE_batched_c | CA generation (CellML) | CVODES_batched_c | CVODE_myokit |
|---|---|---|---|---|
| 2 | 2.1 s | (cached) | 1.9 s | 2.6 s |
| 4 | 3.3 s | (cached) | 3.3 s | 6.2 s |
| 8 | 6.8 s | (cached) | 6.9 s | 17.7 s |
| 16 | 15.2 s | (cached) | 15.5 s |  |
| 32 | 35.7 s | (cached) | 36.0 s |  |
| 64 | 95.2 s | (cached) | 94.9 s |  |
| 128 | 415.7 s | (cached) | 419.5 s |  |
| 256 | 1618.9 s | (cached) | 1619.7 s |  |
| 512 | 5900.3 s | 3026.7 s | 5926.6 s |  |

Worst error of cell 0's states, each second, against CVODES_batched_c, rtol 1e-10, atol 1e-12 (16 cells):

| solver | worst relative error |
|---|---|
| BE_batched_c, TR-BDF2, rtol 1e-4 | 2.1e-03 |
| BE_batched_c, TR-BDF2, rtol 1e-6 | 1.1e-04 |
| CVODES_batched_c, rtol 1e-6 | 2.9e-05 |
| CVODE_myokit, rtol 1e-6 | 2.9e-05 |
