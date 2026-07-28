# Profiling the benchmarks

`benchmarks/profile_benchmark.py` runs one optimiser leg of one benchmark under `cProfile` and
writes a `.prof` (for `snakeviz` / `pstats`) plus a text summary with cumulative- and
internal-time tables.

```bash
# profile the 3compartment GA leg in this checkout
python benchmarks/profile_benchmark.py --benchmark three_compartment --method genetic_algorithm

# profile an OLDER checkout with the identical harness, for an A/B
python benchmarks/profile_benchmark.py --root /path/to/old/worktree --tag old \
    --benchmark three_compartment --num-calls 744
```

`--root` points the import path at an arbitrary source tree, so two commits can be profiled by the
same code rather than by two hand-written scripts. Run it at a **single MPI rank**: with several
ranks, rank 0's profile is dominated by time spent waiting on the others and the real hot path is
buried. Keep `--num-calls` small — a profile only needs enough cost evaluations for the hot path to
dominate fixed setup, and cProfile is not free.

## Why call counts, not wall-clock

Wall-clock on a developer laptop cannot resolve sub-2x differences. Measuring the *same commit*
twice at 8 ranks gave 87.5 s and 186.3 s — a **2.1x spread on identical code**. A heterogeneous CPU
(e.g. the i7-12700H's 6 performance + 8 efficiency cores) is a plausible cause: which physical cores
the ranks land on varies per launch, and every MPI barrier runs at the speed of the slowest rank.

cProfile **call counts are deterministic**. They are unaffected by core placement, thermal state or
scheduling, so they answer "does this commit do more work?" at n=1, which no amount of wall-clock
repetition on this hardware reliably does. Prefer them for any regression question; use wall-clock
only for effects far larger than the machine's variance, and repeat it when you do.

## Worked example: a suspected ~5x regression that was not one

The published 3compartment numbers (237.9 s for the GA leg at 8 cores) could not be reproduced —
the same benchmark measured 754 s and 1129 s on two later runs, suggesting a large regression.
Profiling the GA leg at the publishing commit and at master, single rank, same harness:

| | old commit | master |
|---|---|---|
| total function calls | 24,407,014 | 24,412,164 (+0.02%) |
| `sim_step` ncalls | 178,183 | 178,231 (+0.03%) |
| `sim_step` tottime | 206.5 s | 187.8 s |
| wall | 224.5 s | 203.8 s |

Every hot Python function matched to the exact call count (`viter` 2745442/1455502, `validate`
256500, `follow` 376500/64500, `create_unique_names` 1501, `_validate_solvability` 1500). The
integrator is entered 0.03% more often — CVODE's adaptive stepping, not different work. On
FitzHugh-Nagumo master executed **10.6% fewer** calls (1,825,018 vs 2,041,149).

**Conclusion: no regression.** Master does the same or less work and ran faster in both profiles.
The unreproducible published number is an artefact of the measuring machine, not of the code.

The practical lesson: **absolute wall-clock from this benchmark suite is not comparable across
days or machines.** Published tables should carry the hardware and rank count (they do) and should
not be treated as a regression baseline. For that, compare call counts.
