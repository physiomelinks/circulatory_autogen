"""One-parameter bounded fits with nevergrad's CMA: why CMAESOptimiser adds an unused variable.

In one dimension ng.optimizers.CMA (MetaCMA) selects OnePlusOne, not CMA. This script runs a
smooth one-minimum cost in a box, 200 seeds x 80 evaluations per case, and counts the runs whose
best point ends more than 1% of the box from the minimum, for:

  default   ng.p.Array(init, lower, upper): nevergrad's OnePlusOne
  arctan    set_bounds(method='arctan')
  dummy     a second, unused variable in [0, 1], so nevergrad runs CMA, after the start
            value is evaluated as the first of the budget: what
            libcuflynx.param_id.optimisers.CMAESOptimiser does for one parameter
  scalar    scipy.optimize.minimize_scalar(method='bounded'), same budget (deterministic)

Run: python benchmarks/cmaes_one_parameter.py (no MPI, about five minutes). With nevergrad
1.0.12 the failed runs of 200 were, for default / arctan / dummy: 21 / 0 / 0 (minimum and start
mid-box), 67 / 200 / 0 (start on a bound), 55 / 27 / 0 (start 1% inside a bound), 4 / 0 / 0
(minimum near a bound), 1 / 0 / 0 (minimum on a bound).
"""
import warnings

import nevergrad as ng
import numpy as np
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
LO, HI = 5.0e-5, 1.15e-4
W = HI - LO
BUDGET, SEEDS = 80, range(200)

CASES = {  # name: (start, minimum, width below, width above), as fractions of the box
    'minimum mid-box, start mid-box':   (0.31, 0.71, 0.083, 0.055),
    'minimum mid-box, start on bound':  (0.00, 0.71, 0.083, 0.055),
    'minimum mid-box, start 1% inside': (0.01, 0.71, 0.083, 0.055),
    'minimum near a bound':             (0.31, 0.97, 0.083, 0.055),
    'minimum on a bound':               (0.31, 1.00, 0.083, 0.055),
}


def make_cost(true, s_left, s_right):
    return lambda x: float(((x - true) / (s_left if x < true else s_right)) ** 2)


def nevergrad_run(method, seed, x0, cost):
    if method == 'default':
        p = ng.p.Array(init=np.array([x0]), lower=np.array([LO]), upper=np.array([HI]))
    elif method == 'arctan':
        p = ng.p.Array(init=np.array([x0]))
        p.set_bounds(np.array([LO]), np.array([HI]), method='arctan')
    else:  # dummy
        p = ng.p.Array(init=np.array([x0, 0.5]), lower=np.array([LO, 0.0]),
                       upper=np.array([HI, 1.0]))
    opt = ng.optimizers.CMA(parametrization=p, budget=BUDGET, num_workers=1)
    opt.parametrization.random_state = np.random.RandomState(seed)
    best, at_bound, budget = (np.inf, None), 0, BUDGET
    if method == 'dummy':
        # as CMAESOptimiser does: the start value is evaluated first, outside nevergrad
        best, budget = (cost(x0), x0), BUDGET - 1
    for _ in range(budget):
        c = opt.ask()
        x = float(c.value[0])
        f = cost(x)
        opt.tell(c, f)
        at_bound += x in (LO, HI)
        if f < best[0]:
            best = (f, x)
    return best[1], at_bound


if __name__ == '__main__':
    print('nevergrad', ng.__version__, '| runs (of 200) ending > 1% of the box from the minimum; '
          'worst error; most candidates on a bound in one run')
    for name, (x0f, tf, sl, sr) in CASES.items():
        x0, true = LO + x0f * W, LO + tf * W
        cost = make_cost(true, sl * W, sr * W)
        print(name)
        for method in ('default', 'arctan', 'dummy'):
            res = [nevergrad_run(method, s, x0, cost) for s in SEEDS]
            err = np.array([abs(r[0] - true) / W for r in res])
            print(f'  {method:8s} {int((err > 0.01).sum()):4d}   worst {err.max():6.1%}   '
                  f'on a bound: up to {max(r[1] for r in res)}/{BUDGET}')
        r = minimize_scalar(cost, bounds=(LO, HI), method='bounded',
                            options={'maxiter': BUDGET, 'xatol': 1e-6 * W})
        print(f'  scalar   error {abs(r.x - true) / W:6.2%} in {r.nfev} evaluations')
