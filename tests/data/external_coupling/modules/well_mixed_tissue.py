"""A numpy external model: one well-mixed tissue volume per connected capillary.

dC/dt = J/V + M (1 - exp(-k_reduce C)), as the CellML module well_mixed_volume next to it, so a
coupled run can be checked against the all-CellML model."""
import copy

import numpy as np

from libcuflynx.coupling import ExternalModel


class WellMixedTissue(ExternalModel):
    def setup(self):
        n = len(self.neighbours['C_t'])
        self.C = np.full(n, self.params['C_init'])
        self.V, self.M, self.k = self.params['V'], self.params['M'], self.params['k_reduce']

    def _rhs(self, C, J):
        return J / self.V + self.M * (1.0 - np.exp(-self.k * C))

    def initial_outputs(self):
        return {'C_t': self.C}

    def step(self, t, dt, inputs):
        J = inputs['J_c']
        n = max(1, int(np.ceil(dt / 1e-3)))
        h = dt / n
        for _ in range(n):  # RK4 with the flux held over the coupling step
            k1 = self._rhs(self.C, J)
            k2 = self._rhs(self.C + 0.5 * h * k1, J)
            k3 = self._rhs(self.C + 0.5 * h * k2, J)
            k4 = self._rhs(self.C + h * k3, J)
            self.C = self.C + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return {'C_t': self.C}

    def snapshot(self):
        self._saved = self.C.copy()

    def restore(self):
        self.C = self._saved.copy()
