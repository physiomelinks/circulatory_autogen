"""A param_id's cost with some of its entries held at given values.

A calibration that holds entries still evaluates the param_id it was built for: every call
puts the held values back into the full vector. So the param_id and its backends (a model, or
an emulator, which answers only for the parameters it was trained on) see the parameter set
they were built for. Profile likelihood holds one entry at each sweep value this way; a
calibration of some of an emulator's parameters holds the others (``ParamID.run`` with
``set_calibrated_entries``).
"""
import numpy as np


class HeldEntries:
    """The param_id's cost with the entries ``indices`` held at ``values``, over the others.

    This is the ``cost_obj`` an optimiser calibrates (see ``ParamID.make_optimiser``), over the
    free entries in the param_id's order. A grouped or modifier entry is held as a whole by
    ``expand_modifier_param_vals``. ``param_init`` is the start of the free entries.
    """

    def __init__(self, param_id, indices, values, param_init):
        self.param_id = param_id
        self.indices = [int(i) for i in indices]
        self.values = np.asarray(values, dtype=float).reshape(len(self.indices))
        num_params = getattr(param_id, 'num_params', None) or \
            len(param_id.param_id_info['param_names'])
        if len(set(self.indices)) != len(self.indices) or \
                any(not 0 <= i < num_params for i in self.indices):
            raise ValueError(f'held entries {self.indices} are not distinct entries of the '
                             f'{num_params} this param_id has.')
        self.free = [i for i in range(num_params) if i not in self.indices]
        self.num_params = num_params
        self.param_init = np.asarray(param_init, dtype=float)
        self.best_param_vals = None

    def full(self, param_vals):
        """The param_id's vector: ``param_vals`` (the free entries) with the held values."""
        full = np.empty(self.num_params, dtype=float)
        full[self.indices] = self.values
        full[self.free] = np.asarray(param_vals, dtype=float).reshape(len(self.free))
        return full

    def reduced_info(self):
        """``param_names``, ``param_mins`` and ``param_maxs`` of the free entries."""
        info = self.param_id.param_id_info
        return {'param_names': [info['param_names'][i] for i in self.free],
                'param_mins': np.asarray(info['param_mins'], dtype=float)[self.free],
                'param_maxs': np.asarray(info['param_maxs'], dtype=float)[self.free]}

    def _reduced(self, gradient):
        return np.asarray(gradient, dtype=float).ravel()[self.free]

    def get_cost_from_params(self, param_vals):
        return self.param_id.get_cost_from_params(self.full(param_vals))

    def get_cost(self, param_vals):
        return self.param_id.get_cost(self.full(param_vals))

    def get_lnlikelihood_lnprior_from_params(self, param_vals):
        return self.param_id.get_lnlikelihood_lnprior_from_params(self.full(param_vals))

    def get_gradient(self, param_vals):
        return self._reduced(self.param_id.get_gradient(self.full(param_vals)))

    def get_cost_and_gradient(self, param_vals):
        cost, gradient = self.param_id.get_cost_and_gradient(self.full(param_vals))
        return cost, self._reduced(gradient)

    def set_best_param_vals(self, param_vals):
        self.best_param_vals = np.asarray(param_vals, dtype=float)
