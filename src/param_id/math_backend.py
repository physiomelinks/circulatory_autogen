"""NumPy vs CasADi math primitives chosen once per mode (no per-call mode switch)."""

from __future__ import annotations

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None


def make_math_backend(mode: str):
    if mode == "numpy":
        return NumpyBackend()
    if mode == "casadi":
        if ca is None:
            raise ImportError("casadi_python mode requires casadi to be installed.")
        return CasadiBackend()
    raise ValueError(f"Unknown math backend mode: {mode!r}")


class NumpyBackend:
    __slots__ = ()

    def max(self, x):
        return np.max(x)

    def min(self, x):
        return np.min(x)

    def mean(self, x):
        return np.mean(x)

    def max_minus_min(self, x):
        return np.max(x) - np.min(x)

    def power(self, a, b):
        return np.power(a, b)

    def abs(self, x):
        return np.abs(x)

    def sum(self, x):
        return np.sum(x)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def zeros(self, n):
        return np.zeros(int(n))

    def numel(self, x):
        if np.isscalar(x):
            return 1
        return int(np.asarray(x).size)


class CasadiBackend:
    __slots__ = ()

    def max(self, x):
        return ca.mmax(x)

    def min(self, x):
        return ca.mmin(x)

    def mean(self, x):
        return ca.sum(x) / x.numel()

    def max_minus_min(self, x):
        return ca.mmax(x) - ca.mmin(x)

    def power(self, a, b):
        return ca.power(a, b)

    def abs(self, x):
        return ca.fabs(x)

    def sum(self, x):
        return ca.sum(x)

    def exp(self, x):
        return ca.exp(x)

    def log(self, x):
        return ca.log(x)

    def zeros(self, n):
        return ca.MX.zeros(int(n))

    def numel(self, x):
        return int(x.numel())
