"""NumPy vs CasADi math primitives chosen once per mode (no per-call mode switch)."""

from __future__ import annotations

import functools
import sys

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None


try:
    import aadc as _aadc
except ImportError:
    _aadc = None


def bind_backend(func, backend):
    """Wrap ``func`` so it runs with its module's ``mb`` global temporarily set to ``backend``.

    The built-in / user / external operation and cost funcs dispatch through a *module-level*
    ``mb`` math backend, and each ``register_*`` hook rebinds that global. When two registries for
    different backends must coexist -- e.g. a ``casadi_python`` model needs casadi-mode funcs for
    the AD path AND numpy-mode funcs for the numeric (gradient-free) cost path -- building the
    second registry would otherwise silently change the backend of the first, so numpy operands
    reach the casadi backend (``ca.sum(x)/x.numel()`` -> ``'numpy.ndarray' has no attribute
    'numel'``, issue #315). Binding each registered func to its backend keeps the registries
    independent regardless of build order. Funcs that don't use a module-level ``mb`` are returned
    unchanged.

    The save/restore is safe because observable/cost evaluation is sequential within a process;
    nested calls (an op func calling another) restore correctly via the try/finally stack.
    """
    module = sys.modules.get(getattr(func, "__module__", None))
    if module is None or not hasattr(module, "mb"):
        return func

    @functools.wraps(func)  # preserves __dict__ markers (@differentiable, series_to_constant, is_MLE)
    def bound(*args, **kwargs):
        saved = module.mb
        module.mb = backend
        try:
            return func(*args, **kwargs)
        finally:
            module.mb = saved

    return bound


def make_math_backend(mode: str):
    if mode == "numpy":
        return NumpyBackend()
    if mode == "casadi":
        if ca is None:
            raise ImportError("casadi_python mode requires casadi to be installed.")
        return CasadiBackend()
    if mode == "aadc":
        if _aadc is None:
            raise ImportError("aadc mode requires aadc to be installed.")
        return AadcBackend()
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


class AadcBackend:
    """Math backend for AADC idouble.

    np.power/np.abs/np.exp silently convert idouble to float, losing the tape.
    This backend uses idouble methods directly to keep operations on tape.
    """
    __slots__ = ()

    def max(self, x):
        # For scalar idouble, just return it. For lists, use iif chain.
        if hasattr(x, '__len__'):
            result = x[0]
            for i in range(1, len(x)):
                result = _aadc.iif(x[i] > result, x[i], result)
            return result
        return x

    def min(self, x):
        if hasattr(x, '__len__'):
            result = x[0]
            for i in range(1, len(x)):
                result = _aadc.iif(x[i] < result, x[i], result)
            return result
        return x

    def mean(self, x):
        if hasattr(x, '__len__'):
            s = x[0]
            for i in range(1, len(x)):
                s = s + x[i]
            return s / len(x)
        return x

    def max_minus_min(self, x):
        return self.max(x) - self.min(x)

    def power(self, a, b):
        if isinstance(b, int) and b == 2:
            return a * a
        return a.pow(float(b))

    def abs(self, x):
        # smooth abs: iif(x > 0, x, -x)
        return _aadc.iif(x > _aadc.idouble(0.0), x, -x)

    def sum(self, x):
        if hasattr(x, '__len__'):
            s = x[0]
            for i in range(1, len(x)):
                s = s + x[i]
            return s
        return x

    def exp(self, x):
        return x.exp()

    def log(self, x):
        return x.log()

    def zeros(self, n):
        return [_aadc.idouble(0.0) for _ in range(int(n))]

    def numel(self, x):
        if hasattr(x, '__len__'):
            return len(x)
        return 1


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
