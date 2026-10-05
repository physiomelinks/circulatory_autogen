'''
Stored parameter distributions: a step's MCMC posterior, kept so later steps (and tools like
CUFLynx) can sample from it, evaluate it, and use it as a prior.

A stored distribution is a directory::

    distribution/
        chain.npy           the raw (steps, walkers, params) chain, as sampled
        samples.npy         post-burn-in, flattened (N, params): the source of truth
        stats.json          per-parameter mean/sd/quantiles, and the covariance, on the raw
                            and on the transformed scale
        distribution.json   the manifest: parameters, bounds, transforms, provenance

Every representation is fitted to ``samples.npy`` on demand, so a new one can be added
without re-running MCMC. Each works on a *transformed* scale on which a bounded parameter is
unbounded, so that its density respects the bounds and can follow a skewed posterior:

* both bounds finite: ``z = logit((x - min) / (max - min))``
* only the lower bound finite: ``z = log(x - min)``
* otherwise: ``z = x``

Representations (``kind``):

* ``mvnormal`` -- a multivariate normal on z (mean and full covariance): keeps correlations;
  exact, cheap sampling and density. Unimodal.
* ``normal`` -- independent normals on z (the covariance's diagonal).
* ``kde`` -- a Gaussian kernel density estimate on z (scipy): any shape, but each density
  evaluation costs O(N) and it degrades beyond ~5-6 parameters.

``logpdf`` is the density of x, i.e. of z plus the log Jacobian of the transform, so it can
be added to a log prior directly. ``inflate`` multiplies the z-scale variance (the KDE's
samples are spread about their mean by sqrt(inflate)), to temper an overconfident upstream
posterior.
'''

import json
import os
from dataclasses import dataclass

import numpy as np

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.spec import PRIOR_KINDS

DISTRIBUTION_DIR = 'distribution'
MANIFEST = 'distribution.json'
FORMAT_VERSION = 1

TRANSFORM_LOGIT = 'logit'
TRANSFORM_LOG = 'log'
TRANSFORM_IDENTITY = 'identity'

_EPS = 1e-12


def transform_for_bounds(lo, hi):
    lo_finite, hi_finite = lo is not None and np.isfinite(lo), hi is not None and np.isfinite(hi)
    if lo_finite and hi_finite:
        return TRANSFORM_LOGIT
    if lo_finite:
        return TRANSFORM_LOG
    return TRANSFORM_IDENTITY


@dataclass
class DistributionParameter:
    '''One dimension of a stored distribution: a calibrated parameter (a params_for_id row),
    which may set several ``(vessel, param)`` targets.'''
    targets: list           # [[vessel, param], ...] in the source step's model
    min: float
    max: float
    transform: str

    def model_names(self):
        return [naming.model_name(v, p) for v, p in self.targets]

    def mapped(self, path):
        return DistributionParameter(
            targets=[[naming.map_vessel(v, path), p] for v, p in self.targets],
            min=self.min, max=self.max, transform=self.transform)

    def as_dict(self):
        return {'targets': [list(t) for t in self.targets], 'model_names': self.model_names(),
                'min': _json_float(self.min), 'max': _json_float(self.max),
                'transform': self.transform}


def _json_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _from_json_float(value, default):
    return default if value is None else float(value)


# --------------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------------

def to_z(x, parameters):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z = np.empty_like(x)
    for j, par in enumerate(parameters):
        col = x[:, j]
        if par.transform == TRANSFORM_LOGIT:
            u = np.clip((col - par.min) / (par.max - par.min), _EPS, 1 - _EPS)
            z[:, j] = np.log(u) - np.log1p(-u)
        elif par.transform == TRANSFORM_LOG:
            z[:, j] = np.log(np.maximum(col - par.min, _EPS))
        else:
            z[:, j] = col
    return z


def from_z(z, parameters):
    z = np.atleast_2d(np.asarray(z, dtype=float))
    x = np.empty_like(z)
    for j, par in enumerate(parameters):
        col = z[:, j]
        if par.transform == TRANSFORM_LOGIT:
            x[:, j] = par.min + (par.max - par.min) / (1.0 + np.exp(-col))
        elif par.transform == TRANSFORM_LOG:
            x[:, j] = par.min + np.exp(col)
        else:
            x[:, j] = col
    return x


def log_jacobian(x, parameters):
    '''``sum_j log |dz_j/dx_j|`` per row of ``x``; -inf outside the bounds.'''
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.zeros(x.shape[0])
    for j, par in enumerate(parameters):
        col = x[:, j]
        if par.transform == TRANSFORM_LOGIT:
            width = par.max - par.min
            u = (col - par.min) / width
            inside = (u > 0) & (u < 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                term = -np.log(width) - np.log(u) - np.log1p(-u)
            out += np.where(inside, term, -np.inf)
        elif par.transform == TRANSFORM_LOG:
            d = col - par.min
            with np.errstate(divide='ignore', invalid='ignore'):
                out += np.where(d > 0, -np.log(d), -np.inf)
    return out


# --------------------------------------------------------------------------------------------
# the stored distribution
# --------------------------------------------------------------------------------------------

class StoredDistribution:
    '''A posterior stored by a workflow step; see the module docstring.'''

    def __init__(self, parameters, samples, source=None, directory=None, chain=None):
        self.parameters = list(parameters)
        self.samples = np.atleast_2d(np.asarray(samples, dtype=float))
        if self.samples.shape[1] != len(self.parameters):
            raise ValueError(f'samples have {self.samples.shape[1]} columns for '
                             f'{len(self.parameters)} parameters.')
        if self.samples.shape[0] < 2:
            raise ValueError('a stored distribution needs at least two samples.')
        self.source = dict(source or {})
        self.directory = directory
        self.chain = chain
        self._z = to_z(self.samples, self.parameters)

    # -- naming --------------------------------------------------------------------------

    @property
    def names(self):
        '''The first model name of each dimension (its label).'''
        return [p.model_names()[0] for p in self.parameters]

    def index(self, name):
        for j, par in enumerate(self.parameters):
            if name in par.model_names():
                return j
        raise KeyError(f'"{name}" is not a parameter of this distribution ({self.names}).')

    def mapped_into(self, path):
        '''The same distribution with its parameters named as in the model in which the
        source module is the submodule at ``path``.'''
        if not path:
            return self
        return StoredDistribution([p.mapped(path) for p in self.parameters], self.samples,
                                  source=dict(self.source, mapped_into=path),
                                  directory=self.directory)

    def subset(self, names):
        '''The marginal distribution of the dimensions named ``names`` (model names).'''
        idx = [self.index(n) for n in names]
        return StoredDistribution([self.parameters[i] for i in idx], self.samples[:, idx],
                                  source=self.source, directory=self.directory)

    # -- summaries -----------------------------------------------------------------------

    def stats(self):
        quantiles = (0.025, 0.25, 0.5, 0.75, 0.975)
        out = {'num_samples': int(self.samples.shape[0]), 'parameters': {}}
        for j, name in enumerate(self.names):
            col = self.samples[:, j]
            out['parameters'][name] = {
                'mean': float(np.mean(col)), 'sd': float(np.std(col, ddof=1)),
                'quantiles': {str(q): float(np.quantile(col, q)) for q in quantiles},
                'z_mean': float(np.mean(self._z[:, j])),
                'z_sd': float(np.std(self._z[:, j], ddof=1)),
            }
        out['covariance'] = np.atleast_2d(np.cov(self.samples, rowvar=False)).tolist()
        out['z_covariance'] = np.atleast_2d(np.cov(self._z, rowvar=False)).tolist()
        return out

    def marginal(self, name):
        '''The samples of one parameter.'''
        return self.samples[:, self.index(name)].copy()

    # -- representations -----------------------------------------------------------------

    def representation(self, kind='mvnormal', inflate=1.0):
        if kind not in PRIOR_KINDS:
            raise ValueError(f'unknown distribution kind "{kind}"; one of {list(PRIOR_KINDS)}.')
        if inflate <= 0:
            raise ValueError('inflate must be positive.')
        if kind == 'kde':
            return _KDE(self._z, inflate)
        mean = self._z.mean(axis=0)
        cov = np.atleast_2d(np.cov(self._z, rowvar=False)) * inflate
        if kind == 'normal':
            cov = np.diag(np.diag(cov))
        return _MVNormal(mean, cov)

    def sample(self, n, rng=None, kind='mvnormal', inflate=1.0):
        '''``n`` draws, shape (n, params). ``rng`` is a numpy Generator or a seed.'''
        rng = np.random.default_rng(rng)
        return from_z(self.representation(kind, inflate).sample(n, rng), self.parameters)

    def logpdf(self, x, kind='mvnormal', inflate=1.0):
        '''The log density of ``x`` ((params,) or (n, params)) on the parameters' own scale.'''
        x2 = np.atleast_2d(np.asarray(x, dtype=float))
        jac = log_jacobian(x2, self.parameters)
        out = np.full(x2.shape[0], -np.inf)
        ok = np.isfinite(jac)
        if np.any(ok):
            out[ok] = self.representation(kind, inflate).logpdf(to_z(x2[ok], self.parameters)) \
                + jac[ok]
        return out[0] if np.ndim(x) == 1 else out

    def as_prior(self, kind='mvnormal', inflate=1.0):
        '''A ``logpdf(param_vals) -> float`` for one parameter vector, for
        ``ParamID.set_joint_priors``. The representation is fitted once, here.'''
        rep = self.representation(kind, inflate)
        parameters = self.parameters

        def logpdf(values):
            x = np.asarray(values, dtype=float)[None, :]
            jac = log_jacobian(x, parameters)[0]
            if not np.isfinite(jac):
                return -np.inf
            value = float(rep.logpdf(to_z(x, parameters))[0]) + jac
            return value if np.isfinite(value) else -np.inf

        logpdf.kind, logpdf.inflate = kind, inflate
        return logpdf

    # -- persistence -------------------------------------------------------------------

    def save(self, directory, chain=None):
        os.makedirs(directory, exist_ok=True)
        if chain is not None:
            np.save(os.path.join(directory, 'chain.npy'), np.asarray(chain))
        np.save(os.path.join(directory, 'samples.npy'), self.samples)
        with open(os.path.join(directory, 'stats.json'), 'w') as f:
            json.dump(self.stats(), f, indent=1)
        manifest = {'format_version': FORMAT_VERSION,
                    'parameters': [p.as_dict() for p in self.parameters],
                    'num_samples': int(self.samples.shape[0]),
                    'kinds': list(PRIOR_KINDS),
                    'source': self.source}
        with open(os.path.join(directory, MANIFEST), 'w') as f:
            json.dump(manifest, f, indent=1)
        self.directory = directory
        return directory

    @classmethod
    def load(cls, directory):
        '''The distribution stored in ``directory`` (a ``distribution/`` dir, or a step's
        output dir containing one).'''
        if not os.path.isfile(os.path.join(directory, MANIFEST)) and \
                os.path.isfile(os.path.join(directory, DISTRIBUTION_DIR, MANIFEST)):
            directory = os.path.join(directory, DISTRIBUTION_DIR)
        with open(os.path.join(directory, MANIFEST)) as f:
            manifest = json.load(f)
        if manifest.get('format_version') != FORMAT_VERSION:
            raise ValueError(f'{directory}: stored distribution format '
                             f'{manifest.get("format_version")} is not supported.')
        parameters = [DistributionParameter(targets=[list(t) for t in p['targets']],
                                            min=_from_json_float(p['min'], -np.inf),
                                            max=_from_json_float(p['max'], np.inf),
                                            transform=p['transform'])
                      for p in manifest['parameters']]
        samples = np.load(os.path.join(directory, 'samples.npy'))
        chain_path = os.path.join(directory, 'chain.npy')
        return cls(parameters, samples, source=manifest.get('source'), directory=directory,
                   chain=chain_path if os.path.isfile(chain_path) else None)

    @classmethod
    def from_chain(cls, chain, parameters, burn_in_index=None, source=None):
        '''From an MCMC chain ``(steps, walkers, params)``: steps before ``burn_in_index``
        (default half the chain) are dropped and the rest flattened.'''
        chain = np.asarray(chain, dtype=float)
        if chain.ndim != 3:
            raise ValueError(f'expected a (steps, walkers, params) chain, got shape '
                             f'{chain.shape}.')
        start = chain.shape[0] // 2 if burn_in_index is None else int(burn_in_index)
        start = min(max(start, 0), chain.shape[0] - 1)
        flat = chain[start:].reshape(-1, chain.shape[2])
        flat = flat[np.all(np.isfinite(flat), axis=1)]
        return cls(parameters, flat, source=dict(source or {}, burn_in_index=start))


class _MVNormal:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        d = self.mean.size
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        # a parameter the chain never moved has zero variance; give it a tiny one so the
        # density is defined rather than singular
        cov = cov + np.eye(d) * max(1e-12, 1e-9 * float(np.max(np.abs(np.diag(cov))) or 1.0))
        self.cov = cov
        self._chol = np.linalg.cholesky(cov)
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        self._const = -0.5 * (d * np.log(2 * np.pi) + self._logdet)

    def logpdf(self, z):
        diff = np.atleast_2d(z) - self.mean
        sol = np.linalg.solve(self._chol, diff.T)
        return self._const - 0.5 * np.sum(sol ** 2, axis=0)

    def sample(self, n, rng):
        return self.mean + rng.standard_normal((n, self.mean.size)) @ self._chol.T


class _KDE:
    def __init__(self, z, inflate):
        from scipy.stats import gaussian_kde
        mean = z.mean(axis=0)
        spread = mean + (z - mean) * np.sqrt(inflate)
        self._kde = gaussian_kde(spread.T)

    def logpdf(self, z):
        return self._kde.logpdf(np.atleast_2d(z).T)

    def sample(self, n, rng):
        return self._kde.resample(n, seed=rng).T


def parameters_from_info(param_id_info_like, mins, maxs):
    '''DistributionParameters from a list of qname lists (``[['mod/p'], ...]``) and bounds.'''
    parameters = []
    for qnames, lo, hi in zip(param_id_info_like, mins, maxs):
        targets = []
        for qname in (qnames if isinstance(qnames, (list, tuple)) else [qnames]):
            vessel, _, param = str(qname).partition('/')
            targets.append([vessel, param])
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        parameters.append(DistributionParameter(targets=targets, min=lo, max=hi,
                                                transform=transform_for_bounds(lo, hi)))
    return parameters
