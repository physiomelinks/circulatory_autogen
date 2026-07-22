"""Discoverable schema for the calisim calibration backends.

calisim (https://github.com/Plant-Food-Research-Open/calisim) is a unified interface over many
optimisation libraries. Rather than hand-writing one `Optimiser` subclass per library, CA wraps
calisim once (see `param_id/calisim_wrapper.py`) and exposes every calisim optimisation
engine/method pair as a `param_id_method` named ``calisim_<engine>[_<method>]``.

This module is the *schema* half of that: it builds the PARAM_ID_METHODS entries for those
methods so downstream tools (CUFLynx) pick them up automatically, without hardcoding a list on
the GUI side. It deliberately imports nothing from CA (PrimitiveParsers imports it, so any CA
import here would be circular) and nothing from calisim at module scope -- calisim is an
*optional* dependency and CA must import and run exactly as before without it.

Discovery is static-table-first, then extended from the installed calisim:

* the static table below is always exposed, so the CUFLynx menu is the same on a machine that has
  not installed calisim yet as on one that has;
* if calisim *is* installed, any extra optimisation engine registered with it (including
  third-party ones registered through the ``calisim.external.optimisation`` entry-point group) is
  added, and OpenTURNS' Pagmo algorithm list -- which is only knowable at run time -- is queried
  and added.

Keep the option descriptors in sync with what `CalisimOptimiser` actually reads from
`optimiser_options`; `tests/test_solver_info_validation.py` fails the build if they drift.
"""

import re

# calisim's optimisation module. `method` selects the sampler/algorithm within an engine; for
# emukit and botorch the wrapper ignores `method` entirely (control is via acquisition_func /
# Ax's own generation strategy), so those engines get a single, method-less CA entry.
CALISIM_TASK = 'optimisation'

_CALISIM_ENGINE_METHODS = {
    # engine: (methods, needs_acquisition_func, description)
    'optuna': (
        # The samplers calisim's optuna wrapper accepts (its `supported_samplers` dict).
        ['tpes', 'cmaes', 'nsga', 'qmc', 'gp'],
        False,
        'Optuna sampler',
    ),
    'openturns': (
        # 'kriging' is OpenTURNS' EGO; the rest are Pagmo algorithms. The authoritative list comes
        # from ot.Pagmo.GetAlgorithmNames() at run time (see _discover_openturns_methods); these
        # are the ones present in every OpenTURNS we know of, listed so the menu is populated even
        # without calisim/openturns installed.
        ['kriging', 'de', 'sade', 'de1220', 'pso', 'sga', 'cmaes', 'xnes'],
        False,
        'OpenTURNS EGO / Pagmo algorithm',
    ),
    'emukit': (
        [],
        True,
        'Emukit Bayesian optimisation (the method is selected by acquisition_func)',
    ),
    'botorch': (
        [],
        True,
        'BoTorch/Ax Bayesian optimisation (requires the calisim [torch] extra)',
    ),
}


# ---------------------------------------------------------------------------------------------
# optimiser_options descriptors. Every name here MUST be read by CalisimOptimiser.
# ---------------------------------------------------------------------------------------------
_CALISIM_COMMON_OPTIONS = [
    {'name': 'num_calls_to_function', 'type': 'int', 'default': 100, 'required': False,
     'description': 'Evaluation budget: number of calisim iterations (cost-function calls).'},
    {'name': 'cost_convergence', 'type': 'float', 'default': 1e-4, 'required': False,
     'description': 'Stop once the cost drops below this value.'},
    {'name': 'n_init', 'type': 'int', 'default': 10, 'required': False,
     'description': ('Number of initial (design-of-experiment) evaluations before the algorithm '
                     'proper takes over. Used by the surrogate-based engines (emukit, botorch, '
                     'openturns); optuna samplers take their start-up count through '
                     'method_kwargs instead. For the openturns/Pagmo algorithms this is the '
                     'population size, and they impose algorithm-specific minima (differential '
                     'evolution needs at least 5).')},
    {'name': 'random_seed', 'type': 'int', 'default': 0, 'required': False,
     'description': 'Seed for reproducible runs (passed to the sampler for the optuna engine).'},
    {'name': 'n_jobs', 'type': 'int', 'default': 1, 'required': False,
     'description': 'Number of parallel workers calisim itself may use within one MPI rank.'},
    {'name': 'method_kwargs', 'type': 'dict', 'default': None, 'required': False,
     'description': ('Extra keyword arguments passed straight through to the calisim '
                     'sampler/algorithm constructor, e.g. {"n_startup_trials": 20}.')},
]

_CALISIM_ACQUISITION_OPTION = {
    'name': 'acquisition_func', 'type': 'enum', 'default': 'ei', 'required': False,
    'choices': ['ei', 'poi', 'lp', 'nlcb'],
    'description': ('Acquisition function for the surrogate-based engines (expected improvement, '
                    'probability of improvement, local penalisation, negative lower confidence '
                    'bound).'),
}


def calisim_method_name(engine, method=None):
    """The `param_id_method` string for a calisim engine/method pair."""
    if method:
        return f'calisim_{engine}_{method}'
    return f'calisim_{engine}'


def split_calisim_method_name(param_id_method):
    """Inverse of `calisim_method_name`: ``calisim_optuna_tpes`` -> ``('optuna', 'tpes')``.

    Returns ``(engine, method)`` with ``method == ''`` for the method-less engines. Raises
    ValueError if the string is not a calisim method name.
    """
    if not is_calisim_method(param_id_method):
        raise ValueError(f"'{param_id_method}' is not a calisim param_id_method")
    remainder = param_id_method[len('calisim_'):]
    if not remainder:
        raise ValueError("calisim param_id_method must name an engine, e.g. 'calisim_optuna_tpes'")
    engine, _, method = remainder.partition('_')
    return engine, method


def is_calisim_method(param_id_method):
    """True if `param_id_method` selects a calisim backend."""
    return isinstance(param_id_method, str) and param_id_method.startswith('calisim_')


def _discover_openturns_methods():
    """Pagmo algorithm names, which only the installed OpenTURNS knows. Empty if unavailable."""
    try:
        import openturns as ot
        return [str(name) for name in ot.Pagmo.GetAlgorithmNames()]
    except Exception:
        return []


def _discover_calisim_engines():
    """Optimisation engines registered with the *installed* calisim, if any.

    Covers engines added since this file was written and third-party ones registered through the
    ``calisim.external.optimisation`` entry-point group. Returns an empty list when calisim is not
    installed or its registry cannot be read -- discovery must never break CA's import.
    """
    try:
        # Module-level helper in calisim: BASE_IMPLEMENTATIONS merged with whatever is registered
        # under the `calisim.external.optimisation` entry-point group. Keyed by engine name.
        from calisim.optimisation.implementation import get_implementations
        return [str(engine) for engine in get_implementations()]
    except Exception:
        return []


def _method_entry(engine, method, needs_acquisition_func, engine_description):
    label_method = method.replace('_', ' ') if method else 'default'
    options = list(_CALISIM_COMMON_OPTIONS)
    if needs_acquisition_func:
        options = options + [_CALISIM_ACQUISITION_OPTION]
    return {
        'label': f'calisim: {engine} / {label_method}',
        'gradient_based': False,
        'description': (f'Gradient-free calibration through calisim using the {engine} engine '
                        f'({engine_description}'
                        + (f", method '{method}'" if method else '') + '). '
                        'Requires the optional `calisim` package.'),
        'options': options,
        'calisim': {'engine': engine, 'method': method},
    }


def calisim_param_id_methods():
    """The PARAM_ID_METHODS entries for every available calisim optimisation engine/method.

    Merged into PARAM_ID_METHODS by `parsers.PrimitiveParsers` so CUFLynx auto-populates its
    calibration-method menu and the per-method settings form with no hardcoding.
    """
    engine_methods = {engine: (list(methods), acq, desc)
                      for engine, (methods, acq, desc) in _CALISIM_ENGINE_METHODS.items()}

    # Extend the static table with whatever the installed calisim/openturns actually offer.
    for name in _discover_openturns_methods():
        methods, acq, desc = engine_methods['openturns']
        if name not in methods:
            methods.append(name)
    for engine in _discover_calisim_engines():
        engine_methods.setdefault(engine, ([], False, 'engine discovered from installed calisim'))

    methods_schema = {}
    for engine, (methods, acq, desc) in engine_methods.items():
        if not methods:
            methods_schema[calisim_method_name(engine)] = _method_entry(engine, '', acq, desc)
            continue
        for method in methods:
            if not re.fullmatch(r'[A-Za-z0-9_]+', str(method)):
                # `calisim_<engine>_<method>` is split back apart on the first underscore, so an
                # underscore inside the method name is fine (engine names have none) but anything
                # else would not round-trip. Skip rather than register an unparseable name.
                continue
            methods_schema[calisim_method_name(engine, method)] = _method_entry(
                engine, method, acq, desc)
    return methods_schema
