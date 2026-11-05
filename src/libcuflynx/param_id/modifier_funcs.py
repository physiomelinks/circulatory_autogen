"""Modifier functions: how a calibrated theta computes the model parameters it governs.

A ``params_for_id`` entry with ``modifies`` names one calibrated variable (theta) and N model
parameters; its ``modifier`` names a **modifier function** that maps theta to each target's
value (issue #383):

    p_i = fn(theta, baseline_i, **inputs)

``baseline_i`` is target i's model-default value, and ``inputs`` are extra model constants the
entry names by qname, resolved to their defaults once at setup -- the same never-re-derived
semantics as the baselines, so nothing compounds across calibration iterations.

Functions are declared with the :func:`modifier_func` decorator, which records each input's
name and type (``'float'`` -- one model constant -- or ``'list'`` -- several, passed as a list
of floats). That declaration is introspectable data: a front-end reads it via
``parsers.PrimitiveParsers.param_modifiers()`` to render the entry form, exactly as it reads
the cost-func registry.

Note the vocabulary: an *operation* acts on an **output** and is declared in obs_data; a
*modifier* acts on a **parameter** and is declared in params_for_id. ('operation' is still
accepted as a deprecated alias of the entry's ``modifier`` key, from #378.)

Every modifier function must be **affine in theta** (``a*theta + b`` for fixed inputs). That is
not a style rule: the analytic gradients differentiate through the mapping with a constant
chain-rule weight ``a = d p_i / d theta`` (see ``fsa_backend.combined_entry_sensitivities``),
and theta's starting value is derived by inverting the mapping at the baseline. Affinity is
verified numerically at setup (``probe_affine``) and a non-affine function is refused there,
before it can produce a silently wrong gradient.

Shipped functions live here and in ``libcuflynx.funcs.modifier_funcs_user``. User-defined
functions go in an external file named by the ``modifier_funcs_external_path`` config key --
mirroring the operation-func (outputs) and cost-func registries. See
``funcs_user/modifier_funcs_example.py`` in the repository for a template.
"""
import functools
import importlib
import os

from libcuflynx.param_id.external_funcs import _load_module_from_path

MODIFIER_INPUT_TYPES = ('float', 'list')


def modifier_func(inputs=None, description=None):
    """Declare a modifier function and its inputs.

    ``inputs`` maps each extra argument's name to ``'float'`` (the entry supplies one model
    qname; the function receives that constant's default value) or ``'list'`` (a list of
    qnames; the function receives a list of floats). The function itself takes
    ``(theta, baseline, **inputs)`` and returns the target's value; it must be affine in
    theta (verified at setup).
    """
    inputs = dict(inputs or {})
    for name, kind in inputs.items():
        if kind not in MODIFIER_INPUT_TYPES:
            raise ValueError(
                f"modifier_func input {name!r} has type {kind!r}; must be one of "
                f"{list(MODIFIER_INPUT_TYPES)}.")

    def deco(fn):
        fn.is_modifier_func = True
        fn.modifier_inputs = inputs
        fn.modifier_description = (description
                                   or (fn.__doc__ or '').strip().split('\n')[0]
                                   or fn.__name__)
        return fn
    return deco


# ---------------------------------------------------------------------------- built-ins

@modifier_func(
    inputs={},
    description="one calibrated multiplier applied to every target's default value")
def scale(theta, baseline):
    return theta * baseline


@modifier_func(
    inputs={'subtract': 'list'},
    description="target = theta - sum(subtract): calibrate a total, derive the target as "
                "the remainder (e.g. q_lv_init = q_tot - q_other_inits, issue #383)")
def remainder(theta, baseline, subtract):
    return theta - sum(subtract)


BUILTIN_MODIFIER_FUNCS = {'scale': scale, 'remainder': remainder}


# ------------------------------------------------------------------------------- loading

def _collect_decorated(module):
    return {name: obj for name, obj in vars(module).items()
            if callable(obj) and getattr(obj, 'is_modifier_func', False)
            and not name.startswith('_')}


@functools.lru_cache(maxsize=8)
def _external_funcs_cached(abspath):
    """Load-and-collect an external modifier-funcs file once per process.

    ``expand_modifier_param_vals`` looks the registry up on every cost evaluation, and
    ``_load_module_from_path`` re-executes the file each call -- uncached, an external-path
    calibration would hit the disk and re-exec user code thousands of times per run. Cached by
    absolute path: edits to the file during a running process are not picked up, the same
    resolve-once semantics as the baselines.
    """
    return _collect_decorated(_load_module_from_path(abspath))


def get_modifier_funcs(external_path=None):
    """The full modifier-function registry: built-ins, then the shipped
    ``libcuflynx.funcs.modifier_funcs_user``, then external.

    Later sources win on a name clash, matching the operation/cost func registries: a user
    redefining ``scale`` gets their version, deliberately. Only functions carrying the
    :func:`modifier_func` decorator are collected -- an undecorated helper in the user file is
    ignored rather than half-registered without input metadata. ``external_path`` may be
    falsy (no-op) and raises ``FileNotFoundError`` when set but missing, exactly as
    ``register_funcs_from_file`` does.
    """
    funcs = dict(BUILTIN_MODIFIER_FUNCS)
    funcs.update(_collect_decorated(
        importlib.import_module('libcuflynx.funcs.modifier_funcs_user')))
    if external_path:
        if not os.path.exists(external_path):
            raise FileNotFoundError(
                f"modifier_funcs_external_path not found: {external_path}")
        funcs.update(_external_funcs_cached(os.path.abspath(external_path)))
    return funcs


# ------------------------------------------------------------------------- affine probe

def probe_affine(fn, baseline, resolved_inputs, modifier_name, rel_tol=1e-9):
    """``(a, b)`` of ``fn(theta, baseline, **inputs) = a*theta + b``, or raise if not affine.

    Numeric, not symbolic, so it works for any user function: evaluate at theta = 0, 1, 2 and
    require the second difference to vanish. The probe runs once at setup with the resolved
    inputs, so a function that is affine for the actual inputs passes even if it could be
    non-affine for others -- which is the property the gradients and the x0 inversion need.
    """
    f0 = float(fn(0.0, baseline, **resolved_inputs))
    f1 = float(fn(1.0, baseline, **resolved_inputs))
    f2 = float(fn(2.0, baseline, **resolved_inputs))
    a = f1 - f0
    scale_ref = max(abs(f0), abs(f1), abs(f2), 1.0)
    if abs(f2 - (2.0 * f1 - f0)) > rel_tol * scale_ref:
        raise NotImplementedError(
            f"modifier function {modifier_name!r} is not affine in theta (probed "
            f"f(0)={f0!r}, f(1)={f1!r}, f(2)={f2!r}). The analytic gradients apply a "
            f"constant chain-rule weight dp/dtheta and theta's starting value is derived by "
            f"inverting the mapping, both of which require p = a*theta + b. Rewrite the "
            f"function to be affine in theta, or calibrate the target directly.")
    return a, f0
