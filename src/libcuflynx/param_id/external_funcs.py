"""Load user operation/cost funcs from an external Python file (issue #303).

Downstream tools (e.g. CUFLynx) let users author custom observable-operation funcs and cost
funcs in a GUI. Since #433 the built-in operation/cost funcs ship inside the package
(``libcuflynx.funcs``) and are not editable, so an external file is *the* way to add your own:
the user config points CA at one via
``operation_funcs_external_path`` / ``cost_funcs_external_path``; CA loads it and registers its
top-level funcs alongside the built-ins, using the same registration convention (top-level
callables defined in that file, ``@differentiable`` / ``@series_to_constant`` / ``@is_MLE`` /
``@cost_combiner`` decorators, and a module-level ``mb`` math backend).
"""
import hashlib
import importlib
import importlib.util
import os

from libcuflynx.param_id.math_backend import bind_backend


# The built-in funcs modules, which an external file imports as ``libcuflynx.funcs.<name>``.
_PACKAGED_FUNCS_MODULES = ('cost_funcs_user', 'operation_funcs_user', 'modifier_funcs_user')


def _load_module_from_path(path):
    """Import an arbitrary ``.py`` file as a module (module name derived from the absolute path,
    so different files get distinct namespaces and their funcs' ``__module__`` is stable)."""
    abspath = os.path.abspath(path)
    modname = "ca_external_funcs_" + hashlib.md5(abspath.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(modname, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load external funcs file: {abspath}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name in _PACKAGED_FUNCS_MODULES:
            raise ImportError(f"{abspath} imports {exc.name!r}; write "
                              f"'from libcuflynx.funcs.{exc.name} import ...' instead.") from exc
        raise
    return module


def register_funcs_from_file(path, registry, backend, exclude=frozenset()):
    """Register every top-level func defined in the external file ``path`` into ``registry``.

    Mirrors ``operation_funcs_user.register_user_operations`` / ``cost_funcs_user.register_cost_funcs``:
    registers each top-level callable whose ``__module__`` is that file, bound to ``backend``
    (imported decorators such as ``differentiable`` are skipped by the ``__module__`` check), except
    private names (leading ``_``) and any ``exclude`` names (decorator helpers the file may define).

    No-op when ``path`` is falsy. Raises ``FileNotFoundError`` if a path is given but missing.
    """
    if not path:
        return registry
    if not os.path.exists(path):
        raise FileNotFoundError(f"External funcs file not found: {path}")
    module = _load_module_from_path(path)
    modname = module.__name__
    for name, obj in vars(module).items():
        if name.startswith("_") or name in exclude:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != modname:
            continue
        # Bind each func to this backend so registries for different backends stay independent
        # even though the funcs read the file's module-level ``mb`` (see bind_backend, #315).
        registry[name] = bind_backend(obj, backend)
    return registry
