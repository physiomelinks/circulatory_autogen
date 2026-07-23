"""Load user operation/cost funcs from an external Python file (issue #303).

Downstream tools (e.g. CUFLynx) let users author custom observable-operation funcs and cost
funcs in a GUI. Rather than writing into CA's tracked ``funcs_user/operation_funcs_user.py`` /
``cost_funcs_user.py``, the user config can point CA at an *external* file via
``operation_funcs_external_path`` / ``cost_funcs_external_path``; CA loads it and registers its
top-level funcs alongside the built-ins, using the same registration convention (top-level
callables defined in that file, ``@differentiable`` / ``@series_to_constant`` / ``@is_MLE`` /
``@cost_combiner`` decorators, and a module-level ``mb`` math backend).
"""
import hashlib
import importlib.util
import os


def _load_module_from_path(path):
    """Import an arbitrary ``.py`` file as a module (module name derived from the absolute path,
    so different files get distinct namespaces and their funcs' ``__module__`` is stable)."""
    abspath = os.path.abspath(path)
    modname = "ca_external_funcs_" + hashlib.md5(abspath.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(modname, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load external funcs file: {abspath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_funcs_from_file(path, registry, backend, exclude=frozenset()):
    """Register every top-level func defined in the external file ``path`` into ``registry``.

    Mirrors ``operation_funcs_user.register_user_operations`` / ``cost_funcs_user.register_cost_funcs``:
    binds the file's module-level ``mb`` to ``backend`` (so backend-dependent funcs use the active
    numpy/casadi backend), then registers each top-level callable whose ``__module__`` is that file
    (imported decorators such as ``differentiable`` are skipped by the ``__module__`` check), except
    private names (leading ``_``) and any ``exclude`` names (decorator helpers the file may define).

    No-op when ``path`` is falsy. Raises ``FileNotFoundError`` if a path is given but missing.
    """
    if not path:
        return registry
    if not os.path.exists(path):
        raise FileNotFoundError(f"External funcs file not found: {path}")
    module = _load_module_from_path(path)
    # Bind the file's math backend, exactly as the built-in register hooks do.
    if hasattr(module, "mb"):
        module.mb = backend
    modname = module.__name__
    for name, obj in vars(module).items():
        if name.startswith("_") or name in exclude:
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != modname:
            continue
        registry[name] = obj
    return registry
