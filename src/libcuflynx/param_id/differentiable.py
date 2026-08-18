"""Marking and checking functions that are safe for CasADi symbolic execution."""

CIRCULATORY_DIFFERENTIABLE = "_circulatory_differentiable"


def differentiable(func):
    setattr(func, CIRCULATORY_DIFFERENTIABLE, True)
    return func


def is_circulatory_differentiable(func):
    return getattr(func, CIRCULATORY_DIFFERENTIABLE, False)


def assert_casadi_differentiable(obs_info, cost_type, operation_funcs_dict, cost_funcs_dict):
    """
    Raise ValueError if any operation or cost referenced by obs_info / cost_type
    is not marked with @differentiable.
    """
    if obs_info is not None:
        ops = obs_info.get("operations") or []
        for name in dict.fromkeys(ops):
            if name is None:
                continue
            fn = operation_funcs_dict.get(name)
            if fn is None:
                raise ValueError(f"Unknown operation {name!r} in obs_info['operations'].")
            if not is_circulatory_differentiable(fn):
                raise ValueError(
                    f"Operation {name!r} is used in casadi_python mode but is not marked "
                    f"@differentiable (not safe for symbolic execution)."
                )
    if cost_type is not None and cost_funcs_dict is not None:
        raw = cost_type if isinstance(cost_type, (list, tuple)) else [cost_type]
        names = list(dict.fromkeys(raw))
        for name in names:
            if name is None:
                continue
            fn = cost_funcs_dict.get(name)
            if fn is None:
                raise ValueError(f"Unknown cost function {name!r} in cost_type.")
            if not is_circulatory_differentiable(fn):
                raise ValueError(
                    f"Cost function {name!r} is used in casadi_python mode but is not marked "
                    f"@differentiable."
                )


def assert_mle_cost_for_bayesian(cost_type, cost_funcs_dict, context):
    """
    MCMC and Laplace use ``ln L = -cost`` in paramID; ``cost`` must be an MLE / negative
    log-likelihood-style objective (functions decorated with ``@is_MLE``).

    Raises:
        ValueError: if any referenced cost is missing or lacks ``is_MLE``.
    """
    if cost_type is None or cost_funcs_dict is None:
        return
    raw = cost_type if isinstance(cost_type, (list, tuple)) else [cost_type]
    for name in dict.fromkeys(raw):
        if name is None:
            continue
        fn = cost_funcs_dict.get(name)
        if fn is None:
            raise ValueError(f"Unknown cost function {name!r} in cost_type (needed for {context}).")
        if not getattr(fn, "is_MLE", False):
            raise ValueError(
                f"Cost function {name!r} is not marked @is_MLE but {context} requires an MLE-style "
                f"cost (e.g. gaussian_MLE) so that ln L = -cost is valid."
            )
