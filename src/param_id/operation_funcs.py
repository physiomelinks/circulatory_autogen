import numpy as np
import casadi as ca

OPERATION_FUNCS = {
    "numpy": {},
    "casadi": {}
}

def operation(mode="numpy"):
    def wrapper(func):
        if mode == "both":
            OPERATION_FUNCS["numpy"][func.__name__] = func
            OPERATION_FUNCS["casadi"][func.__name__] = func
        else:
            OPERATION_FUNCS[mode][func.__name__] = func
        return func
    return wrapper

def series_to_constant(func):
    func.series_to_constant = True
    return func

@operation(mode="numpy")
@series_to_constant
def max(x, series_output=False):
    if series_output:
        return x
    else:
        return np.max(x)

@operation(mode="numpy")
@series_to_constant
def min(x, series_output=False):
    if series_output:
        return x
    else:
        return np.min(x)

@operation(mode="numpy")
@series_to_constant
def mean(x, series_output=False):
    if series_output:
        return x
    else:
        return np.mean(x)

@operation(mode="numpy")
@series_to_constant
def max_minus_min(x, series_output=False):
    if series_output:
        return x
    else:
        return np.max(x) - np.min(x)
    

@operation(mode="casadi")
@series_to_constant
def max(x, series_output=False):
    if series_output:
        return x
    else:
        return ca.mmax(x)

@operation(mode="casadi")
@series_to_constant
def min(x, series_output=False):
    if series_output:
        return x
    else:
        return ca.mmin(x)

@operation(mode="casadi")
@series_to_constant
def mean(x, series_output=False):
    if series_output:
        return x
    else:
        return ca.mmean(x)

@operation(mode="casadi")
@series_to_constant
def max_minus_min(x, series_output=False):
    if series_output:
        return x
    else:
        return ca.max(x) - ca.min(x)


@operation(mode="both")
def addition(x1, x2):
    return x1 + x2

@operation(mode="both")
def subtraction(x1, x2):
    return x1 - x2

@operation(mode="both")
def multiplication(x1, x2):
    return x1 * x2

@operation(mode="both")
def division(x1, x2):
    return x1 / x2
