"""Matplotlib backend choice for libcuflynx's figure-writing code."""
import sys

import matplotlib


def _backend_chosen():
    """True when a backend was set explicitly (MPLBACKEND, matplotlibrc, matplotlib.use)."""
    get = getattr(matplotlib.rcParams, '_get_backend_or_none', None)
    if get is not None:
        return get() is not None
    from matplotlib import rcsetup
    return dict.__getitem__(matplotlib.rcParams, 'backend') is not rcsetup._auto_backend_sentinel


def use_agg_unless_interactive():
    """Select the non-interactive Agg backend for script runs that only save figures.

    Leaves the backend alone inside IPython/Jupyter and whenever one was already chosen, so
    plt.show() in a notebook keeps working after libcuflynx is imported. Otherwise (scripts,
    MPI ranks, with or without a DISPLAY) figures render with Agg and no windows open.
    """
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules or _backend_chosen():
        return
    matplotlib.use('Agg')
