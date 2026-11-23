"""The paper style of libcuflynx's figures.

libcuflynx's plot functions draw inside ``paper_style()``, so the style applies to their figures
only and importing libcuflynx leaves ``matplotlib.rcParams`` alone. ``Setup_Plot`` applies the
same style globally, for a script that wants it on its own figures too.
"""
import matplotlib as mpl
import numpy as np


def paper_style_params(case=3):
    """The rcParams of the paper style.

    case 1 is a single column of a two-column paper, case 2 a full-width paper figure,
    case 3 (the default of libcuflynx's figures) a figure 17.4 cm wide.
    """
    assert case in [1, 2, 3]
    rc = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.family': 'serif',
        'font.serif': 'cmr10',
        'mathtext.fontset': 'dejavuserif',
        'axes.formatter.use_mathtext': True,
        'grid.color': 'xkcd:light grey',
        'legend.edgecolor': 'k',
        'legend.fancybox': False,
        'legend.framealpha': 1.0,
    }
    if case == 1:
        fig_width = 3.8
        sizes = (8, 8, 8, 8, 6, 0.8)
    elif case == 2:
        fig_width = 6.9
        sizes = (20, 16, 16, 20, 20, 1.8)
    else:
        fig_width = 17.4 / 2.54  # 17.4 cm in inches
        sizes = (14, 12, 12, 14, 12, 0.5)
    for key, value in zip(('font.size', 'xtick.labelsize', 'ytick.labelsize', 'axes.labelsize',
                           'legend.fontsize', 'lines.linewidth'), sizes):
        rc[key] = value
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    rc['figure.figsize'] = [fig_width, fig_width * golden_mean]
    return rc


def paper_style(case=3):
    """The paper style for the figures drawn inside it, as a context manager or a decorator.

    Use it around the whole of a plot function, its ``savefig`` included: Matplotlib creates
    some artists (tick labels) only when the figure is drawn.
    """
    return mpl.rc_context(paper_style_params(case))


def Setup_Plot(case):
    """Apply the paper style to every figure from here on (changes ``matplotlib.rcParams``)."""
    mpl.rcParams.update(paper_style_params(case))
