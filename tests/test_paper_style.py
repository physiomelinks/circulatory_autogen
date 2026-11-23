"""libcuflynx's paper style applies to its own figures only: importing libcuflynx leaves
matplotlib.rcParams alone, and its plot functions draw inside ``paper_style()``."""
import json
import os
import subprocess
import sys

import matplotlib
import pytest

from libcuflynx.utilities.paperPlotSetup import paper_style, paper_style_params

pytestmark = pytest.mark.unit

# Modules that set the paper style globally at import before it moved into the plot functions.
_PLOTTING_MODULES = [
    'libcuflynx.param_id.paramID',
    'libcuflynx.param_id.plot_outputs',
    'libcuflynx.sensitivity_analysis.sensitivityAnalysis',
    'libcuflynx.identifiabilty_analysis.identifiabilityAnalysis',
    'libcuflynx.utilities.plot_generated_model_outputs',
    'libcuflynx.scripts.heart_paper_plots',
]

# Imports every libcuflynx module in a fresh interpreter and prints, as JSON, the rcParams that
# changed (the backend aside: libcuflynx picks Agg for scripts with no backend chosen, see
# test_matplotlib_backend.py), the modules imported and those an optional dependency kept out.
_IMPORT_ALL = r'''
import importlib, json, pkgutil, matplotlib
before = {k: repr(v) for k, v in matplotlib.rcParams.items()}
import libcuflynx
imported, skipped = [], {}
for info in pkgutil.walk_packages(libcuflynx.__path__, 'libcuflynx.'):
    try:
        importlib.import_module(info.name)
        imported.append(info.name)
    except Exception as exc:
        skipped[info.name] = repr(exc)
after = {k: repr(v) for k, v in matplotlib.rcParams.items()}
changed = sorted(k for k in after if k != 'backend' and after[k] != before.get(k))
print(json.dumps({'changed': changed, 'imported': imported, 'skipped': skipped}))
'''


def test_importing_libcuflynx_leaves_rcparams_unchanged():
    env = {k: v for k, v in os.environ.items() if k != 'MPLBACKEND'}
    result = subprocess.run([sys.executable, '-c', _IMPORT_ALL], env=env, capture_output=True,
                            text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    missing = [m for m in _PLOTTING_MODULES if m not in report['imported']]
    assert not missing, {m: report['skipped'].get(m) for m in missing}
    assert report['changed'] == []


def test_a_plot_function_draws_in_the_paper_style_and_restores_the_callers_settings():
    seen = {}

    @paper_style()
    def plot():
        seen.update({k: matplotlib.rcParams[k] for k in ('font.family', 'xtick.direction', 'font.size')})

    with matplotlib.rc_context({'font.family': 'sans-serif', 'xtick.direction': 'out',
                                'font.size': 10.0}):
        plot()
        assert matplotlib.rcParams['font.family'] == ['sans-serif']
        assert matplotlib.rcParams['xtick.direction'] == 'out'
        assert matplotlib.rcParams['font.size'] == 10.0
    assert seen == {'font.family': ['serif'], 'xtick.direction': 'in', 'font.size': 14.0}


def test_a_plot_function_restores_the_callers_settings_when_it_raises():
    @paper_style()
    def plot():
        raise RuntimeError('boom')

    with matplotlib.rc_context({'font.size': 10.0}):
        with pytest.raises(RuntimeError):
            plot()
        assert matplotlib.rcParams['font.size'] == 10.0


@pytest.mark.parametrize('case, width, font_size', [(1, 3.8, 8), (2, 6.9, 20), (3, 17.4 / 2.54, 14)])
def test_setup_plot_applies_the_same_style_globally(case, width, font_size):
    from libcuflynx.utilities.paperPlotSetup import Setup_Plot

    with matplotlib.rc_context():
        Setup_Plot(case)
        for key, value in paper_style_params(case).items():
            assert matplotlib.rcParams[key] == matplotlib.rcParams.validate[key](value), key
        assert matplotlib.rcParams['font.size'] == font_size
        assert matplotlib.rcParams['figure.figsize'][0] == pytest.approx(width)
