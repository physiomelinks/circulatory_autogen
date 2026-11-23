"""Importing libcuflynx's plotting modules keeps the matplotlib backend the user chose."""
import os
import subprocess
import sys

import pytest

_MODULES = [
    'libcuflynx.param_id.paramID',
    'libcuflynx.sensitivity_analysis.sensitivityAnalysis',
    'libcuflynx.identifiabilty_analysis.identifiabilityAnalysis',
    'libcuflynx.utilities.plot_generated_model_outputs',
]


def _run(code, env_extra=None, drop=()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)


def _chosen_backend():
    try:
        import matplotlib_inline  # noqa: F401
        return 'module://matplotlib_inline.backend_inline'
    except ImportError:
        return 'svg'


@pytest.mark.parametrize('module', _MODULES)
def test_import_keeps_a_backend_set_beforehand(module):
    """A notebook sets the inline backend before importing libcuflynx; plt.show() then needs it."""
    backend = _chosen_backend()
    code = (f"import importlib, matplotlib\n"
            f"matplotlib.use({backend!r})\n"
            f"importlib.import_module({module!r})\n"
            f"print(matplotlib.get_backend())\n")
    result = _run(code, drop=('MPLBACKEND',))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == backend


def test_scripts_without_a_chosen_backend_use_agg():
    """With no backend chosen and not in IPython (a script or MPI rank, DISPLAY set or not),
    figures render with Agg."""
    code = ("import matplotlib\n"
            "from libcuflynx.utilities.mpl_backend import use_agg_unless_interactive\n"
            "use_agg_unless_interactive()\n"
            "import matplotlib.pyplot as plt\n"
            "plt.figure()\n"
            "print(matplotlib.get_backend())\n")
    result = _run(code, env_extra={'DISPLAY': ':0'}, drop=('MPLBACKEND',))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1].lower() == 'agg'
