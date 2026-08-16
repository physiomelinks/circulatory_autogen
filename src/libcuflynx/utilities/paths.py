"""Where libcuflynx reads user inputs from, and where it writes user outputs to.

Historically a dozen modules computed "the repo root" from their own location
(``os.path.join(os.path.dirname(__file__), '../../..')``) and hung ``resources/``,
``generated_models/``, ``param_id_output/`` and friends off it. That is correct only
while the code is being run out of a checkout. Once ``libcuflynx`` is installed, the
same expression resolves inside ``site-packages`` — which is a meaningless place to
look for a user's model inputs, and (worse) a writable one, so the mistake does not
even announce itself: it just scatters per-run artefacts through the installed
package (issue #431).

This module is the single answer to "which directory did you mean?", and it
distinguishes three questions that were previously conflated:

``package_dir()``
    The installed package itself. Read-only: package *data* lives here, nothing else.
``repo_root()``
    The circulatory_autogen checkout this source file lives in, or ``None`` when the
    package is installed rather than run from a checkout. Detected by markers on
    disk, never assumed.
``user_data_root()``
    The directory user inputs/outputs default under when the config names none: the
    checkout when there is one (so the developer workflow is unchanged), otherwise
    the process's current working directory. Overridable with ``$CUFLYNX_USER_DIR``.

Prefer a config-supplied path (``resources_dir``, ``generated_models_dir``,
``param_id_output_dir``, ``external_modules_dir``) over any of these. The defaults
here exist only for the case where the config supplies nothing.
"""

import os

__all__ = [
    'CUFLYNX_USER_DIR_ENV_VAR',
    'package_dir',
    'repo_root',
    'user_data_root',
    'default_resources_dir',
    'default_generated_models_dir',
    'default_param_id_output_dir',
    'default_sensitivity_outputs_dir',
    'default_user_inputs_dir',
    'default_module_config_user_dir',
    'default_funcs_user_dir',
]

# Set this to run against inputs/outputs somewhere other than the cwd without editing
# a config. It is the "explicit argument" escape hatch for a pip-installed run.
CUFLYNX_USER_DIR_ENV_VAR = 'CUFLYNX_USER_DIR'

# .../libcuflynx  (this file is .../libcuflynx/utilities/paths.py)
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What has to be present next to a ``src/`` directory for it to be *this* project's
# checkout rather than an arbitrary directory that happens to sit above the package.
# Both must exist, so an install into ``<somewhere>/src/libcuflynx`` cannot be mistaken
# for a checkout.
_REPO_MARKERS = ('pyproject.toml', 'user_run_files')


def package_dir():
    """The installed ``libcuflynx`` package directory. For reading package data only."""
    return _PACKAGE_DIR


def repo_root():
    """The circulatory_autogen checkout containing this source, or ``None``.

    ``None`` is the normal answer for a ``pip install libcuflynx``; a checkout (including
    an editable install, whose ``__file__`` still points into the working tree) answers
    with its root. Recomputed per call rather than cached at import so a test that
    relocates the tree is not answered from a stale value.
    """
    parent = os.path.dirname(_PACKAGE_DIR)
    if os.path.basename(parent) != 'src':
        return None
    candidate = os.path.dirname(parent)
    if all(os.path.exists(os.path.join(candidate, marker)) for marker in _REPO_MARKERS):
        return candidate
    return None


def user_data_root():
    """The directory user inputs/outputs default under. Never inside the package.

    ``$CUFLYNX_USER_DIR`` wins if set; then the checkout, if this is one; then the cwd.
    """
    override = os.environ.get(CUFLYNX_USER_DIR_ENV_VAR)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    root = repo_root()
    if root is not None:
        return root
    return os.getcwd()


def default_resources_dir():
    """Default ``resources_dir`` — model inputs, and where the dated run config is archived."""
    return os.path.join(user_data_root(), 'resources')


def default_generated_models_dir():
    """Default ``generated_models_dir`` — generation output."""
    return os.path.join(user_data_root(), 'generated_models')


def default_param_id_output_dir():
    """Default ``param_id_output_dir`` — calibration output."""
    return os.path.join(user_data_root(), 'param_id_output')


def default_sensitivity_outputs_dir():
    """Default parent of the per-model ``sa_options['output_dir']``."""
    return os.path.join(user_data_root(), 'sensitivity_outputs')


def default_user_inputs_dir():
    """Where ``user_inputs.yaml`` is looked for when no path override is given."""
    return os.path.join(user_data_root(), 'user_run_files')


def default_module_config_user_dir():
    """The user's CellML module library.

    Still a plain directory in the checkout; #432 will ship the built-in module
    libraries as package data, after which this stays the *user's* half of the split.
    Callers must tolerate it not existing — in an install it usually will not.
    """
    return os.path.join(user_data_root(), 'module_config_user')


def default_funcs_user_dir():
    """The ``funcs_user/`` directory of user cost/operation/modifier funcs.

    The built-in funcs still live here and are imported from it by bare name; #433
    moves those into the package and leaves this purely a user extension point.
    Callers must tolerate it not existing.
    """
    return os.path.join(user_data_root(), 'funcs_user')
