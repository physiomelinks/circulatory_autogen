"""``cuflynx-init``: write a starter study into a directory.

The package ships no user inputs of its own, so after ``pip install libcuflynx`` there is
nothing for ``cuflynx-generate`` to read. This writes ``user_run_files/user_inputs.yaml`` and
the 3compartment model's resources (module array, parameters, params_for_id, obs_data) into a
directory, from which every stage command then runs unchanged.

Existing files are never overwritten unless ``--force`` is given.
"""
import argparse
import os
import sys

from libcuflynx.utilities.package_resources import package_data_file
from libcuflynx.utilities.paths import CUFLYNX_USER_DIR_ENV_VAR, user_data_root

#: Starter files, as (file name in the package's ``scripts/starter/``, path written under the
#: target directory).
STARTER_FILES = (
    ('user_inputs.yaml', os.path.join('user_run_files', 'user_inputs.yaml')),
    ('3compartment_module_array.csv', os.path.join('resources', '3compartment_module_array.csv')),
    ('3compartment_parameters.json', os.path.join('resources', '3compartment_parameters.json')),
    ('3compartment_params_for_id.json',
     os.path.join('resources', '3compartment_params_for_id.json')),
    ('3compartment_obs_data.json', os.path.join('resources', '3compartment_obs_data.json')),
)


def write_starter(directory, force=False):
    """Write the starter files under *directory*; return the paths written.

    Raises ``FileExistsError`` naming every file that is already there, before writing any,
    unless *force*.
    """
    directory = os.path.abspath(directory)
    targets = [(name, os.path.join(directory, relative)) for name, relative in STARTER_FILES]
    existing = [path for _, path in targets if os.path.exists(path)]
    if existing and not force:
        raise FileExistsError(
            'cuflynx-init would overwrite ' + ', '.join(existing) + '. Choose another '
            'directory, or pass --force to replace them.')
    for name, path in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = package_data_file('libcuflynx.scripts', 'starter', name).read_bytes()
        with open(path, 'wb') as f:
            f.write(data)
    return [path for _, path in targets]


def next_steps(directory):
    """What to run after writing the starter into *directory*."""
    directory = os.path.abspath(directory)
    lines = []
    if os.path.abspath(user_data_root()) != directory:
        lines += [f'The user directory is {user_data_root()} here; point it at the study first:',
                  f'    export {CUFLYNX_USER_DIR_ENV_VAR}={directory}']
    elif os.path.abspath(os.getcwd()) != directory:
        lines += [f'    cd {directory}']
    lines += ['    cuflynx-generate                # generate the 3compartment model',
              '    mpiexec -n 2 cuflynx-param-id   # generate, then calibrate it',
              f'Edit {os.path.join(directory, "user_run_files", "user_inputs.yaml")} to change '
              'the model, solver or calibration.']
    return '\n'.join(lines)


def main(argv=None):
    """Entry point for the ``cuflynx-init`` command."""
    parser = argparse.ArgumentParser(
        description='Write a starter study: user_run_files/user_inputs.yaml and the '
                    '3compartment model\'s resources, ready for cuflynx-generate and '
                    'cuflynx-param-id.')
    parser.add_argument('directory', nargs='?', default='.',
                        help='where to write the study (default: the current directory)')
    parser.add_argument('--force', action='store_true',
                        help='replace starter files that are already there')
    args = parser.parse_args(argv)
    try:
        written = write_starter(args.directory, force=args.force)
    except FileExistsError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    print('Wrote:\n' + '\n'.join(f'    {path}' for path in written))
    print('Next:\n' + next_steps(args.directory))
    return 0


if __name__ == '__main__':
    sys.exit(main())
