"""Which solver backends, at which versions, a test run used.

conftest.py prints ``backend_info()`` in the pytest header and records it in the junit file, so
every CI log says what ran; tests/run_release_checks.sh prints it too.

SUNDIALS is the one to read closely. Myokit compiles its simulations against the SUNDIALS it
was configured with, and a SUNDIALS built without CMAKE_BUILD_TYPE=Release makes Myokit several
times slower without changing a single result. The build type cannot be read back from the
library, but SUNDIALS 7 writes it into its sundials_config.h (``SUN_BUILD_TYPE``). So
``sundials_warning`` warns when that header names a build type other than Release or
RelWithDebInfo, or names none, rather than refusing the build. Older SUNDIALS (the Ubuntu
package is 6.4) records nothing and is not warned about.
"""
import importlib
import os
import platform
import shutil
import subprocess


def _version(module, attribute='__version__'):
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - absent or broken: either way, not this run's backend
        return None
    value = getattr(mod, attribute, None)
    if callable(value):
        try:
            value = value()
        except Exception:  # noqa: BLE001
            value = None
    if value is None:
        try:
            from importlib.metadata import version
            value = version(module)
        except Exception:  # noqa: BLE001 - imported, but not from a distribution
            value = 'installed'
    return str(value)


def myokit_sundials():
    """{'version', 'lib', 'build_type'}: the SUNDIALS Myokit builds its simulations against."""
    try:
        import myokit
    except Exception:  # noqa: BLE001
        return None
    try:
        version = str(myokit.Sundials.version())
    except Exception:  # noqa: BLE001 - Myokit without a working SUNDIALS
        version = 'unavailable'
    libs = [p for p in (getattr(myokit, 'SUNDIALS_LIB', None) or []) if p]
    includes = [p for p in (getattr(myokit, 'SUNDIALS_INC', None) or []) if p] + ['/usr/include']
    return {'version': version, 'lib': os.pathsep.join(libs) or 'system default',
            'build_type': sundials_build_type(includes)}


def _compiler():
    cc = os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc') or shutil.which('cl')
    if not cc:
        return None
    try:
        out = subprocess.run([cc, '--version'], capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:  # noqa: BLE001 - MSVC's cl takes no --version
        return os.path.basename(cc)


def backend_info():
    """{name: version or None} for every backend libcuflynx can run, plus SUNDIALS and cc."""
    sundials = myokit_sundials()
    if sundials:
        build = sundials['build_type'] or 'build type not recorded'
        sundials = f"{sundials['version']} ({build}) at {sundials['lib']}"
    return {
        'python': platform.python_version(),
        'myokit': _version('myokit'),
        'sundials (myokit)': sundials,
        'casadi': _version('casadi'),
        'libcellml': _version('libcellml'),
        'libopencor': _version('libopencor', 'version_string'),
        'aadc': _version('aadc'),
        'mpi4py': _version('mpi4py'),
        'scipy': _version('scipy'),
        'cmake': shutil.which('cmake') and 'installed',
        'cc': _compiler(),
    }


def sundials_build_type(include_dirs):
    """SUN_BUILD_TYPE from sundials_config.h in one of ``include_dirs``, or None if not recorded."""
    for folder in include_dirs:
        try:
            with open(os.path.join(folder, 'sundials', 'sundials_config.h')) as f:
                for line in f:
                    if line.startswith('#define SUN_BUILD_TYPE'):
                        return line.split(None, 2)[2].strip().strip('"')
        except OSError:
            continue
    return None


def sundials_warning(build_type):
    """A warning when a recorded SUNDIALS build type is not an optimised one, else None."""
    if build_type is None or build_type in ('Release', 'RelWithDebInfo', 'MinSizeRel'):
        return None
    return (f"Myokit's SUNDIALS was built with CMAKE_BUILD_TYPE={build_type or '(none)'}, not "
            'Release: Myokit runs several times slower (results are unchanged), so do not '
            'compare its timings with other machines.')


def header_lines():
    info = backend_info()
    lines = ['backends: ' + ', '.join(f'{k} {v}' for k, v in info.items() if v)]
    absent = [k for k, v in info.items() if not v]
    if absent:
        lines.append('not installed: ' + ', '.join(absent))
    sundials = myokit_sundials()
    warning = sundials and sundials_warning(sundials['build_type'])
    if warning:
        lines.append('WARNING: ' + warning)
    return lines


if __name__ == '__main__':
    import sys
    print('\n'.join(header_lines()))
    # --require a,b,c: exit 1 unless each is installed (tests/run_release_checks.sh)
    if sys.argv[1:2] == ['--require']:
        info = backend_info()
        names = {'sundials': 'sundials (myokit)'}
        missing = [b for b in sys.argv[2].split(',') if not info.get(names.get(b, b))]
        if missing:
            print('missing: ' + ', '.join(missing))
            sys.exit(1)
