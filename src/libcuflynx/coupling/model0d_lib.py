'''Load (and if needed build) the C interface of a generated C++ 0D model with ctypes.

The interface (``model0d_capi.cpp``, from templates/model0d_capi.cpp.j2) describes itself:
the number, names, units, directions and sizes of the exchanged variables come from the
library, so this module is the same for every model.
'''

import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from libcuflynx.generators.cpp.api import CAPI_VERSION

INFO_FILE = 'external_models.json'
LIB_TARGET = 'model0d_capi'
DIRECTIONS = {0: 'to_external', 1: 'from_external'}


class Model0dError(RuntimeError):
    '''A failure reported by the generated 0D model (e.g. CVODE), or in loading it.'''


def load_info(model_dir):
    path = os.path.join(model_dir, INFO_FILE)
    if not os.path.isfile(path):
        raise Model0dError(f'{path} not found: generate the model with model_type: cpp from a module array '
                           f'that has a python external model (api transport "python").')
    with open(path) as f:
        return json.load(f)


def _library_path(build_dir):
    for pattern in ('lib' + LIB_TARGET + '.so', 'lib' + LIB_TARGET + '.dylib', LIB_TARGET + '.dll'):
        found = glob.glob(os.path.join(build_dir, '**', pattern), recursive=True)
        if found:
            return found[0]
    return None


def _sources_newer_than(model_dir, path):
    mtime = os.path.getmtime(path)
    for pattern in ('*.c', '*.cpp', '*.h', 'CMakeLists.txt'):
        for src in glob.glob(os.path.join(model_dir, pattern)):
            if os.path.getmtime(src) > mtime:
                return True
    return False


def build_library(model_dir, build_dir=None, cmake_args=(), verbose=False):
    '''Configure and build the model's shared library with CMake; return its path.

    SUNDIALS (for CVODE models) is found from ``SUNDIALS_DIR``. Inside a conda environment
    the environment's prefix is searched first, so the library uses the same C++ runtime and
    SUNDIALS as the Python that loads it (building against the system's instead is the usual
    cause of "GLIBCXX_... not found" when the library is loaded).'''
    if shutil.which('cmake') is None:
        raise Model0dError('cmake not found: install CMake (in a conda environment: '
                           'mamba install cmake cxx-compiler sundials) to build the 0D model.')
    build_dir = build_dir or os.path.join(model_dir, 'build')
    args = ['cmake', '-S', model_dir, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release', *cmake_args]
    if os.environ.get('SUNDIALS_DIR'):
        args.append(f"-DSUNDIALS_DIR={os.environ['SUNDIALS_DIR']}")
    if os.environ.get('CONDA_PREFIX'):
        args.append(f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}")
    for cmd in (args, ['cmake', '--build', build_dir, '--target', LIB_TARGET, '-j']):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if verbose or proc.returncode != 0:
            sys.stdout.write(proc.stdout[-6000:])
        if proc.returncode != 0:
            raise Model0dError(f"{' '.join(cmd)} failed:\n{proc.stderr[-4000:]}")
    path = _library_path(build_dir)
    if path is None:
        raise Model0dError(f'built, but no {LIB_TARGET} library in {build_dir}')
    return path


class Model0dLibrary:
    '''The C interface of one generated model: builds it if missing or out of date, loads it,
    and reads its exchange table.'''

    def __init__(self, model_dir, build=True, build_dir=None, verbose=False):
        self.model_dir = os.path.abspath(model_dir)
        self.info = load_info(self.model_dir)
        if self.info.get('capi_version') != CAPI_VERSION:
            raise Model0dError(f"{INFO_FILE} was written for C interface version {self.info.get('capi_version')}, "
                               f"this libcuflynx uses {CAPI_VERSION}: regenerate the model.")
        build_dir = build_dir or os.path.join(self.model_dir, 'build')
        path = _library_path(build_dir)
        if build and (path is None or _sources_newer_than(self.model_dir, path)):
            path = build_library(self.model_dir, build_dir, verbose=verbose)
        if path is None:
            raise Model0dError(f'No {LIB_TARGET} library in {build_dir}: build it '
                               f'(cmake -S {self.model_dir} -B {build_dir} && cmake --build {build_dir}).')
        self.path = path
        try:
            # RTLD_LOCAL (the default): the model's own symbols stay private to this library, so
            # several models can be loaded in one process.
            lib = ctypes.CDLL(path)
        except OSError as e:
            hint = ''
            if 'GLIBCXX' in str(e) or 'CXXABI' in str(e):
                hint = ('\nThe library was built with a newer C++ compiler than the one this Python uses. '
                        'In a conda environment build with its compiler (mamba install cxx-compiler cmake '
                        'sundials) after deleting the build folder.')
            raise Model0dError(f'Cannot load {path}: {e}{hint}') from e
        self._lib = lib
        self._declare()
        if lib.cf_api_version() != CAPI_VERSION:
            raise Model0dError(f'{path} implements C interface version {lib.cf_api_version()}, this libcuflynx '
                               f'uses {CAPI_VERSION}: rebuild it from a regenerated model.')
        self.exchange = []
        for i in range(lib.cf_n_exchange()):
            self.exchange.append({
                'index': i,
                'name': lib.cf_exchange_name(i).decode(),
                'units': lib.cf_exchange_units(i).decode(),
                'direction': DIRECTIONS[lib.cf_exchange_direction(i)],
                'size': lib.cf_exchange_size(i),
            })

    def _declare(self):
        lib = self._lib
        c_double_p = ctypes.POINTER(ctypes.c_double)
        sigs = {
            'cf_api_version': ([], ctypes.c_int),
            'cf_model_name': ([], ctypes.c_char_p),
            'cf_last_error': ([], ctypes.c_char_p),
            'cf_n_exchange': ([], ctypes.c_int),
            'cf_exchange_name': ([ctypes.c_int], ctypes.c_char_p),
            'cf_exchange_units': ([ctypes.c_int], ctypes.c_char_p),
            'cf_exchange_direction': ([ctypes.c_int], ctypes.c_int),
            'cf_exchange_size': ([ctypes.c_int], ctypes.c_int),
            'cf_create': ([ctypes.c_char_p], ctypes.c_void_p),
            'cf_destroy': ([ctypes.c_void_p], None),
            'cf_time': ([ctypes.c_void_p], ctypes.c_double),
            'cf_get': ([ctypes.c_void_p, ctypes.c_int, c_double_p, ctypes.c_int], ctypes.c_int),
            'cf_set': ([ctypes.c_void_p, ctypes.c_int, c_double_p, ctypes.c_int], ctypes.c_int),
            'cf_step': ([ctypes.c_void_p, ctypes.c_double], ctypes.c_int),
            'cf_snapshot': ([ctypes.c_void_p], ctypes.c_int),
            'cf_restore': ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
            'cf_drop_snapshot': ([ctypes.c_void_p], ctypes.c_int),
            'cf_open_output': ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
            'cf_write_output': ([ctypes.c_void_p], ctypes.c_int),
            'cf_lookup': ([ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            'cf_value': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_double),
        }
        for name, (argtypes, restype) in sigs.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype

    def error(self):
        return self._lib.cf_last_error().decode()

    def create(self, solver=None):
        '''A new instance of the 0D model, at its initial state.'''
        handle = self._lib.cf_create((solver or '').encode())
        if not handle:
            raise Model0dError(self.error())
        return Model0d(self, handle)


class Model0d:
    '''One instance of the generated 0D model, behind its C interface.'''

    def __init__(self, library, handle):
        self.library = library
        self._lib = library._lib
        self._h = ctypes.c_void_p(handle)
        self.exchange = {x['name']: x for x in library.exchange}

    def _check(self, status):
        if status != 0:
            raise Model0dError(self.library.error())

    @property
    def time(self):
        return self._lib.cf_time(self._h)

    def get(self, name):
        x = self.exchange[name]
        buf = (ctypes.c_double * x['size'])()
        self._check(self._lib.cf_get(self._h, x['index'], buf, x['size']))
        return np.array(buf[:], dtype=float)

    def set(self, name, values):
        x = self.exchange[name]
        if x['direction'] != 'from_external':
            raise Model0dError(f"{name} is computed by the 0D model (to_external): it can't be set.")
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 1 and x['size'] > 1:
            arr = np.full(x['size'], float(arr[0]))
        if arr.size != x['size']:
            raise Model0dError(f"{name} takes {x['size']} value(s) (one per connected 0D module), got {arr.size}")
        buf = (ctypes.c_double * x['size'])(*arr)
        self._check(self._lib.cf_set(self._h, x['index'], buf, x['size']))

    def step(self, dt):
        self._check(self._lib.cf_step(self._h, float(dt)))

    def snapshot(self):
        self._check(self._lib.cf_snapshot(self._h))

    def restore(self, keep=False):
        self._check(self._lib.cf_restore(self._h, 1 if keep else 0))

    def drop_snapshot(self):
        self._check(self._lib.cf_drop_snapshot(self._h))

    def open_output(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._check(self._lib.cf_open_output(self._h, os.path.join(directory, '').encode()))

    def write_output(self):
        self._check(self._lib.cf_write_output(self._h))

    def lookup(self, name):
        '''(index, is_state) of a 0D state or variable named "vessel/variable".'''
        is_state = ctypes.c_int(0)
        index = self._lib.cf_lookup(name.encode(), ctypes.byref(is_state))
        if index < 0:
            raise Model0dError(self.library.error())
        return index, bool(is_state.value)

    def value(self, ref):
        index, is_state = ref
        return self._lib.cf_value(self._h, index, 1 if is_state else 0)

    def close(self):
        if self._h is not None:
            self._lib.cf_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
