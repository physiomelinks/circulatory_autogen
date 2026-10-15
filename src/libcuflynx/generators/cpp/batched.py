"""C for the BE_batched_c and CVODES_batched_c solvers: libCellML's C for the whole model, plus the kernels of
:mod:`libcuflynx.generators.kernel_ir` and a runtime, built as one shared library with CMake.

Files written to the build folder (templates in ``templates/batched_*.j2``):

=========================  ===========================================================================
``model0d_core.c/.h``      libCellML's C, one variables array (as ``model_type: cpp`` writes it):
                           initial values, computed constants, and the algebraic variables for outputs
``batched_kernels.c/.h``   each kernel's rates and exact Jacobian, over a block's own states and inputs
``batched_runtime.c``      the integrator, OpenMP over blocks: BE_batched_c's (adaptive backward Euler or
                           TR-BDF2, modified Newton; ``batched_runtime.c.j2``) or CVODES_batched_c's (one CVODES
                           per block; ``batched_cvodes_runtime.c.j2``, which extends it and needs SUNDIALS 6+).
                           Both export the same functions, so a caller can swap them.
``CMakeLists.txt``         the library ``batched_model``
``blocks.json``            the blocks: kernel, model state indices, model variable indices of its inputs
=========================  ===========================================================================

The folder is named after a digest of its sources, so a model already built is loaded, not rebuilt.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import tempfile
import time

import numpy as np

from libcuflynx.generators import kernel_ir
from libcuflynx.generators.cpp.generator import core_code, template_environment

TARGET = 'batched_model'
#: The runtime's integration methods, in the order its options number them.
METHODS = ('backward_euler', 'trbdf2')
#: The solvers this module builds, with each one's runtime template and methods (CVODES_batched_c's BDF is the
#: CVODE method of the other CVODE solvers).
SOLVERS = {
    'BE_batched_c': {'runtime': 'batched_runtime.c.j2', 'methods': METHODS},
    'CVODES_batched_c': {'runtime': 'batched_cvodes_runtime.c.j2', 'methods': ('CVODE',)},
}

_C_FUNCTIONS = {
    'sqrt': 'sqrt', 'exp': 'exp', 'ln': 'log', 'log10': 'log10', 'abs': 'fabs', 'sign': 'bk_sign',
    'floor': 'floor', 'ceil': 'ceil', 'trunc': 'trunc', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
    'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'pow': 'pow', 'min': 'bk_min', 'max': 'bk_max',
    'rem': 'fmod',
}
_C_INFIX = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=',
            'eq': '==', 'ne': '!=', 'and': '&&', 'or': '||'}


class BatchedBuildError(RuntimeError):
    pass


def c_literal(value):
    if math.isnan(value):
        return 'NAN'
    if math.isinf(value):
        return 'INFINITY' if value > 0 else '(-INFINITY)'
    text = repr(float(value))
    return text if ('.' in text or 'e' in text) else text + '.0'


def c_expression(e):
    """An expression of kernel_ir as C (states ``y[i]``, inputs ``in[k]``, time ``t``)."""
    op = e[0]
    if op == 'num':
        return c_literal(e[1])
    if op == 'y':
        return f'y[{e[1]}]'
    if op == 'in':
        return f'in[{e[1]}]'
    if op == 't':
        return 't'
    if op == 'tmp':
        return e[1]
    if op in _C_INFIX:
        return f'({c_expression(e[1])} {_C_INFIX[op]} {c_expression(e[2])})'
    if op == 'neg':
        return f'(-{c_expression(e[1])})'
    if op == 'not':
        return f'(!{c_expression(e[1])})'
    if op == 'xor':
        return f'(({c_expression(e[1])} != 0) != ({c_expression(e[2])} != 0))'
    if op == 'select':
        return f'({c_expression(e[1])} ? {c_expression(e[2])} : {c_expression(e[3])})'
    if op in _C_FUNCTIONS:
        return f'{_C_FUNCTIONS[op]}({", ".join(c_expression(x) for x in e[1:])})'
    raise BatchedBuildError(f'no C for operation {op}')


def _reads_time(program):
    def walk(e):
        return e[0] == 't' or (e[0] not in ('num', 'y', 'in', 'tmp') and any(walk(x) for x in e[1:]))
    return any(walk(e) for _, e in program.assignments) or any(walk(e) for e in program.rates)


def kernel_context(kernel, n_blocks):
    """The template context of one kernel: its C statements."""
    n = kernel.n_states
    rates = [f'const double {name} = {c_expression(e)};' for name, e in kernel.rates.assignments]
    rates += [f'dy[{i}] = {c_expression(e)};' for i, e in enumerate(kernel.rates.rates)]
    jacobian = [f'const double {name} = {c_expression(e)};' for name, e in kernel.jacobian.assignments]
    jacobian += [f'J[{r * n + c}] = {c_expression(e)};  /* d rate {r} / d state {c} */'
                 for r, c, e in kernel.jacobian.entries]
    return {'index': kernel.index, 'n_states': n, 'n_inputs': kernel.n_inputs, 'nonzeros': kernel.nonzeros,
            'reads_time': _reads_time(kernel.rates), 'n_blocks': n_blocks, 'rates': rates, 'jacobian': jacobian}


def block_table(blocks):
    """The blocks as the C arrays the library builds its instances from (each array ends with a 0, so none
    is empty)."""
    state_offsets, input_offsets, states, inputs = [0], [0], [], []
    for b in blocks:
        states += b.states
        inputs += b.inputs
        state_offsets.append(len(states))
        input_offsets.append(len(inputs))

    def c_array(values):
        return ', '.join(str(v) for v in list(values) + [0])
    return {'kernel': c_array(b.kernel for b in blocks), 'state_offsets': c_array(state_offsets),
            'states': c_array(states), 'input_offsets': c_array(input_offsets), 'inputs': c_array(inputs)}


def sources(am, km, model_name='model', outputs=(), state_init=None, solver='BE_batched_c'):
    """{file name: text} of the library for an analysed model and its kernel model; ``outputs`` are the algebraic
    variables bk_instance_outputs computes (kernel_ir.outputs_program); ``state_init`` the one-array variable each
    state's initial value names, or None for a literal (describe()'s ``state_init_ref``), for
    bk_instance_initialise_states; ``solver`` one of SOLVERS, whose runtime integrates the blocks."""
    if solver not in SOLVERS:
        raise ValueError(f'solver must be one of {tuple(SOLVERS)}, not {solver!r}')
    if am.externalVariableCount():
        raise BatchedBuildError(f'{solver} does not take models with external variables yet')
    env = template_environment()
    counts = {}
    for b in km.blocks:
        counts[b.kernel] = counts.get(b.kernel, 0) + 1
    ctx = {'model_name': model_name, 'kernels': [kernel_context(k, counts[k.index]) for k in km.kernels],
           'n_blocks': len(km.blocks), 'n_states': km.state_count, 'target': TARGET,
           'block_table': block_table(km.blocks), 'solver': solver, 'cvodes': solver == 'CVODES_batched_c'}
    program = kernel_ir.outputs_program(am, outputs)
    ctx['outputs'] = [f'in[{k}] = {c_expression(e)};' for k, e in program]
    ctx['output_targets'] = ', '.join(str(i) for i in kernel_ir.output_indices(am, outputs) + [0])   # never empty
    ctx['n_outputs'] = len(outputs)
    refs = state_init if state_init is not None else [None] * km.state_count
    ctx['state_init'] = ', '.join(str(-1 if r is None else r) for r in list(refs) + [None])   # never empty
    interface, implementation = core_code(am)
    blocks = [{'kernel': b.kernel, 'states': b.states, 'inputs': b.inputs} for b in km.blocks]
    return {
        'model0d_core.h': interface,
        'model0d_core.c': implementation,
        'batched_kernels.h': env.get_template('batched_kernels.h.j2').render(**ctx),
        'batched_kernels.c': env.get_template('batched_kernels.c.j2').render(**ctx),
        'batched_runtime.c': env.get_template(SOLVERS[solver]['runtime']).render(**ctx),
        'CMakeLists.txt': env.get_template('batched_CMakeLists.txt.j2').render(**ctx),
        'blocks.json': json.dumps(blocks),
    }


def build_dir_for(files, parent):
    digest = hashlib.sha1()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(files[name].encode())
    return os.path.join(parent, f'batched_c_{digest.hexdigest()[:12]}')


def _lock(f):
    """An exclusive lock on open file ``f`` until it is closed: flock on POSIX, a one-byte region lock on Windows."""
    try:
        import fcntl
    except ImportError:
        import msvcrt
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)   # retries for ~10 s, then raises
                return
            except OSError:
                continue
    fcntl.flock(f, fcntl.LOCK_EX)


def build(files, parent=None, verbose=False):
    """Write ``files`` to a folder named after their digest under ``parent`` (the system temp folder by
    default) and build the library there unless it is already built; return (library path, seconds spent
    building, 0 when it was already built). A lock file serialises concurrent builds of the same sources."""
    from libcuflynx.coupling.model0d_lib import _library_path, build_library
    parent = parent or os.path.join(tempfile.gettempdir(), 'libcuflynx_batched_c')
    out = build_dir_for(files, parent)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, '.lock'), 'a+') as lock:
        _lock(lock)
        path = _library_path(os.path.join(out, 'build'), TARGET)
        if path is not None:
            return path, 0.0
        t0 = time.perf_counter()
        for name, text in files.items():
            with open(os.path.join(out, name), 'w') as f:
                f.write(text)
        try:
            path = build_library(out, os.path.join(out, 'build'), verbose=verbose, target=TARGET)
        except Exception as exc:
            hint = ''
            if 'cvodes' in files.get('CMakeLists.txt', '').lower():
                hint = ('\nCVODES_batched_c needs SUNDIALS 6 or later (CVODES): set SUNDIALS_DIR to its install '
                        'prefix if CMake does not find it.')
            raise BatchedBuildError(str(exc) + hint) from exc
        return path, time.perf_counter() - t0


def describe(flat_model, am, km):
    """What the solver needs of a model, as JSON-able data: names, kinds, the variable each state's initial value
    names (its one-array index, or None for a literal) and the blocks. Every component/variable name of the flat
    model maps to the index of the variable libCellML keeps for it."""
    vt = kernel_ir.VT
    kinds = {vt.CONSTANT: 'constant', vt.COMPUTED_CONSTANT: 'computed_constant',
             vt.ALGEBRAIC_VARIABLE: 'algebraic', vt.EXTERNAL_VARIABLE: 'external'}
    offsets = {kind: next((v.index for v in km.variables if v.kind == kind), None) for kind in kinds.values()}

    def index_of(variable):
        av = am.analyserVariable(variable)
        if av is None:
            return None, None
        if av.type() == vt.STATE:
            return 'state', av.index()
        if av.type() in kinds:
            return 'var', offsets[kinds[av.type()]] + av.index()
        return None, None

    def components(parent):
        for i in range(parent.componentCount()):
            c = parent.component(i)
            yield c
            yield from components(c)

    state_index, var_index = {}, {}
    for c in components(flat_model):
        for i in range(c.variableCount()):
            kind, index = index_of(c.variable(i))
            if kind == 'state':
                state_index[f'{c.name()}/{c.variable(i).name()}'] = index
            elif kind == 'var':
                var_index[f'{c.name()}/{c.variable(i).name()}'] = index
    state_names = [f'{s.component}/{s.name}' for s in km.states]
    variable_names = [f'{v.component}/{v.name}' for v in km.variables]
    state_index.update({name: i for i, name in enumerate(state_names)})
    var_index.update({name: k for k, name in enumerate(variable_names)})
    init_ref = []
    for st in km.states:
        ref = None
        if isinstance(st.initial, str):
            c = flat_model.component(st.component, True)
            if c is not None and c.variable(st.initial) is not None:
                kind, index = index_of(c.variable(st.initial))
                ref = index if kind == 'var' else None
        init_ref.append(ref)
    return {
        'state_count': km.state_count, 'variable_count': km.variable_count,
        'state_names': state_names, 'variable_names': variable_names,
        'variable_kinds': [v.kind for v in km.variables],
        'state_index': state_index, 'var_index': var_index, 'state_init_ref': init_ref,
        'voi': f'{km.voi.component}/{km.voi.name}',
        'blocks': [{'kernel': b.kernel, 'states': b.states, 'inputs': b.inputs} for b in km.blocks],
        'kernels': [{'states': k.n_states, 'inputs': k.n_inputs, 'nonzeros': k.nonzeros} for k in km.kernels],
    }


_IMPORT = re.compile(r'<import\b[^>]*?xlink:href="([^"]+)"')


def _frozen_version():
    """What identifies the generator's code when its source files are not on disk (a frozen application): the
    installed libcuflynx's version, else the application's own timestamp."""
    try:
        from importlib.metadata import version
        return 'libcuflynx ' + version('libcuflynx')
    except Exception:
        import sys
        return f'application {os.path.getmtime(sys.executable)}'


def source_digest(cellml_path):
    """A digest of a CellML file and every file it imports, recursively, with this module's templates and
    code (so a changed generator does not load a stale build)."""
    digest = hashlib.sha1()
    seen, pending = set(), [os.path.abspath(cellml_path)]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        with open(path, 'rb') as f:
            data = f.read()
        digest.update(path.encode())
        digest.update(data)
        for href in sorted(set(_IMPORT.findall(data.decode('utf-8', 'replace')))):
            if '://' not in href:
                pending.append(os.path.normpath(os.path.join(os.path.dirname(path), href)))
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [__file__, kernel_ir.__file__]:
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                digest.update(f.read())
        else:
            # a frozen application ships compiled modules only: its version stands for the generator's code
            digest.update(_frozen_version().encode())
    for path in sorted(
            os.path.join(here, 'templates', n) for n in os.listdir(os.path.join(here, 'templates'))
            if n.startswith('batched_')):
        with open(path, 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()[:16]


def load_or_build(cellml_path, parent=None, verbose=False, outputs=(), solver='BE_batched_c'):
    """The description and built library of a CellML model for ``solver`` (one of SOLVERS): from the cache under
    ``parent`` (keyed by :func:`source_digest`, the outputs and the solver) when there, else analysed, generated
    and built. Returns (description, library path, seconds spent building, True when cached)."""
    if solver not in SOLVERS:
        raise ValueError(f'solver must be one of {tuple(SOLVERS)}, not {solver!r}')
    parent = parent or os.path.join(tempfile.gettempdir(), 'libcuflynx_batched_c')
    key = source_digest(cellml_path)
    if solver != 'BE_batched_c':
        key += '_' + solver
    if outputs:
        key += '_' + hashlib.sha1('\n'.join(outputs).encode()).hexdigest()[:8]
    index = os.path.join(parent, 'models', key + '.json')
    if os.path.isfile(index):
        with open(index) as f:
            cached = json.load(f)
        if os.path.isfile(cached['library']):
            return cached['description'], cached['library'], 0.0, True
    flat, am = kernel_ir.analyse_cellml(cellml_path)
    km = kernel_ir.build_kernel_model(am)
    description = describe(flat, am, km)
    files = sources(am, km, os.path.splitext(os.path.basename(cellml_path))[0], outputs,
                    description['state_init_ref'], solver)
    path, seconds = build(files, parent, verbose=verbose)
    os.makedirs(os.path.dirname(index), exist_ok=True)
    staging = index + f'.{os.getpid()}.tmp'
    with open(staging, 'w') as f:
        json.dump({'library': path, 'description': description}, f)
    os.replace(staging, index)
    return description, path, seconds, False


class BatchedLibrary:
    """The built library of one model, through ctypes."""

    def __init__(self, path):
        self.path = path
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            # a CVODES library built against the headers of another SUNDIALS than the libraries it links
            if 'SUN' in str(exc) or 'CVode' in str(exc):
                build_dir = os.path.dirname(os.path.dirname(path))
                raise BatchedBuildError(f'{exc}\nCVODES_batched_c needs SUNDIALS 6 or later (CVODES): set '
                                        f'SUNDIALS_DIR to one install, and delete {build_dir}.') from exc
            raise
        d, i, l, p = ctypes.c_double, ctypes.c_int, ctypes.c_long, ctypes.c_void_p
        dp, ip, lp = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_long)
        sigs = {
            'bk_kernel_count': ([], i), 'bk_kernel_states': ([i], i), 'bk_kernel_inputs': ([i], i),
            'bk_kernel_nonzeros': ([i], i),
            'bk_kernel_rates': ([i, d, dp, dp, dp], None), 'bk_kernel_jacobian': ([i, d, dp, dp, dp], None),
            'bk_create': ([i, ip, ip, ip, ip, ip, i, i], p), 'bk_destroy': ([p], None),
            'bk_set_options': ([p, d, d, d, d, d, i, i, l, i, i], None), 'bk_invalidate': ([p], None),
            'bk_set_time_origin': ([p, d], None), 'bk_run': ([p, dp, dp, d, i, dp, dp, i], i),
            'bk_block_status': ([p, i, dp], i), 'bk_stats': ([p, lp], None), 'bk_openmp_threads': ([], i),
            'bk_initialise': ([dp, dp, dp], None), 'bk_compute_computed_constants': ([dp, dp, dp], None),
            'bk_outputs': ([i, dp, dp, dp, dp, i], None),
            'bk_set_trace': ([p, i, i, dp, dp], None), 'bk_clear_trace': ([p, i], None),
            'bk_sim_outputs': ([p, i, dp, dp, dp, dp, i], None), 'bk_solver_name': ([], ctypes.c_char_p),
            'bk_instances_advance': ([i, ctypes.POINTER(p), d, i], i), 'bk_set_stop_at_outputs': ([p, i], None),
        }
        for name, (argtypes, restype) in sigs.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
        self._lib = lib
        #: the solver the library integrates with (a key of SOLVERS)
        self.solver = lib.bk_solver_name().decode()

    def __getattr__(self, name):
        return getattr(self._lib, name)


def ptr(a, ctype=ctypes.c_double):
    return a.ctypes.data_as(ctypes.POINTER(ctype)) if a is not None else None


STAT_NAMES = ('steps', 'rejected_steps', 'rate_evaluations', 'jacobians', 'finite_difference_jacobians',
              'factorisations', 'newton_failures')


class BatchedSimulation:
    """The blocks of one model and their integrator state, in a built library."""

    def __init__(self, library, blocks, n_states, n_variables):
        self.lib = library
        self.n_states = n_states
        self.n_variables = n_variables
        kernel = np.array([b['kernel'] for b in blocks], dtype=np.int32)
        state_offsets = np.cumsum([0] + [len(b['states']) for b in blocks]).astype(np.int32)
        input_offsets = np.cumsum([0] + [len(b['inputs']) for b in blocks]).astype(np.int32)
        states = np.array([s for b in blocks for s in b['states']], dtype=np.int32)
        inputs = np.array([k for b in blocks for k in b['inputs']] or [0], dtype=np.int32)
        self._keep = (kernel, state_offsets, input_offsets, states, inputs)
        self.n_blocks = len(blocks)
        self._h = library.bk_create(self.n_blocks, ptr(kernel, ctypes.c_int), ptr(state_offsets, ctypes.c_int),
                                    ptr(states, ctypes.c_int), ptr(input_offsets, ctypes.c_int),
                                    ptr(inputs, ctypes.c_int), n_states, n_variables)

    def set_options(self, rtol=1e-4, atol=1e-12, max_step=0.0, min_step=0.0, newton_tol=0.01, max_newton=6,
                    max_jacobian_age=50, max_steps=10_000_000, finite_difference_jacobian=False, method=None):
        """The integrator's settings; ``method`` one of the library's solver's methods (its first by default).
        CVODES_batched_c takes 0 for newton_tol, max_newton and max_jacobian_age as CVODES's own defaults."""
        methods = SOLVERS[self.lib.solver]['methods']
        method = method or methods[0]
        if method not in methods:
            raise ValueError(f'method must be one of {methods}, not {method!r}')
        self.lib.bk_set_options(self._h, rtol, atol, max_step or 0.0, min_step, newton_tol, max_newton,
                                max_jacobian_age, int(max_steps), int(bool(finite_difference_jacobian)),
                                methods.index(method))

    def set_stop_at_outputs(self, on):
        """BE_batched_c: steps end on each run's last output time, and a block whose inputs changed there keeps its
        step size, Jacobian and LU factors (faster when inputs change at every run; off by default). CVODES_batched_c
        ignores it."""
        self.lib.bk_set_stop_at_outputs(self._h, int(bool(on)))

    def invalidate(self):
        self.lib.bk_invalidate(self._h)

    def set_time_origin(self, origin):
        self.lib.bk_set_time_origin(self._h, origin)

    def run(self, states, variables, t0, t_out, out=None, threads=1):
        """Integrate from t0 through t_out; ``states`` (updated to the last output time), ``variables`` and
        ``out`` (len(t_out) x n_states) are float64 numpy arrays. Returns [(block, status, time)] of failures."""
        t_out = np.ascontiguousarray(t_out, dtype=float)
        failed = self.lib.bk_run(self._h, ptr(states), ptr(variables), float(t0), len(t_out), ptr(t_out),
                                 ptr(out), int(threads))
        if not failed:
            return []
        out_failures = []
        t = ctypes.c_double()
        for b in range(self.n_blocks):
            status = self.lib.bk_block_status(self._h, b, ctypes.byref(t))
            if status:
                out_failures.append((b, status, t.value))
        return out_failures

    def set_trace(self, variable, t, values):
        """Drives one-array variable ``variable`` by the piecewise-linear trace (t, values) on the model's clock
        (held at its end values outside it); steps end on its breakpoints."""
        t = np.ascontiguousarray(t, dtype=float)
        values = np.ascontiguousarray(values, dtype=float)
        if t.shape != values.shape or t.ndim != 1 or len(t) == 0:
            raise ValueError('a trace needs t and values of the same length')
        if np.any(np.diff(t) <= 0):
            raise ValueError('a trace\'s times must increase')
        self.lib.bk_set_trace(self._h, int(variable), len(t), ptr(t), ptr(values))

    def clear_trace(self, variable):
        self.lib.bk_clear_trace(self._h, int(variable))

    def outputs(self, t, states_rows, variables, variables_rows, threads=1):
        """Every variable at each output row (libCellML's computeVariables), traced ones at their trace's
        value at the row's time."""
        self.lib.bk_sim_outputs(self._h, len(t), ptr(t), ptr(states_rows), ptr(variables), ptr(variables_rows),
                                int(threads))

    def stats(self):
        buf = (ctypes.c_long * len(STAT_NAMES))()
        self.lib.bk_stats(self._h, buf)
        return dict(zip(STAT_NAMES, list(buf)))

    def close(self):
        if self._h:
            self.lib.bk_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
