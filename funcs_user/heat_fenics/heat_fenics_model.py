"""A FEniCSx (dolfinx) heat-equation solver, plugged into CA as an ``external_python`` model.

This is the flagship example of the ``model_type: external_python`` / ``solver: external``
backend: a solver that owns its own time-stepping. CA does not integrate anything here --
it hands over ``dt``/``sim_time``/``pre_time``, asks for a run, and reads three probe
traces back. Everything between those two points is dolfinx.

The physics
-----------
Backward Euler for ``u_t = k Δu`` on the unit square, P1 Lagrange, Dirichlet ``u = u_D`` on
the whole boundary, and a Gaussian bump as the initial condition::

    u(x, 0) = exp(-|x - (0.5, 0.5)|^2 / (2 σ^2)),   σ = 0.15

Weak form, with ``u_n`` the previous step::

    ∫ u v dx + dt·k ∫ ∇u·∇v dx  =  ∫ u_n v dx

Two calibratable parameters, both ``fem.Constant`` so that changing them is an in-place
write to ``.value`` and never a re-compilation of the form:

* ``heat/k``   -- diffusivity, in the stiffness term.
* ``heat/u_D`` -- the Dirichlet value on the whole boundary.

Three probes are evaluated every step: ``heat/T_p1`` at (0.25, 0.25), ``heat/T_p2`` at
(0.5, 0.5) and ``heat/T_p3`` at (0.75, 0.75).

.. note::
   p1 and p3 sit symmetrically about the centre of a mesh that is itself invariant under a
   180° rotation, and the initial bump is radially symmetric -- so ``T_p1`` and ``T_p3``
   must agree to round-off. That is deliberate: it is a free correctness check on the probe
   locating and on the assembly, and ``tests/test_heat_fenics_example.py`` asserts it.

Time scales
-----------
The slowest mode of the unit square decays at ``λ = 2 k π² ≈ 19.7 k``, so at the default
``k = 1`` the bump has a time constant of about 0.05 s. That is what sets the suggested
``dt = 0.0005`` / ``sim_time = 0.05`` (100 steps, ~one time constant of decay): long enough
for an interesting transient across the whole ``k ∈ [0.2, 5]`` calibration box, short enough
that a whole run is milliseconds once the forms are compiled.

MPI
---
The mesh is built on ``MPI.COMM_SELF``, not ``COMM_WORLD``. CA parallelises over
*independent simulations* -- each rank runs its own parameter sample -- so every rank must
own a complete serial mesh. Building on ``COMM_WORLD`` would instead distribute one mesh
across the ranks and deadlock the moment two ranks asked for different parameters.

Version compatibility
---------------------
Written and tested against **dolfinx 0.8.x / 0.9.x** (conda-forge ``fenics-dolfinx``). The
handful of calls whose names have moved between releases -- the function-space constructor,
the bounding-box tree, the PETSc assembly helpers -- are looked up through
:func:`_resolve`, which raises a message naming the tested versions rather than an
``AttributeError`` from three frames down. See ``README.md`` for the install line.
"""
import numpy as np

try:
    from mpi4py import MPI
    import ufl
    import dolfinx
    from dolfinx import fem
    from dolfinx import mesh as dmesh
    from dolfinx import geometry as dgeometry
    from dolfinx.fem import petsc as fem_petsc
    from petsc4py import PETSc
except ImportError as _exc:                                   # pragma: no cover - env dependent
    raise ImportError(
        'the heat_fenics example needs FEniCSx (dolfinx) and petsc4py, which are not '
        'installed. They are conda-forge packages, not pip ones:\n'
        '    conda create -n fenicsx -c conda-forge fenics-dolfinx python=3.11\n'
        '    conda activate fenicsx\n'
        f'See funcs_user/heat_fenics/README.md. Underlying error: {_exc}') from _exc


#: The dolfinx releases this file is known to work with. Named in every API-drift message so
#: that "it broke after I updated" has an actionable answer.
TESTED_DOLFINX_VERSIONS = '0.8.x and 0.9.x'

#: Where the three probes sit, in the order they appear in ``output_names``.
PROBE_POINTS = ((0.25, 0.25), (0.5, 0.5), (0.75, 0.75))

#: Centre and width of the initial Gaussian bump.
BUMP_CENTRE = (0.5, 0.5)
BUMP_SIGMA = 0.15

#: Default mesh resolution. Overridable via ``solver_info['user_config']['nx']``.
DEFAULT_NX = 16


def _dolfinx_version():
    return getattr(dolfinx, '__version__', 'unknown')


def _resolve(name_candidates, *modules):
    """First attribute in ``name_candidates`` found on any of ``modules``.

    dolfinx renames things between minor releases (``FunctionSpace`` -> ``functionspace``,
    ``BoundingBoxTree`` -> ``bb_tree``, ``fem.set_bc`` -> ``fem.petsc.set_bc``). Looking them
    up by a list of candidates keeps this example working across the releases it is known
    good on, and fails with a message that says which one it was written for otherwise.
    """
    for name in name_candidates:
        for module in modules:
            attribute = getattr(module, name, None)
            if attribute is not None:
                return attribute
    raise RuntimeError(
        f'none of {list(name_candidates)} exist on '
        f'{[getattr(m, "__name__", str(m)) for m in modules]} in dolfinx '
        f'{_dolfinx_version()}. funcs_user/heat_fenics/heat_fenics_model.py was written '
        f'against dolfinx {TESTED_DOLFINX_VERSIONS}; this API has moved since. Either '
        f'install a tested version (`conda install -c conda-forge "fenics-dolfinx=0.9"`) '
        f'or update this file.')


def _resolve_optional(name, *modules):
    """Like :func:`_resolve`, but returns ``None`` instead of raising."""
    for module in modules:
        attribute = getattr(module, name, None)
        if attribute is not None:
            return attribute
    return None


def _scalar_type():
    scalar = getattr(dolfinx, 'default_scalar_type', None)
    return scalar if scalar is not None else PETSc.ScalarType


class HeatFEniCSxModel:
    """Transient heat conduction on the unit square, solved with dolfinx.

    Implements CA's ``external_python`` contract: ``init_solver`` / ``update_times`` /
    ``set_param_vals`` / ``run`` / ``get_results``, plus the optional ``reset``,
    ``get_init_param_vals``, ``extra_plots`` and ``close``.
    """

    # --- self description -------------------------------------------------------------
    # Literal values only: CA's tooling reads these by parsing the file, without importing
    # it, so that a machine with no dolfinx can still list the model's parameters.
    parameters = {"heat/k": 1.0, "heat/u_D": 0.0}
    output_names = ["heat/T_p1", "heat/T_p2", "heat/T_p3"]

    def __init__(self):
        self._mesh = None
        self._V = None
        self._k_const = None
        self._uD_const = None
        self._dt_const = None
        self._a_form = None
        self._L_form = None
        self._bc = None
        self._u_n = None
        self._uh = None
        self._probe_points = None
        self._probe_cells = None
        self._dof_coords = None
        self._solver = None
        self._matrix = None
        self._rhs_vector = None
        self._sigma = BUMP_SIGMA
        self.dt = None
        self.start_time = 0.0
        self.sim_time = None
        self.pre_time = 0.0
        self.num_steps = 0
        self._samples = None
        self._times = None
        self._snapshot_mid = None
        self._snapshot_final = None
        self._snapshot_mid_time = None

    # --- required: one-off setup ------------------------------------------------------

    def init_solver(self, config):
        """Build the mesh, the function space, the forms and the probe cells -- once.

        Args:
            config: dict with ``dt``, ``sim_time``, ``pre_time``, ``start_time`` and
                ``solver_info``. ``solver_info['user_config']`` is the free-form block a
                user sets in ``user_inputs.yaml``; ``nx`` (mesh resolution, default 16),
                ``ny``, ``bump_sigma`` and ``petsc_pc`` are read from it.
        """
        solver_info = config.get('solver_info') or {}
        user_config = solver_info.get('user_config') or {}

        nx = int(user_config.get('nx', DEFAULT_NX))
        ny = int(user_config.get('ny', nx))
        sigma = float(user_config.get('bump_sigma', BUMP_SIGMA))
        if nx < 2 or ny < 2:
            raise ValueError(f'user_config nx/ny must be at least 2, got nx={nx}, ny={ny}')

        # COMM_SELF: one complete mesh per rank. See the module docstring.
        create_unit_square = _resolve(('create_unit_square', 'UnitSquareMesh'), dmesh)
        self._mesh = create_unit_square(MPI.COMM_SELF, nx, ny)

        make_space = _resolve(('functionspace', 'FunctionSpace'), fem)
        self._V = make_space(self._mesh, ('Lagrange', 1))

        scalar_type = _scalar_type()
        self._k_const = fem.Constant(self._mesh, scalar_type(self.parameters['heat/k']))
        self._uD_const = fem.Constant(self._mesh, scalar_type(self.parameters['heat/u_D']))
        # dt lives in the form as a Constant too, so update_times is an in-place write and
        # never triggers a re-compilation of the generated kernels.
        self._dt_const = fem.Constant(self._mesh, scalar_type(1.0))

        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)
        self._u_n = fem.Function(self._V, name='u_n')
        self._uh = fem.Function(self._V, name='u')

        a = (u * v * ufl.dx
             + self._dt_const * self._k_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
        L = self._u_n * v * ufl.dx
        self._a_form = fem.form(a)
        self._L_form = fem.form(L)

        self._bc = self._make_boundary_condition()
        self._sigma = sigma
        self._set_initial_condition()

        self._dof_coords = self._V.tabulate_dof_coordinates()[:, :2].copy()
        self._locate_probes()

        self._solver = PETSc.KSP().create(self._mesh.comm)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(user_config.get('petsc_pc', PETSc.PC.Type.LU))

        self.update_times(config['dt'], config.get('start_time', 0.0),
                          config['sim_time'], config.get('pre_time', 0.0))

    def _make_boundary_condition(self):
        """Dirichlet ``u = u_D`` on the whole boundary."""
        tdim = self._mesh.topology.dim
        fdim = tdim - 1
        self._mesh.topology.create_connectivity(fdim, tdim)
        exterior_facet_indices = _resolve(('exterior_facet_indices',), dmesh)
        boundary_facets = exterior_facet_indices(self._mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(self._V, fdim, boundary_facets)
        if len(boundary_dofs) == 0:
            raise RuntimeError(
                'no boundary dofs were found, so the Dirichlet condition would be a no-op. '
                f'This example was written against dolfinx {TESTED_DOLFINX_VERSIONS}; check '
                f'that dolfinx {_dolfinx_version()} still returns facet indices from '
                'mesh.exterior_facet_indices(mesh.topology).')
        # The Constant form of dirichletbc takes the function space explicitly.
        return fem.dirichletbc(self._uD_const, boundary_dofs, self._V)

    def _set_initial_condition(self):
        cx, cy = BUMP_CENTRE
        sigma = self._sigma

        def bump(x):
            return np.exp(-((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / (2.0 * sigma ** 2))

        self._u_n.interpolate(bump)
        self._uh.x.array[:] = self._u_n.x.array

    def _locate_probes(self):
        """Find, once, which cell each probe point falls in.

        Point evaluation is the API most likely to drift between dolfinx releases, so it is
        isolated here and every failure mode gets its own message.
        """
        points = np.zeros((len(PROBE_POINTS), 3), dtype=np.float64)
        for idx, (px, py) in enumerate(PROBE_POINTS):
            points[idx, 0] = px
            points[idx, 1] = py

        bb_tree = _resolve(('bb_tree', 'BoundingBoxTree'), dgeometry)
        compute_collisions = _resolve(
            ('compute_collisions_points', 'compute_collisions'), dgeometry)
        compute_colliding_cells = _resolve(('compute_colliding_cells',), dgeometry)

        tree = bb_tree(self._mesh, self._mesh.topology.dim)
        candidates = compute_collisions(tree, points)
        colliding = compute_colliding_cells(self._mesh, candidates, points)

        cells = []
        for idx in range(len(points)):
            links = colliding.links(idx)
            if len(links) == 0:
                raise RuntimeError(
                    f'probe {idx + 1} at {PROBE_POINTS[idx]} is not inside any mesh cell. '
                    f'On the unit square that should be impossible, so it points at a '
                    f'change in the dolfinx geometry API (tested against '
                    f'{TESTED_DOLFINX_VERSIONS}, running {_dolfinx_version()}).')
            cells.append(int(links[0]))

        self._probe_points = points
        self._probe_cells = np.asarray(cells, dtype=np.int32)

    # --- required: the record grid -----------------------------------------------------

    def update_times(self, dt, start_time, sim_time, pre_time):
        """Set the output grid. Cheap by construction: nothing is re-assembled here.

        ``run`` then produces samples at ``start_time + i*dt`` for ``i`` in ``0..N``, with
        ``N = int(pre_time/dt) + int(sim_time/dt)`` -- the same arithmetic CA uses, so the
        two agree on the length exactly rather than approximately.
        """
        if self._dt_const is None:
            raise RuntimeError('update_times() was called before init_solver(); there is no '
                               'assembled problem to re-grid yet')
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f'dt must be positive, got {dt}')
        self.dt = dt
        self.start_time = float(start_time)
        self.sim_time = float(sim_time)
        self.pre_time = float(pre_time)
        self.num_steps = int(self.pre_time / dt) + int(self.sim_time / dt)
        self._dt_const.value = _scalar_type()(dt)
        self._times = self.start_time + dt * np.arange(self.num_steps + 1, dtype=np.float64)
        self._samples = None

    # --- required: parameters ----------------------------------------------------------

    def set_param_vals(self, param_dict):
        """Write new parameter values in place. Never requires a re-init.

        Both parameters are ``fem.Constant``s that already appear in the compiled forms, so
        this is an assignment to a one-element array -- no re-assembly, no FFCx round trip.
        """
        for name, value in param_dict.items():
            if name == 'heat/k':
                value = float(value)
                if value <= 0.0:
                    raise ValueError(f'heat/k must be positive, got {value}')
                self._k_const.value = _scalar_type()(value)
            elif name == 'heat/u_D':
                self._uD_const.value = _scalar_type()(float(value))
            else:
                raise ValueError(
                    f'unknown parameter "{name}" for the heat_fenics model; it knows '
                    f'{sorted(self.parameters)}')

    def get_init_param_vals(self, names):
        """The declared defaults, in the order asked for."""
        missing = [name for name in names if name not in self.parameters]
        if missing:
            raise ValueError(
                f'no default for {missing}; the heat_fenics model declares '
                f'{sorted(self.parameters)}')
        return [self.parameters[name] for name in names]

    # --- required: solve ---------------------------------------------------------------

    def run(self):
        """Solve the whole grid from the initial condition. Repeatable.

        Every call restarts from the Gaussian bump, so back-to-back runs at the same
        parameters give bit-identical traces -- which is what lets a calibration or a
        sensitivity sweep reuse one instance for thousands of samples.

        Returns:
            bool: True on success; False if the solve raised or produced non-finite values.
        """
        try:
            self._solve()
        except Exception as error:                            # noqa: BLE001 - reported below
            print(f'[heat_fenics] the solve failed and the run is being reported as '
                  f'diverged: {type(error).__name__}: {error}')
            return False

        for name, trace in self._samples.items():
            if not np.all(np.isfinite(trace)):
                print(f'[heat_fenics] {name} contains non-finite values; reporting the run '
                      f'as diverged (k={float(self._k_const.value)}, dt={self.dt})')
                return False
        return True

    def _solve(self):
        assemble_matrix = _resolve(('assemble_matrix',), fem_petsc, fem)
        assemble_vector = _resolve(('assemble_vector',), fem_petsc, fem)
        apply_lifting = _resolve(('apply_lifting',), fem_petsc, fem)
        set_bc = _resolve(('set_bc',), fem_petsc, fem)

        self.reset()

        # k changes between runs, so the matrix is rebuilt here rather than in init_solver.
        # It is ~n_dofs entries at this size, so the cost is noise next to the form
        # compilation that init_solver already paid.
        self._destroy_matrix()
        self._matrix = assemble_matrix(self._a_form, bcs=[self._bc])
        self._matrix.assemble()
        # dolfinx's create_vector builds the vector with the function space's ghost layout,
        # which is what assemble_vector expects. createVecRight is the serial fallback for a
        # release that has moved or renamed it.
        create_vector = _resolve_optional('create_vector', fem_petsc)
        self._rhs_vector = (create_vector(self._L_form) if create_vector is not None
                            else self._matrix.createVecRight())
        self._solver.setOperators(self._matrix)

        self._record(0)
        mid_step = max(1, self.num_steps // 2)

        for step in range(1, self.num_steps + 1):
            with self._rhs_vector.localForm() as local:
                local.set(0.0)
            assemble_vector(self._rhs_vector, self._L_form)
            apply_lifting(self._rhs_vector, [self._a_form], [[self._bc]])
            self._rhs_vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                         mode=PETSc.ScatterMode.REVERSE)
            set_bc(self._rhs_vector, [self._bc])

            self._solver.solve(self._rhs_vector, self._solution_vec())
            self._uh.x.scatter_forward()

            self._u_n.x.array[:] = self._uh.x.array
            self._record(step)

            if step == mid_step:
                self._snapshot_mid = self._uh.x.array.copy()
                self._snapshot_mid_time = float(self._times[step])

        self._snapshot_final = self._uh.x.array.copy()

    def _solution_vec(self):
        """The PETSc Vec behind ``self._uh``.

        ``Function.x.petsc_vec`` in dolfinx 0.8+; ``Function.vector`` before that.
        """
        petsc_vec = getattr(self._uh.x, 'petsc_vec', None)
        if petsc_vec is not None:
            return petsc_vec
        petsc_vec = getattr(self._uh, 'vector', None)
        if petsc_vec is not None:
            return petsc_vec
        raise RuntimeError(
            f'a dolfinx Function in {_dolfinx_version()} exposes neither `x.petsc_vec` nor '
            f'`vector`, so there is nothing to solve into. This example was written against '
            f'dolfinx {TESTED_DOLFINX_VERSIONS}.')

    def _record(self, step):
        values = self._probe(self._u_n)
        for idx, name in enumerate(self.output_names):
            self._samples[name][step] = values[idx]

    def _probe(self, function):
        """The three probe values of ``function``, as a length-3 array."""
        return np.asarray(function.eval(self._probe_points, self._probe_cells),
                          dtype=float).reshape(-1)

    # --- required: results -------------------------------------------------------------

    def get_results(self):
        """The three probe traces on the record grid, ``pre_time`` samples included.

        Returns:
            dict: ``{output_name: 1D np.ndarray}``, each of length ``N + 1``. CA discards
            the leading ``int(pre_time/dt)`` samples itself.
        """
        if self._samples is None:
            raise RuntimeError('get_results() was called before run(); there is nothing to '
                               'return yet')
        return {name: trace.copy() for name, trace in self._samples.items()}

    def get_time(self):
        """The record grid itself -- handy when driving this file directly."""
        return self._times.copy()

    # --- optional ----------------------------------------------------------------------

    def reset(self):
        """Back to the initial condition, with an empty set of recorded samples."""
        self._set_initial_condition()
        self._samples = {name: np.zeros(self.num_steps + 1, dtype=float)
                         for name in self.output_names}
        self._snapshot_mid = None
        self._snapshot_final = None
        self._snapshot_mid_time = None

    def extra_plots(self):
        """Two figures: the field at mid-time and at the final time.

        Returned rather than shown or saved, so CA (and the CUFLynx GUI) decides where they
        go. ``matplotlib.figure.Figure`` is used directly rather than ``pyplot`` -- no global
        state and no backend to configure, which is what makes this safe on a headless node.
        """
        from matplotlib.figure import Figure  # lazy: plotting is not needed to simulate

        if self._snapshot_final is None:
            raise RuntimeError('extra_plots() was called before a successful run(); there '
                               'are no fields to draw yet')

        figures = []
        panels = (
            (self._snapshot_mid, self._snapshot_mid_time, 'mid-time'),
            (self._snapshot_final, float(self._times[-1]), 'final time'),
        )
        for field, time, label in panels:
            if field is None:
                continue
            figures.append(self._field_figure(Figure, field, time, label))
        return figures

    def _field_figure(self, figure_class, field, time, label):
        x = self._dof_coords[:, 0]
        y = self._dof_coords[:, 1]

        figure = figure_class(figsize=(5.0, 4.2))
        axes = figure.add_subplot(111)
        # No explicit triangle list: the domain is convex, so matplotlib's own Delaunay
        # triangulation of the P1 dof coordinates is the mesh, and this stays clear of the
        # dolfinx cell-connectivity API entirely.
        mappable = axes.tripcolor(x, y, field, shading='gouraud')
        figure.colorbar(mappable, ax=axes, label='u')

        for idx, (px, py) in enumerate(PROBE_POINTS):
            axes.plot(px, py, 'o', markersize=7, markerfacecolor='none',
                      markeredgecolor='white', markeredgewidth=1.8)
            axes.annotate(f'p{idx + 1}', (px, py), textcoords='offset points',
                          xytext=(8, 6), color='white', fontsize=9)

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_aspect('equal')
        axes.set_title(f'heat_fenics: u at {label} (t = {time:.4g} s, '
                       f'k = {float(self._k_const.value):.4g})')
        figure.tight_layout()
        return figure

    def close(self):
        """Release the PETSc objects. Safe to call more than once."""
        self._destroy_matrix()
        if self._solver is not None:
            self._solver.destroy()
            self._solver = None

    def _destroy_matrix(self):
        if self._rhs_vector is not None:
            self._rhs_vector.destroy()
            self._rhs_vector = None
        if self._matrix is not None:
            self._matrix.destroy()
            self._matrix = None


#: What CA looks for when it loads this file.
SIM_HELPER = HeatFEniCSxModel


if __name__ == '__main__':
    # Drive the model without CA -- the quickest way to check an install, and how the
    # numbers in heat_fenics_obs_data.json can be regenerated exactly. See README.md.
    model = HeatFEniCSxModel()
    model.init_solver({'dt': 0.0005, 'sim_time': 0.05, 'pre_time': 0.0, 'start_time': 0.0,
                       'solver_info': {'user_config': {'nx': 16}}})
    assert model.run(), 'the heat_fenics reference run diverged'
    results = model.get_results()
    p2 = results['heat/T_p2']
    print(f'samples          : {len(p2)}')
    print(f'mean(heat/T_p2)  : {np.mean(p2):.6f}')
    print(f'min(heat/T_p2)   : {np.min(p2):.6f}')
    print(f'p1 == p3         : {np.allclose(results["heat/T_p1"], results["heat/T_p3"])}')
    model.close()
