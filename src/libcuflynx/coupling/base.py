'''The class an external Python model implements to be coupled to a generated C++ 0D model.'''


class ExternalModel:
    '''Base class for an external model (optional: any class with these methods works).

    The coupling runner creates it once, with keyword arguments:

    * ``params``: ``{name: value}``, the module's constants (``<name>_<row>`` in the parameters
      file) and global constants, from its ``variables_and_units``;
    * ``neighbours``: ``{port variable: [0D module, ...]}``, the 0D modules each of its port
      variables is connected to. A variable connected to N modules is exchanged as an array of
      N values in this order;
    * ``comm``: the MPI communicator (``None`` when not run under MPI);
    * ``info``: ``{'units': {variable: CellML units}, 'directions': {variable: 'to_external' |
      'from_external'}, 'coupling_dt', 'output_dir', 'row'}``.

    Then, every coupling step of ``dt`` from ``t``:

    * ``step(t, dt, inputs)`` receives ``inputs``, ``{variable: np.ndarray}`` for every
      ``to_external`` variable (values of the 0D model at ``t``), advances the external model to
      ``t + dt``, and returns ``{variable: np.ndarray}`` for every ``from_external`` variable.
      Those values are held in the 0D model while it steps from ``t`` to ``t + dt``.

    Values are in the CellML units of the 0D variables (``info['units']``). Under MPI the 0D
    model runs on every rank and ``step`` must return the same values on every rank (reduce
    with ``comm.allreduce``).
    '''

    def __init__(self, params, neighbours, comm=None, info=None):
        self.params = dict(params)
        self.neighbours = dict(neighbours)
        self.comm = comm
        self.info = dict(info or {})
        self.setup()

    # --- override these -----------------------------------------------------------------------
    def setup(self):
        '''Build the model (mesh, forms, solvers): called once, from __init__.'''

    def initial_outputs(self):
        '''{variable: np.ndarray}: the from_external values at t = 0, before the first step.'''
        raise NotImplementedError

    def step(self, t, dt, inputs):
        '''Advance from t to t + dt with the 0D values ``inputs``; return the from_external values.'''
        raise NotImplementedError

    # --- optional -----------------------------------------------------------------------------
    def write(self, t):
        '''Called at every output time (the 0D model's dt), e.g. to save fields or probes.'''

    def close(self):
        '''Called once at the end of the run.'''

    # Needed only with ``subiterations`` > 0 (fixed-point iteration of each coupling step):
    # save the state at the start of the step, and go back to it.
    # def snapshot(self): ...
    # def restore(self): ...
