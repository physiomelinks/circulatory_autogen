'''
C++ generation for model_type: cpp, from libCellML's C output and Jinja2 templates.

The libCellML-generated C code (the model equations) is written out unmodified. Everything else
-- the ``Model0d`` class, the solvers, output, the driver, the CMake build, and the code that fills
external variables -- comes from the templates in ``templates/``. External variables are declared
in data: delays through a module's ``delay_info``, and couplings to other models through ``api``
blocks in the module configs (see ``api.py``).
'''

import json
import math
import os
import re
import sys
import tempfile

import jinja2

from libcuflynx.utilities.paths import default_resources_dir
from libcuflynx.generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from libcuflynx.generators.cpp import externals as ext
from libcuflynx.generators.cpp.api import CAPI_VERSION, python_model_path, unit_factor
from libcuflynx.generators.naming import build_symbols

try:
    import libcellml
    from libcellml import Analyser, AnalyserExternalVariable, AnalyserModel, Generator, GeneratorProfile
    import libcuflynx.utilities.libcellml_helper_funcs as cellml
    LIBCELLML_available = True
except ImportError as e:  # pragma: no cover
    print('Error -> ', e)
    LIBCELLML_available = False

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
SUPPORTED_SOLVERS = ('CVODE', 'RK4', 'explEul', 'Heun', 'midpoint')

CORE_NAME = 'model0d_core'


def template_environment():
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )


class CppGenerationError(RuntimeError):
    pass


class CVS0DCppGenerator(object):
    '''
    Generates the C++ for a 0D model (model_type: cpp), optionally coupled to other models.

    Output, in cpp_generated_models_dir:
      model0d_core.c/.h   libCellML C code for the model equations (unmodified)
      model0d.h/.cpp      the Model0d class: solver, output, external variables
      main0d.cpp          driver, standalone or launched by the 1D-0D coupler
      CMakeLists.txt      builds libmodel0d and main0d
      <name>_coupler1d0d.json   (coupled to 1D) connection info read by the coupler and 1D solver
      coupler_config.json       (coupled to 1D) what the coupler launches and how, from the api
                                block of the process the connections name (e.g. FV1D_solver)
      circulation_api.h/.cpp, api_test_driver.cpp   (a provider api is linked, e.g. lifex)
      model0d_capi.cpp, external_models.json   (python external models: a C interface built as
                                the shared library model0d_capi, and what libcuflynx.coupling runs)
    '''

    def __init__(self, model, generated_model_subdir, file_prefix, resources_dir=None,
                 solver='CVODE', dtSample=1e-3, dtSolver=1e-4, nMaxSteps=5000,
                 reltol=1e-7, abstol=1e-9,
                 couple_to_1d=False, cpp_generated_models_dir=None,
                 model_1d_config_path=None, create_main_0d=False,
                 conn_1d_0d_info=None, DEBUG=False, human_readable=True):
        if not LIBCELLML_available:
            raise CppGenerationError('libCellML is not available, cannot generate C++ files.')

        self.model = model
        self.generated_model_subdir = generated_model_subdir
        os.makedirs(self.generated_model_subdir, exist_ok=True)
        self.file_prefix = file_prefix
        self.resources_dir = resources_dir if resources_dir is not None else default_resources_dir()
        self.generated_model_file_path = os.path.join(self.generated_model_subdir, self.file_prefix + '.cellml')

        self.couple_to_1d = couple_to_1d
        self.conn_1d_0d_info = conn_1d_0d_info if couple_to_1d else None
        self.model_1d_config_path = model_1d_config_path
        # main0d.cpp is always written now: it runs standalone as well as under the coupler.
        self.create_main = True

        if cpp_generated_models_dir is None:
            cpp_generated_models_dir = self.generated_model_subdir + '_cpp' if couple_to_1d else self.generated_model_subdir
        self.cpp_generated_models_dir = cpp_generated_models_dir
        os.makedirs(self.cpp_generated_models_dir, exist_ok=True)

        self.solver = solver
        self.dtSample = dtSample
        self.dtSolver = dtSolver
        self.nMaxSteps = nMaxSteps
        self.reltol = reltol
        self.abstol = abstol
        self.DEBUG = DEBUG
        # Name every state/variable index in the generated code (S_<component>_<var>,
        # V_<component>_<var>), as the Python generator names its attributes.
        self.human_readable = human_readable

        # set from inp_data_dict in generate_cellml()
        self.sim_time = None
        self.pre_time = None
        self.cpp_output_dir = None
        self.coupler_pipe_dir = None
        self.parameters_csv = None

        # cardiac-cycle period and number of cycles of a coupled run (see _coupled_run_length)
        self.T0 = 1.0
        self.nCC = 5

        # filled by generate_cpp()
        self.externals = []
        self.analysed_model = None

    # -----------------------------------------------------------------------------------------
    # pipeline steps called by script_generate_with_new_architecture
    # -----------------------------------------------------------------------------------------

    def init_from_inp_data_dict(self, inp_data_dict):
        raise NotImplementedError('Construct CVS0DCppGenerator with its arguments.')

    def generate_cellml(self, inp_data_dict):
        print('generating CellML files, before C++ generation')
        self.sim_time = inp_data_dict.get('sim_time')
        self.pre_time = inp_data_dict.get('pre_time')
        self.cpp_output_dir = inp_data_dict.get('cpp_output_dir')
        self.coupler_pipe_dir = inp_data_dict.get('coupler_pipe_dir')
        if inp_data_dict.get('input_param_file') and inp_data_dict.get('resources_dir'):
            self.parameters_csv = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['input_param_file'])
        cellml_generator = CVS0DCellMLGenerator(self.model, inp_data_dict)
        cellml_generator.generate_files()

    def annotate_cellml(self):
        '''Kept for the generate script's call order. External variables no longer go through an
        RDF annotation round trip: they are read straight from the module configs.'''
        return None

    def set_annotated_model_file_path(self, annotated_model_file_path):
        return None

    def _build_solver_init_function(self):
        '''The generated Model0d::set_ode_solver body, as source text (solver_info reaches it).'''
        env = template_environment()
        return env.get_template('solver_init.j2').render(
            solver=self.solver, n_max_steps=self.nMaxSteps, dt_solver=self.dtSolver,
            reltol=self.reltol, abstol=self.abstol)

    def generate_cpp(self):
        if self.solver == 'PETSC':
            raise CppGenerationError(
                "solver 'PETSC' is not supported by the template C++ generator yet. Use 'CVODE' "
                "(implicit; needed for stiff models such as 1D-0D coupling) or an explicit scheme "
                f"({', '.join(s for s in SUPPORTED_SOLVERS if s != 'CVODE')}).")
        if self.solver not in SUPPORTED_SOLVERS:
            raise CppGenerationError(f'Unknown C++ solver {self.solver!r}; expected one of {SUPPORTED_SOLVERS}.')

        flat_model = self._flatten()
        vessels_df = self.model.vessels_df

        connections, volume_specs, pipe_externals = ext.collect_named_pipe_connections(
            vessels_df, self.conn_1d_0d_info, flat_model)
        delays = ext.collect_delays(vessels_df, flat_model)
        providers = ext.collect_provider_apis(vessels_df, flat_model)
        if len(providers) > 1:
            raise CppGenerationError('Only one provider api per model is supported.')
        python_externals, exchange = ext.collect_python_exchange(vessels_df, flat_model)
        if exchange and (connections or providers):
            raise CppGenerationError('A model coupled to Python external models (api transport "python") cannot '
                                     'also be coupled to the 1D solver or to a cpp_class provider yet.')

        externals = list(pipe_externals) + list(delays)
        for x in exchange:
            for spec in x.specs:
                # start from the parameter value of the boundary condition until the external sets it
                spec.initial = self._initial_value(spec.ref.variable)
                externals.append(spec)
        for prov in providers:
            for spec in prov['set_specs'].values():
                if spec.initial is None:
                    # start from the parameter value until the other model sets it
                    spec.initial = self._initial_value(spec.ref.variable)
                externals.append(spec)
        self._check_unique_externals(externals)

        analyser = Analyser()
        for spec in externals:
            aev = AnalyserExternalVariable(spec.ref.variable)
            for dep in spec.deps:
                aev.addDependency(dep.variable)
            analyser.addExternalVariable(aev)
        analyser.analyseModel(flat_model)
        self._report_analyser_issues(analyser)
        am = cellml.get_analysed_model(analyser)
        if am.type() != AnalyserModel.Type.ODE:
            raise CppGenerationError(f'The analysed model is not an ODE model (type {am.type()}); '
                                     f'check the analyser issues above.')
        self.analysed_model = am
        self._build_index_symbols(am)

        # Resolve every model reference to its state/variables index in the generated code.
        refs = [s.ref for s in externals]
        for d in delays:
            refs += [d.meta['source'], d.meta['amount']]
        for c in connections:
            refs += [c.output_ref] + ([c.control_ref] if c.control_ref is not None else [])
        for prov in providers:
            refs += list(prov['get_refs'].values()) + list(prov['state_refs'].values())
        for x in exchange:
            refs += x.refs
        for r in refs:
            self._resolve(r)
        for spec in externals:
            if spec.ref.kind != 'variable':
                raise CppGenerationError(f'External variable {spec.ref.label} did not become an EXTERNAL variable.')

        if self.couple_to_1d and self.conn_1d_0d_info is not None:
            ext.fill_conn_info(self.conn_1d_0d_info, connections, volume_specs)
            self._write_conn_info()
        process_api = self._process_api(connections, volume_specs)
        if self.couple_to_1d:
            self.T0, self.nCC = self._coupled_run_length()

        pipes = ext.build_named_pipe_code(connections, volume_specs) if connections else None
        hooks = pipes['hooks'] if pipes else {h: [] for h in ext.HOOKS}

        self._write_core(am)
        self._render_all(pipes, hooks, externals, delays, providers, len(connections),
                         len(connections) + len(volume_specs), exchange)
        if exchange:
            self._write_external_models(python_externals, exchange)
        if process_api is not None:
            self._write_coupler_config(process_api)
        self.externals = externals
        print(f'C++ files generated in {self.cpp_generated_models_dir}. Build with: '
              f'cmake -S {self.cpp_generated_models_dir} -B {os.path.join(self.cpp_generated_models_dir, "build")} '
              f'&& cmake --build {os.path.join(self.cpp_generated_models_dir, "build")}')
        return True

    # -----------------------------------------------------------------------------------------

    def _flatten(self):
        model = cellml.parse_model(self.generated_model_file_path, False)
        importer = cellml.resolve_imports(model, os.path.dirname(self.generated_model_file_path), False)
        flat_model = cellml.flatten_model(model, importer)
        self._numeric_initial_values(flat_model)
        with open(os.path.join(self.generated_model_subdir, self.file_prefix + '_flat.cellml'), 'w') as f:
            f.write(cellml.print_model(flat_model))
        return flat_model

    def _numeric_initial_values(self, flat_model):
        '''Replace state initial values given by computed variables with their numbers.

        CellML 2.0 allows a state's initial value to be a variable computed from constants (e.g.
        a gate starting at its steady state, m_init = m_inf(V_rest)); libCellML 0.6's analyser
        accepts only constants there. Myokit evaluates them, so the generated C starts from the
        same values as the CellML model run with Myokit.'''
        analyser = Analyser()
        analyser.analyseModel(flat_model)
        if not any('is initialised using variable' in analyser.issue(i).description()
                   for i in range(analyser.issueCount())):
            return
        import tempfile
        import myokit
        import myokit.formats
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'flat.cellml')
            with open(path, 'w') as f:
                f.write(cellml.print_model(flat_model))
            mk = myokit.formats.importer('cellml').model(path)
        values = {s.qname(): float(v) for s, v in zip(mk.states(), mk.initial_values(as_floats=True))}
        replaced = 0
        for c in range(flat_model.componentCount()):
            comp = flat_model.component(c)
            for v in range(comp.variableCount()):
                var = comp.variable(v)
                init = var.initialValue()
                if not init:
                    continue
                try:
                    float(init)
                    continue
                except ValueError:
                    pass
                key = f'{comp.name()}.{var.name()}'
                if key in values:
                    var.setInitialValue(repr(values[key]))
                    replaced += 1
        if self.DEBUG:
            print(f'C++ generation: {replaced} computed initial values evaluated with Myokit')

    @staticmethod
    def _initial_value(variable):
        '''Numeric initial value of a variable or of any variable connected to it, else 0.
        (libCellML drops the initial value of an external variable.)'''
        seen, todo = set(), [variable]
        while todo:
            v = todo.pop()
            key = (v.parent().name() if v.parent() is not None else '', v.name())
            if key in seen:
                continue
            seen.add(key)
            init = v.initialValue()
            if init:
                try:
                    return float(init)
                except ValueError:
                    pass
            todo += [v.equivalentVariable(i) for i in range(v.equivalentVariableCount())]
        return 0.0

    @staticmethod
    def _check_unique_externals(externals):
        seen = {}
        for spec in externals:
            key = spec.ref.label
            if key in seen:
                raise CppGenerationError(f'{key} is set externally twice ({seen[key]} and {spec.source}).')
            seen[key] = spec.source

    def _report_analyser_issues(self, analyser):
        errors = []
        for i in range(analyser.issueCount()):
            issue = analyser.issue(i)
            level = issue.level()
            text = issue.description()
            if level == libcellml.Issue.Level.ERROR:
                errors.append(text)
            elif self.DEBUG:
                print('libCellML analyser:', text)
        if errors:
            raise CppGenerationError('libCellML analyser errors:\n  ' + '\n  '.join(errors))

    def _resolve(self, ref):
        am = self.analysed_model
        if ref.kind is not None:
            return ref
        for astate in am.states():
            if am.areEquivalentVariables(astate.variable(), ref.variable):
                ref.kind, ref.index = 'state', astate.index()
                ref.symbol = self.state_symbols[ref.index] if self.human_readable else None
                return ref
        for av in am.variables():
            if am.areEquivalentVariables(av.variable(), ref.variable):
                ref.kind, ref.index = 'variable', av.index()
                ref.symbol = self.variable_symbols[ref.index] if self.human_readable else None
                return ref
        raise CppGenerationError(f'{ref.label} is not a state or variable of the analysed model.')

    @staticmethod
    def _process_api(connections, volume_specs):
        '''The api block of the process (e.g. the FV 1D solver) the pipe connections talk to, or None.'''
        apis = [c.api for c in connections] + [v['api'] for v in volume_specs]
        procs = {a['process_api']['name']: a['process_api'] for a in apis if 'process_api' in a}
        if len(procs) > 1:
            raise CppGenerationError('The pipe connections name more than one process; only one is supported.')
        return next(iter(procs.values()), None)

    def _global_parameter(self, name, units):
        '''A parameter's value: from the model's parameters, or, when no 0D module uses it (e.g. the
        inflow period T of a model whose inflow is in the 1D part), from the parameters file, as
        the 1D model generator reads it.'''
        params = getattr(self.model, 'parameters_array', None)
        if params is not None and len(params):
            match = params[params['variable_name'] == name]
            if len(match):
                return float(match['value'][0])
        if self.parameters_csv and os.path.isfile(self.parameters_csv):
            import pandas as pd
            df = pd.read_csv(self.parameters_csv, skipinitialspace=True, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            match = df[(df['variable_name'].str.strip() == name) & (df['units'].str.strip() == units)]
            if len(match):
                return float(match['value'].iloc[0])
        return None

    def _coupled_run_length(self):
        '''Period T0 and number of cycles nCC for the coupler, the 1D solver and main0d.

        T0 is the global period T (the heart period, or the inflow period of an open-loop model),
        which the FV1D_solver module declares as a global constant and the 1D model already reads.
        The coupled run lasts nCC whole periods, enough to cover pre_time + sim_time.
        '''
        T0 = self._global_parameter('T', 'second')
        if T0 is None or T0 <= 0.0:
            print("WARNING: no global parameter 'T' (cycle period) for the coupled run; using T0 = 1 s.")
            T0 = 1.0
        if self.sim_time is None:
            return T0, 5
        total = float(self.sim_time) + float(self.pre_time or 0.0)
        return T0, max(1, math.ceil(total / T0 - 1e-9))

    def _write_coupler_config(self, process_api):
        '''coupler_config.json: what the coordinator (the coupler) launches, and with what.'''
        from libcuflynx.utilities.package_resources import package_data_file
        program = process_api['program']
        if 'package' in program:
            script = os.path.abspath(str(package_data_file(program['package'], program['script'])))
        else:
            script = os.path.abspath(os.path.join(program['path'], program['script']))
        cpp_dir = os.path.abspath(self.cpp_generated_models_dir)
        # the system temp folder, so TMPDIR moves it where /tmp is unwritable
        pipe_dir = self.coupler_pipe_dir or os.path.join(tempfile.gettempdir(), 'cuflynx_pipes', self._model_name())
        config = {
            'inputFold': os.path.join(cpp_dir, ''),
            'networkName': self._model_name(),
            'ODEsolver': self.solver,
            'T0': self.T0,
            'nCC': self.nCC,
            'tmp_pipe_path': os.path.join(os.path.abspath(pipe_dir), ''),
            'solver0d_path': os.path.join(cpp_dir, 'build', 'main0d'),
            'python_path': sys.executable,
            'solver1d_path': script,
            'initFile_sim1d_path': os.path.abspath(self.model_1d_config_path) if self.model_1d_config_path else 'None',
            'initStatePath': 'None',
        }
        text = template_environment().get_template('coupler_config.json.j2').render(
            config=config, process_name=process_api['name'])
        json.loads(text)  # the template must produce valid JSON
        with open(os.path.join(self.cpp_generated_models_dir, 'coupler_config.json'), 'w') as f:
            f.write(text)

    def _json_name(self):
        name = self.file_prefix[:-3] if self.file_prefix.endswith('_0d') else self.file_prefix
        return name + '_coupler1d0d.json'

    def _model_name(self):
        return self.file_prefix[:-3] if self.file_prefix.endswith('_0d') else self.file_prefix

    def _write_conn_info(self):
        # Normal connections have no "port_volume_sum" key and volume-sum entries come last: the
        # coupler counts one-to-one connections by the absence of that key.
        for d in (self.generated_model_subdir, self.cpp_generated_models_dir):
            with open(os.path.join(d, self._json_name()), 'w') as f:
                json.dump(self.conn_1d_0d_info, f, indent=4)

    def _build_index_symbols(self, am):
        """S_/V_ names for every state and variable index, from the analysed model."""
        def pairs(items):
            ordered = sorted(items, key=lambda a: a.index())
            return ordered, [(a.variable().parent().name(), a.variable().name()) for a in ordered]
        self._states_ordered, state_pairs = pairs(am.states())
        self._variables_ordered, variable_pairs = pairs(am.variables())
        self.state_symbols = ['S_' + n for n in build_symbols(state_pairs)]
        self.variable_symbols = ['V_' + n for n in build_symbols(variable_pairs)]

    def _write_core(self, am):
        profile = GeneratorProfile(GeneratorProfile.Profile.C)
        profile.setInterfaceFileNameString(CORE_NAME + '.h')
        gen = Generator()
        interface = cellml.generate_interface_code(gen, am, profile)
        implementation = cellml.generate_implementation_code(gen, am, profile)
        if self.human_readable:
            interface = self._add_index_enums(interface)
            implementation = self._name_indices(implementation)
        with open(os.path.join(self.cpp_generated_models_dir, CORE_NAME + '.h'), 'w') as f:
            f.write(interface)
        with open(os.path.join(self.cpp_generated_models_dir, CORE_NAME + '.c'), 'w') as f:
            f.write(implementation)

    def _add_index_enums(self, interface):
        """Declare the named indices in the libCellML header, before its function declarations."""
        def entries(ordered, symbols):
            lines = []
            for a, sym in zip(ordered, symbols):
                v = a.variable()
                units = v.units().name() if v.units() is not None else ''
                kind = libcellml.AnalyserVariable.typeAsString(a.type()) if hasattr(a, 'type') else 'state'
                lines.append(f'    {sym} = {a.index()}, /* {v.parent().name()}.{v.name()} [{units}] {kind} */')
            return '\n'.join(lines)
        enums = ('/* Named indices into the states/rates and variables arrays (added by libcuflynx):\n'
                 '   S_<component>_<variable> for states and rates, V_<component>_<variable> for variables. */\n'
                 'typedef enum {\n' + entries(self._states_ordered, self.state_symbols) + '\n} StateIndex;\n\n'
                 'typedef enum {\n' + entries(self._variables_ordered, self.variable_symbols) + '\n} VariableIndex;\n\n')
        anchor = 'double * createStatesArray();'
        if anchor not in interface:
            raise CppGenerationError('Unexpected libCellML header layout: cannot place the index enums.')
        return interface.replace(anchor, enums + anchor, 1)

    def _name_indices(self, implementation):
        """Replace the numeric indices in libCellML's compute functions with the named ones."""
        states, variables = self.state_symbols, self.variable_symbols
        functions = ('initialiseVariables', 'computeComputedConstants', 'computeRates', 'computeVariables')

        def rename(body):
            body = re.sub(r'\bstates\[(\d+)\]', lambda m: f'states[{states[int(m.group(1))]}]', body)
            body = re.sub(r'\brates\[(\d+)\]', lambda m: f'rates[{states[int(m.group(1))]}]', body)
            body = re.sub(r'\bvariables\[(\d+)\]', lambda m: f'variables[{variables[int(m.group(1))]}]', body)
            body = re.sub(r'(externalVariable\(voi, states, rates, variables, )(\d+)\)',
                          lambda m: f'{m.group(1)}{variables[int(m.group(2))]})', body)
            return body

        out = implementation
        for fn in functions:
            m = re.search(rf'^void {fn}\(.*?\n\{{\n(.*?)^\}}\n', out, re.M | re.S)
            if m is None:
                raise CppGenerationError(f'Unexpected libCellML output: function {fn} not found.')
            body = rename(m.group(1))
            if re.search(r'\b(states|rates|variables)\[\d+\]|externalVariable\([^)]*, \d+\)', body):
                raise CppGenerationError(f'Unexpected libCellML output: numeric indices left in {fn}.')
            out = out[:m.start(1)] + body + out[m.end(1):]
        return out

    def _default_output_dir(self):
        if self.cpp_output_dir:
            return os.path.join(os.path.abspath(self.cpp_output_dir), '')
        if self.couple_to_1d:
            # where the previous generator (and the tutorial) put coupled 0D results
            return os.path.join(os.path.abspath(os.path.join(self.cpp_generated_models_dir, '..', '..')),
                                'simulation_outputs_cpp', self._model_name(), '')
        return os.path.join(os.path.abspath(self.cpp_generated_models_dir), 'simulation_outputs_cpp', '')

    def _render_all(self, pipes, hooks, externals, delays, providers, n_conn, n_conn_tot, exchange=()):
        env = template_environment()
        pre = float(self.pre_time) if self.pre_time is not None else None
        sim = float(self.sim_time) if self.sim_time is not None else None
        ctx = dict(
            file_prefix=self.file_prefix,
            model_name=self._model_name(),
            core_header=CORE_NAME + '.h',
            core_source=CORE_NAME + '.c',
            libcellml_version=libcellml.versionString(),
            solver=self.solver,
            n_max_steps=self.nMaxSteps,
            dt_solver=self.dtSolver,
            dt_sample=self.dtSample,
            reltol=self.reltol,
            abstol=self.abstol,
            has_externals=len(externals) > 0,
            externals=externals,
            delays=delays,
            pipes=pipes,
            hooks=hooks,
            n_connections=n_conn,
            n_connections_total=n_conn_tot,
            output_dir=self._default_output_dir(),
            default_T0=self.T0,
            default_nCC=self.nCC,
            default_end_time=(pre or 0.0) + sim if sim is not None else 20.0,
            default_save_time=pre if pre is not None else 0.0,
            cmake_project=''.join(c if c.isalnum() else '_' for c in self._model_name()) or 'model0d',
            extra_targets=[],
        )

        files = {
            'model0d.h': env.get_template('model0d.h.j2').render(**ctx),
            'model0d.cpp': env.get_template('model0d.cpp.j2').render(**ctx),
            'main0d.cpp': env.get_template('main0d.cpp.j2').render(**ctx),
        }
        for prov in providers:
            api_ctx = self._provider_context(prov)
            files['circulation_api.h'] = env.get_template('api_provider.h.j2').render(**ctx, **api_ctx)
            files['circulation_api.cpp'] = env.get_template('api_provider.cpp.j2').render(**ctx, **api_ctx)
            files['api_test_driver.cpp'] = env.get_template('api_test_driver.cpp.j2').render(**ctx, **api_ctx)
            ctx['extra_targets'] = [
                'add_library(circulation_api STATIC circulation_api.cpp)\n'
                'target_link_libraries(circulation_api PUBLIC model0d)\n'
                'add_executable(api_test_driver api_test_driver.cpp)\n'
                'target_link_libraries(api_test_driver PRIVATE circulation_api)'
            ]
        if exchange:
            files['model0d_capi.cpp'] = env.get_template('model0d_capi.cpp.j2').render(
                **ctx, exchange=exchange, capi_version=CAPI_VERSION)
            ctx['extra_targets'] = ctx['extra_targets'] + [
                '# C interface for Python external models (libcuflynx.coupling loads it with ctypes):\n'
                '# only the cf_* functions are exported, so several models can share one process.\n'
                'add_library(model0d_capi SHARED model0d_capi.cpp)\n'
                'target_link_libraries(model0d_capi PRIVATE model0d)\n'
                'set_target_properties(model0d_capi PROPERTIES CXX_VISIBILITY_PRESET hidden\n'
                '                      VISIBILITY_INLINES_HIDDEN ON)\n'
                'if(CMAKE_SYSTEM_NAME STREQUAL "Linux")\n'
                '    target_link_options(model0d_capi PRIVATE "LINKER:--exclude-libs,ALL")\n'
                'endif()'
            ]
        files['CMakeLists.txt'] = env.get_template('CMakeLists.txt.j2').render(**ctx)
        for name, text in files.items():
            with open(os.path.join(self.cpp_generated_models_dir, name), 'w') as f:
                f.write(text)
        # files the previous generator wrote that would now be stale next to the new ones
        for stale in ('model0d.cc', self.file_prefix + '.cc', self.file_prefix + '.h'):
            p = os.path.join(self.cpp_generated_models_dir, stale)
            if os.path.exists(p) and stale != 'model0d.h':
                os.remove(p)

    def _row_parameters(self, row_name, variables_and_units):
        '''{variable: value} of an external row's constants (<var>_<row> in the parameters file)
        and global constants (<var>), for its Python model.'''
        params = getattr(self.model, 'parameters_array', None)
        out = {}
        if params is None or not isinstance(variables_and_units, list):
            return out
        values = {str(n): v for n, v in zip(params['variable_name'], params['value'])}
        for entry in variables_and_units:
            var, kind = entry[0], entry[3]
            key = f'{var}_{row_name}' if kind == 'constant' else var if kind == 'global_constant' else None
            if key is None:
                continue
            if key not in values:
                raise CppGenerationError(f"Parameter {key} (constant {var} of external module '{row_name}') "
                                         f"is not in the parameters file.")
            out[var] = float(values[key])
        return out

    def _write_external_models(self, python_externals, exchange):
        '''external_models.json: what libcuflynx.coupling runs -- the C interface's exchange table,
        and for each Python external model its class, parameters and variables.'''
        index = {x.name: i for i, x in enumerate(exchange)}
        rows = self.model.vessels_df.set_index('name')
        models = []
        for entry in python_externals:
            api = entry['api']
            params = self._row_parameters(entry['row'], rows.loc[entry['row'], 'variables_and_units'])
            models.append({
                'row': entry['row'],
                'name': api.get('name', entry['row']),
                'file': python_model_path(api),
                'class': api['python']['class'],
                'parameters': params,
                # a coupling_dt constant of the module (set per instance in the parameters file)
                # overrides the api block's default
                'coupling_dt': float(params.get('coupling_dt', api.get('coupling_dt', self.dtSample))),
                'subiterations': int(api.get('subiterations', 0)),
                'tol': float(api.get('tol', 1e-8)),
                'relaxation': float(api.get('relaxation', 1.0)),
                'variables': [{'variable': x.variable, 'exchange_index': index[x.name], 'direction': x.direction,
                               'units': x.units, 'neighbours': x.neighbours} for x in entry['variables']],
            })
        info = {
            'capi_version': CAPI_VERSION,
            'model_name': self._model_name(),
            'library': 'model0d_capi',
            'solver': self.solver,
            'pre_time': float(self.pre_time or 0.0),
            'sim_time': float(self.sim_time) if self.sim_time is not None else None,
            'dt_output': float(self.dtSample),
            'output_dir': self._default_output_dir(),
            'external_models': models,
        }
        with open(os.path.join(self.cpp_generated_models_dir, 'external_models.json'), 'w') as f:
            json.dump(info, f, indent=2)

    def _provider_context(self, prov):
        '''Context for the provider (e.g. lifex Circulation) templates.'''
        api = prov['api']
        functions = []
        for fn in api['functions']:
            f = dict(fn)
            f['factor'] = unit_factor(fn)
            if fn['kind'] == 'set':
                f['ref'] = prov['set_specs'][fn['variable']].ref
            elif fn['kind'] == 'get':
                f['ref'] = prov['get_refs'][fn['variable']]
            elif fn['kind'] == 'set_state':
                f['ref'] = prov['state_refs'][fn['variable']]
                if f['ref'].kind != 'state':
                    raise CppGenerationError(f"{fn['name']}: {fn['variable']} is not a state, so it can't be set.")
            functions.append(f)

        chambers = []
        enum = api.get('chamber_enum', {})
        for cname, fields in enum.get('values', {}).items():
            ch = {'name': cname}
            for field_name, var in fields.items():
                if field_name in ('pressure_set', 'set'):
                    ch[field_name] = prov['set_specs'][var].ref
                else:
                    ch[field_name] = prov['get_refs'][var]
            chambers.append(ch)
        return dict(
            api=api,
            api_functions=functions,
            chambers=chambers,
            chamber_units={k: unit_factor({'api_units': v}) for k, v in enum.get('units', {}).items()},
            namespace=api.get('namespace', 'cuflynx'),
            class_name=api.get('class_name', 'Circulation'),
            chamber_header=enum.get('header'),
            chamber_type=enum.get('type', 'Chamber'),
            provider_vessel=prov['vessel'],
        )
