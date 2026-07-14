"""
Concise Python code generator for CellML models using libCellML.

This class parses a CellML file, resolves imports, analyses the model,
and emits a ready-to-run Python module.
"""
import ast
import os
import re
from typing import Optional

from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper

try:
    import libcellml as lc
except ImportError as e:  # pragma: no cover - runtime environment check
    raise ImportError("libcellml is required to generate Python models.") from e

try:
    import utilities.libcellml_helper_funcs as cellml_utils
    import utilities.libcellml_utilities as libcellml_utils
except ImportError:
    cellml_utils = None
    libcellml_utils = None


class _CasadiCompatTransformer(ast.NodeTransformer):
    """CasADi codegen transforms: ternaries -> ca.if_else, divisions -> guarded denominators."""

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ca", ctx=ast.Load()),
                attr="if_else",
                ctx=ast.Load(),
            ),
            args=[node.test, node.body, node.orelse],
            keywords=[],
        )

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not isinstance(node.op, ast.Div):
            return node
        guarded_denom = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ca", ctx=ast.Load()),
                attr="fmax",
                ctx=ast.Load(),
            ),
            args=[
                ast.Call(
                    func=ast.Name(id="fabs", ctx=ast.Load()),
                    args=[node.right],
                    keywords=[],
                ),
                ast.Constant(value=1e-300),
            ],
            keywords=[],
        )
        return ast.BinOp(left=node.left, op=ast.Div(), right=guarded_denom)


class _AadcCompatTransformer(ast.NodeTransformer):
    """AADC codegen transforms: ternaries -> aadc.iif, math -> aadc.math, etc."""

    _AADC_MATH = {'cos', 'sin', 'tan', 'exp', 'log', 'sqrt',
                  'acos', 'asin', 'atan', 'cosh', 'sinh', 'tanh'}

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            result = node.values[-1]
            for val in reversed(node.values[:-1]):
                result = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                       attr='iand', ctx=ast.Load()),
                    args=[val, result], keywords=[])
            return result
        if isinstance(node.op, ast.Or):
            result = node.values[-1]
            for val in reversed(node.values[:-1]):
                result = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                       attr='ior', ctx=ast.Load()),
                    args=[val, result], keywords=[])
            return result
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                               attr='iif', ctx=ast.Load()),
            args=[node.test, node.body, node.orelse], keywords=[])

    def visit_Call(self, node):
        self.generic_visit(node)
        # bare cos/sin/exp etc. → aadc.math.*
        if isinstance(node.func, ast.Name) and node.func.id in self._AADC_MATH:
            node.func = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                     attr='math', ctx=ast.Load()),
                attr=node.func.id, ctx=ast.Load())
            return node
        # math.cos etc. → aadc.math.*
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and node.func.attr in self._AADC_MATH):
            node.func = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                     attr='math', ctx=ast.Load()),
                attr=node.func.attr, ctx=ast.Load())
            return node
        # bare floor(x) → math.floor(_aadc_passive(x))
        if isinstance(node.func, ast.Name) and node.func.id == 'floor':
            node.func = ast.Attribute(value=ast.Name(id='math', ctx=ast.Load()),
                                       attr='floor', ctx=ast.Load())
            node.args = [ast.Call(func=ast.Name(id='_aadc_passive', ctx=ast.Load()),
                                  args=node.args, keywords=[])]
            return node
        # math.floor(x) → math.floor(_aadc_passive(x))
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and node.func.attr == 'floor'):
            node.args = [ast.Call(func=ast.Name(id='_aadc_passive', ctx=ast.Load()),
                                  args=node.args, keywords=[])]
            return node
        # pow(x, 2.0) → x * x
        if isinstance(node.func, ast.Name) and node.func.id == 'pow' and len(node.args) == 2:
            import copy
            if isinstance(node.args[1], ast.Constant) and node.args[1].value == 2.0:
                return ast.BinOp(left=node.args[0], op=ast.Mult(), right=copy.deepcopy(node.args[0]))
            return ast.BinOp(left=node.args[0], op=ast.Pow(), right=node.args[1])
        # fabs(x) / math.fabs(x) → aadc.math.abs(x)
        if isinstance(node.func, ast.Name) and node.func.id == 'fabs' and len(node.args) == 1:
            node.func = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                     attr='math', ctx=ast.Load()),
                attr='abs', ctx=ast.Load())
            return node
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and node.func.attr == 'fabs'):
            node.func = ast.Attribute(
                value=ast.Attribute(value=ast.Name(id='aadc', ctx=ast.Load()),
                                     attr='math', ctx=ast.Load()),
                attr='abs', ctx=ast.Load())
            return node
        return node


class PythonGenerator:
    """
    Generate a Python module from a CellML file.

    Usage:
        gen = PythonGenerator('model.cellml', output_dir='out')
        py_path = gen.generate()
    """

    def __init__(
        self,
        cellml_path: str,
        output_dir: Optional[str] = None,
        module_name: Optional[str] = None,
        human_readable: bool = True,
        casadi_compat: bool = False,
        aadc_compat: bool = False,
    ):
        self.cellml_path = cellml_path
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(cellml_path))
        self.module_name = module_name or os.path.splitext(os.path.basename(cellml_path))[0]
        self.human_readable = human_readable
        self.casadi_compat = casadi_compat
        self.aadc_compat = aadc_compat
        if casadi_compat and aadc_compat:
            raise ValueError("casadi_compat and aadc_compat are mutually exclusive")

    @staticmethod
    def _make_identifier(text: str) -> str:
        identifier = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
        identifier = re.sub(r"_+", "_", identifier)
        if not identifier:
            identifier = "unnamed"
        if identifier[0].isdigit():
            identifier = f"N_{identifier}"
        return identifier

    def _build_qualified_symbols(self, info_list):
        qualified_names = []
        attr_names = {}
        counts = {}

        for idx, info in enumerate(info_list):
            qualified_name = f"{info['component']}.{info['name']}"
            qualified_names.append(qualified_name)

            base_name = self._make_identifier(
                f"{info['component']}_{info['name']}"
            ).lower()
            count = counts.get(base_name, 0)
            attr_name = base_name if count == 0 else f"{base_name}_{count + 1}"
            counts[base_name] = count + 1
            attr_names[idx] = attr_name

        return qualified_names, attr_names

    @staticmethod
    def _extract_generated_metadata(code: str):
        namespace = {}
        exec(code, namespace)
        return (
            namespace["__version__"],
            namespace["LIBCELLML_VERSION"],
            namespace["STATE_COUNT"],
            namespace["VARIABLE_COUNT"],
            namespace["VOI_INFO"],
            namespace["STATE_INFO"],
            namespace["VARIABLE_INFO"],
        )

    @staticmethod
    def _extract_function_block(code: str, function_name: str) -> str:
        pattern = rf"^def {function_name}\(.*?(?=^def |\Z)"
        match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
        if not match:
            raise ValueError(f"Unable to locate function '{function_name}' in generated code.")
        return match.group(0).rstrip() + "\n"

    def _rewrite_indexed_function(self, function_block: str, state_attrs, variable_attrs, setup_lines) -> str:
        lines = function_block.splitlines()
        def_line = lines[0]
        body = "\n".join(lines[1:])

        body = re.sub(
            r"states\[(\d+)\]",
            lambda match: f"state.{state_attrs[int(match.group(1))]}",
            body,
        )
        body = re.sub(
            r"rates\[(\d+)\]",
            lambda match: f"rate.{state_attrs[int(match.group(1))]}",
            body,
        )
        body = re.sub(
            r"variables\[(\d+)\]",
            lambda match: f"var.{variable_attrs[int(match.group(1))]}",
            body,
        )

        setup = "\n".join(setup_lines)
        if body.strip():
            body = f"{setup}\n{body}" if setup else body
        else:
            body = setup

        return def_line + "\n" + body.rstrip() + "\n"

    @staticmethod
    def _apply_casadi_if_else_transform(function_block: str) -> str:
        """Apply CasADi-compat AST transforms (if_else, guarded division) to a function block."""
        tree = ast.parse(function_block)
        transformed = _CasadiCompatTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed) + "\n"

    @staticmethod
    def _apply_aadc_transform(function_block: str) -> str:
        """Apply AADC-compat AST transforms (iif, aadc.math, iand, etc.) to a function block."""
        tree = ast.parse(function_block)
        transformed = _AadcCompatTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed) + "\n"

    @staticmethod
    def _comparison_helper_lines(casadi_compat: bool, aadc_compat: bool = False) -> list:
        if aadc_compat:
            return [
                "import aadc",
                "",
                "def _aadc_passive(x):",
                "    return x.val() if hasattr(x, 'val') else float(x)",
                "",
                "def lt_func(x, y):",
                "    return x < y",
                "",
                "def leq_func(x, y):",
                "    return x <= y",
                "",
                "def gt_func(x, y):",
                "    return x > y",
                "",
                "def geq_func(x, y):",
                "    return x >= y",
                "",
                "def and_func(x, y):",
                "    return aadc.iand(x, y)",
                "",
                "def max(x, y):",
                "    return aadc.iif(x >= y, x, y)",
                "",
            ]
        if casadi_compat:
            return [
                "def lt_func(x, y):",
                "    return ca.if_else(x < y, 1.0, 0.0)",
                "",
                "def leq_func(x, y):",
                "    return ca.if_else(x <= y, 1.0, 0.0)",
                "",
                "def gt_func(x, y):",
                "    return ca.if_else(x > y, 1.0, 0.0)",
                "",
                "def geq_func(x, y):",
                "    return ca.if_else(x >= y, 1.0, 0.0)",
                "",
                "def and_func(x, y):",
                "    return ca.if_else(x > 0, ca.if_else(y > 0, 1.0, 0.0), 0.0)",
                "",
                "def max(x, y):",
                "    return ca.if_else(x > y, x, y)",
                "",
            ]
        return [
            "def lt_func(x, y):",
            "    return 1.0 if x < y else 0.0",
            "",
            "def leq_func(x, y):",
            "    return 1.0 if x <= y else 0.0",
            "",
            "def gt_func(x, y):",
            "    return 1.0 if x > y else 0.0",
            "",
            "def geq_func(x, y):",
            "    return 1.0 if x >= y else 0.0",
            "",
            "def and_func(x, y):",
            "    return 1.0 if bool(x) & bool(y) else 0.0",
            "",
            "def max(x, y):",
            "    return x if x > y else y",
            "",
        ]

    @staticmethod
    def _format_info_dict(name: str, info: dict) -> str:
        return (
            f'{name} = {{"name": {info["name"]!r}, "units": {info["units"]!r}, '
            f'"component": {info["component"]!r}, "type": VariableType.{info["type"].name}}}'
        )

    @staticmethod
    def _format_info_list(name: str, info_list) -> str:
        lines = [f"{name} = ["]
        for info in info_list:
            lines.append(
                f'    {{"name": {info["name"]!r}, "units": {info["units"]!r}, '
                f'"component": {info["component"]!r}, "type": VariableType.{info["type"].name}}},'
            )
        lines.append("]")
        return "\n".join(lines)

    @staticmethod
    def _format_string_list(name: str, values) -> str:
        lines = [f"{name} = ["]
        for value in values:
            lines.append(f"    {value!r},")
        lines.append("]")
        return "\n".join(lines)

    def _build_utilities_code(
        self,
        raw_code: str,
        version: str,
        libcellml_version: str,
        state_count: int,
        variable_count: int,
        voi_info: dict,
        state_info,
        variable_info,
        state_names,
        state_attrs,
        variable_names,
        variable_attrs,
    ) -> str:
        header = raw_code.splitlines()[0]

        utility_parts = [
            header,
            "",
            "from enum import Enum",
            "from math import *",
        ]
        if self.casadi_compat:
            utility_parts.extend([
                "import casadi as ca",
                "",
            ])
        if self.aadc_compat:
            utility_parts.extend([
                "import math",
                "import aadc",
                "",
            ])
        utility_parts.extend([
            "",
            f"__version__ = {version!r}",
            f"LIBCELLML_VERSION = {libcellml_version!r}",
            f"STATE_COUNT = {state_count}",
            f"VARIABLE_COUNT = {variable_count}",
            "",
            "class VariableType(Enum):",
            "    VARIABLE_OF_INTEGRATION = 0",
            "    STATE = 1",
            "    CONSTANT = 2",
            "    COMPUTED_CONSTANT = 3",
            "    ALGEBRAIC = 4",
            "",
            self._format_info_dict("VOI_INFO", voi_info),
            "",
            self._format_info_list("STATE_INFO", state_info),
            "",
            self._format_info_list("VARIABLE_INFO", variable_info),
            "",
            self._format_string_list("STATE_QUALIFIED_NAMES", state_names),
            self._format_string_list("STATE_ATTR_NAMES", [state_attrs[idx] for idx in range(len(state_attrs))]),
            "STATE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(STATE_QUALIFIED_NAMES)}",
            "STATE_ATTR_TO_INDEX = {name: idx for idx, name in enumerate(STATE_ATTR_NAMES)}",
            "",
            self._format_string_list("VARIABLE_QUALIFIED_NAMES", variable_names),
            self._format_string_list("VARIABLE_ATTR_NAMES", [variable_attrs[idx] for idx in range(len(variable_attrs))]),
            "VARIABLE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(VARIABLE_QUALIFIED_NAMES)}",
            "VARIABLE_ATTR_TO_INDEX = {name: idx for idx, name in enumerate(VARIABLE_ATTR_NAMES)}",
            "",
            "class _ArrayView:",
            '    __slots__ = ("_values",)',
            "    NAME_TO_INDEX = {}",
            "    QUALIFIED_NAMES = []",
            "",
            "    def __init__(self, values):",
            '        object.__setattr__(self, "_values", values)',
            "",
            "    def __getattr__(self, name):",
            "        idx = self.NAME_TO_INDEX.get(name)",
            "        if idx is None:",
            '            raise AttributeError(f\"{type(self).__name__} has no attribute {name!r}\")',
            "        return self._values[idx]",
            "",
            "    def __setattr__(self, name, value):",
            '        if name == "_values":',
            '            object.__setattr__(self, name, value)',
            "            return",
            "        idx = self.NAME_TO_INDEX.get(name)",
            "        if idx is None:",
            '            raise AttributeError(f\"{type(self).__name__} has no attribute {name!r}\")',
            "        self._values[idx] = value",
            "",
            "    def as_dict(self):",
            "        return {name: self._values[idx] for idx, name in enumerate(self.QUALIFIED_NAMES)}",
            "",
            "    def __dir__(self):",
            "        return sorted(set(object.__dir__(self)) | set(self.NAME_TO_INDEX.keys()))",
            "",
            "    def __repr__(self):",
            '        return f\"{type(self).__name__}({self.as_dict()})\"',
            "",
            "class StateView(_ArrayView):",
            "    NAME_TO_INDEX = STATE_ATTR_TO_INDEX",
            "    QUALIFIED_NAMES = STATE_QUALIFIED_NAMES",
            "",
            "class VarView(_ArrayView):",
            "    NAME_TO_INDEX = VARIABLE_ATTR_TO_INDEX",
            "    QUALIFIED_NAMES = VARIABLE_QUALIFIED_NAMES",
            "",
            "class RateView(_ArrayView):",
            "    NAME_TO_INDEX = STATE_ATTR_TO_INDEX",
            "    QUALIFIED_NAMES = STATE_QUALIFIED_NAMES",
            "",
            "def create_states_array():",
            "    return [nan] * STATE_COUNT",
            "",
            "def create_variables_array():",
            "    return [nan] * VARIABLE_COUNT",
            "",
            "def describe_states(states):",
            "    return StateView(states).as_dict()",
            "",
            "def describe_variables(variables):",
            "    return VarView(variables).as_dict()",
            "",
        ])
        utility_parts.extend(self._comparison_helper_lines(self.casadi_compat, self.aadc_compat))

        return "\n".join(utility_parts) + "\n"

    def _build_main_code(self, raw_code: str, utility_filename: str, state_attrs, variable_attrs) -> str:
        header = raw_code.splitlines()[0]
        initialise_variables = self._rewrite_indexed_function(
            self._extract_function_block(raw_code, "initialise_variables"),
            state_attrs,
            variable_attrs,
            [
                "    state = StateView(states)",
                "    var = VarView(variables)",
            ],
        )
        compute_computed_constants = self._rewrite_indexed_function(
            self._extract_function_block(raw_code, "compute_computed_constants"),
            state_attrs,
            variable_attrs,
            ["    var = VarView(variables)"],
        )
        compute_rates = self._rewrite_indexed_function(
            self._extract_function_block(raw_code, "compute_rates"),
            state_attrs,
            variable_attrs,
            [
                "    state = StateView(states)",
                "    rate = RateView(rates)",
                "    var = VarView(variables)",
            ],
        )
        compute_variables = self._rewrite_indexed_function(
            self._extract_function_block(raw_code, "compute_variables"),
            state_attrs,
            variable_attrs,
            [
                "    state = StateView(states)",
                "    rate = RateView(rates)",
                "    var = VarView(variables)",
            ],
        )

        if self.casadi_compat:
            initialise_variables = self._apply_casadi_if_else_transform(initialise_variables)
            compute_computed_constants = self._apply_casadi_if_else_transform(
                compute_computed_constants
            )
            compute_rates = self._apply_casadi_if_else_transform(compute_rates)
            compute_variables = self._apply_casadi_if_else_transform(compute_variables)

        if self.aadc_compat:
            initialise_variables = self._apply_aadc_transform(initialise_variables)
            compute_computed_constants = self._apply_aadc_transform(compute_computed_constants)
            compute_rates = self._apply_aadc_transform(compute_rates)
            compute_variables = self._apply_aadc_transform(compute_variables)

        export_names = [
            "__version__",
            "LIBCELLML_VERSION",
            "STATE_COUNT",
            "VARIABLE_COUNT",
            "VariableType",
            "VOI_INFO",
            "STATE_INFO",
            "VARIABLE_INFO",
            "STATE_QUALIFIED_NAMES",
            "VARIABLE_QUALIFIED_NAMES",
            "STATE_NAME_TO_INDEX",
            "VARIABLE_NAME_TO_INDEX",
            "STATE_ATTR_NAMES",
            "VARIABLE_ATTR_NAMES",
            "STATE_ATTR_TO_INDEX",
            "VARIABLE_ATTR_TO_INDEX",
            "StateView",
            "VarView",
            "RateView",
            "create_states_array",
            "create_variables_array",
            "describe_states",
            "describe_variables",
            "lt_func",
            "leq_func",
            "gt_func",
            "geq_func",
            "and_func",
            "max",
        ]
        if self.aadc_compat:
            export_names.append("_aadc_passive")
        export_block = "\n".join(f"    {name!r}," for name in export_names)

        main_parts = [
            header,
            "",
            "from math import *",
            "from pathlib import Path",
            "import importlib.util as _importlib_util",
        ]
        if self.casadi_compat:
            main_parts.extend([
                "import casadi as ca",
                "",
            ])
        if self.aadc_compat:
            main_parts.extend([
                "import math",
                "import aadc",
                "",
            ])
        main_parts.extend([
            "",
            f"_UTILITIES_PATH = Path(__file__).with_name({utility_filename!r})",
            '_UTILITIES_SPEC = _importlib_util.spec_from_file_location(f"{__name__}_utilities", _UTILITIES_PATH)',
            "_UTILITIES = _importlib_util.module_from_spec(_UTILITIES_SPEC)",
            "if _UTILITIES_SPEC.loader is None:",
            '    raise ImportError(f"Unable to load utilities module: {_UTILITIES_PATH}")',
            "_UTILITIES_SPEC.loader.exec_module(_UTILITIES)",
            "",
            "for _name in (",
            export_block,
            "):",
            "    globals()[_name] = getattr(_UTILITIES, _name)",
            "del _name",
            "",
            initialise_variables.rstrip(),
            "",
            compute_computed_constants.rstrip(),
            "",
            compute_rates.rstrip(),
            "",
            compute_variables.rstrip(),
            "",
        ])

        return "\n".join(main_parts) + "\n"

    def _make_human_readable(self, code: str):
        (
            version,
            libcellml_version,
            state_count,
            variable_count,
            voi_info,
            state_info,
            variable_info,
        ) = self._extract_generated_metadata(code)
        state_names, state_attrs = self._build_qualified_symbols(state_info)
        variable_names, variable_attrs = self._build_qualified_symbols(variable_info)
        utility_filename = f"{self.module_name}_utilities.py"

        utilities_code = self._build_utilities_code(
            code,
            version,
            libcellml_version,
            state_count,
            variable_count,
            voi_info,
            state_info,
            variable_info,
            state_names,
            state_attrs,
            variable_names,
            variable_attrs,
        )
        main_code = self._build_main_code(code, utility_filename, state_attrs, variable_attrs)
        return main_code, utility_filename, utilities_code

    def _parse_model(self):
        parser = lc.Parser(False)  # non-strict to allow 1.1 models
        with open(self.cellml_path, "r", encoding="utf-8") as fh:
            model = parser.parseModel(fh.read())
        if parser.errorCount() > 0:
            msgs = "\n".join(parser.error(i).description() for i in range(parser.errorCount()))
            raise ValueError(f"CellML parse errors:\n{msgs}")
        return model

    def _resolve_and_flatten(self, model):
        if cellml_utils is not None:
            importer = cellml_utils.resolve_imports(model, os.path.dirname(self.cellml_path), False)
            flat_model = cellml_utils.flatten_model(model, importer)
            return flat_model

        importer = lc.Importer()
        importer.resolveImports(model, os.path.dirname(self.cellml_path))
        return importer.flattenModel(model)

    def _analyse(self, flat_model):
        analyser = lc.Analyser()
        analyser.analyseModel(flat_model)
        libcellml_utils.print_issues(analyser)
        analysed_model = cellml_utils.get_analysed_model(analyser)
        if analysed_model.type() != lc.AnalyserModel.Type.ODE:
            raise ValueError(
                'Generated model is not a valid ODE model according to'
                'libcellml, Model invalid for Python generation. '
                'Issues to fix in CellML to get python generation working are \n'
                'shown above.'
            )
        return analysed_model

    def generate(self) -> str:
        """Generate Python code and return the output file path."""
        model = self._parse_model()
        flat_model = self._resolve_and_flatten(model)
        analysed_model = self._analyse(flat_model)

        profile = lc.GeneratorProfile(lc.GeneratorProfile.Profile.PYTHON)
        generator = lc.Generator()
        generator.setProfile(profile)
        generator.setModel(analysed_model)

        code = generator.implementationCode()
        utilities_filename = None
        utilities_code = None
        if self.human_readable and code:
            code, utilities_filename, utilities_code = self._make_human_readable(code)

        if code == "":
            raise ValueError("Generated Python code is empty. Model invalid")

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.module_name}.py")
        if utilities_filename is not None and utilities_code is not None:
            utilities_path = os.path.join(self.output_dir, utilities_filename)
            with open(utilities_path, "w", encoding="utf-8") as fh:
                fh.write(utilities_code)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(code)

        if self.casadi_compat:
            compile(code, output_path, "exec")
            if utilities_code is not None:
                utilities_path = os.path.join(self.output_dir, utilities_filename)
                compile(utilities_code, utilities_path, "exec")
            print("CasADi model generation successful (compile check passed).")
            return output_path

        if self.aadc_compat:
            compile(code, output_path, "exec")
            if utilities_code is not None:
                utilities_path = os.path.join(self.output_dir, utilities_filename)
                compile(utilities_code, utilities_path, "exec")
            print("AADC model generation successful (compile check passed).")
            return output_path

        sim_helper = PythonSimulationHelper(output_path, dt=0.00001, sim_time=0.00001)
        sim_helper.set_solve_ivp_method("BDF")
        success = sim_helper.run()

        if success:
            print("Model generation has been successful.")
            return output_path

        print("Model generation has failed. Or the simulation fails when trying to simulate in Python")
        raise ValueError(
            "Model generation has failed. Or the simulation fails when trying to simulate"
            "for 0.00001 seconds in Python"
        )

