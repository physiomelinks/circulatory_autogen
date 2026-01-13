"""
Concise Python code generator for CellML models using libCellML.

This class parses a CellML file, resolves imports, analyses the model,
and emits a ready-to-run Python module.
"""
import os
from typing import Optional

try:
    import libcellml as lc
except ImportError as e:  # pragma: no cover - runtime environment check
    raise ImportError("libcellml is required to generate Python models.") from e

# Use existing helper utilities for lenient import resolution (CellML 1.1 tolerant)
try:
    import utilities.libcellml_helper_funcs as cellml_utils
except ImportError:
    cellml_utils = None


class PythonGenerator:
    """
    Generate a Python module from a CellML file.

    Usage:
        gen = PythonGenerator('model.cellml', output_dir='out')
        py_path = gen.generate()
    """

    def __init__(self, cellml_path: str, output_dir: Optional[str] = None, module_name: Optional[str] = None):
        self.cellml_path = cellml_path
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(cellml_path))
        self.module_name = module_name or os.path.splitext(os.path.basename(cellml_path))[0]

    def _parse_model(self):
        parser = lc.Parser(False)  # non-strict to allow 1.1 models
        with open(self.cellml_path, "r", encoding="utf-8") as fh:
            model = parser.parseModel(fh.read())
        if parser.errorCount() > 0:
            msgs = "\n".join(parser.error(i).description() for i in range(parser.errorCount()))
            raise ValueError(f"CellML parse errors:\n{msgs}")
        return model

    def _resolve_and_flatten(self, model):
        # Prefer the project helper to tolerate CellML 1.1 imports; fall back to libcellml importer.
        if cellml_utils is not None:
            importer = cellml_utils.resolve_imports(model, os.path.dirname(self.cellml_path), False)
            flat_model = cellml_utils.flatten_model(model, importer)
            return flat_model

        importer = lc.Importer()
        importer.resolveImports(model, os.path.dirname(self.cellml_path))
        flat_model = importer.flattenModel(model)
        # For legacy 1.1 models, errors can be benign; do not fail hard here.
        return flat_model

    def _analyse(self, flat_model):
        analyser = lc.Analyser()
        analyser.analyseModel(flat_model)
        # Keep going even if there are analyser issues to match existing generation behaviour.
        return analyser.model()

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
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.module_name}.py")
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        return output_path

