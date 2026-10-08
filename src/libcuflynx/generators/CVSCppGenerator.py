'''
Compatibility module: the C++ generator now lives in libcuflynx.generators.cpp.

``CVS0DCppGenerator`` renders the model's C++ from libCellML output plus Jinja2 templates
(libcuflynx/generators/cpp/templates); ``CVS1DPythonGenerator`` writes the Python 1D solver's
input files (libcuflynx/generators/Python1DGenerator.py). Import them from here or from their new
modules.
'''

from libcuflynx.generators.cpp.generator import CVS0DCppGenerator, CppGenerationError
from libcuflynx.generators.Python1DGenerator import CVS1DPythonGenerator

__all__ = ['CVS0DCppGenerator', 'CppGenerationError', 'CVS1DPythonGenerator']
