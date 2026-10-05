'''
Writing a model and its calibration up: LaTeX symbols for every variable
(``variable_mapping``), the equations module by module (``model_latex``), flow charts of a
calibration (``flowchart``) and a methods section built from them (``methods``).

Commands: ``cuflynx-variable-mapping`` and ``cuflynx-methods-latex``.
'''

from libcuflynx.reporting.methods import calibration_methods_latex, workflow_methods_latex
from libcuflynx.reporting.model_latex import module_sections
from libcuflynx.reporting.variable_mapping import (check_variable_mapping,
                                                   default_variable_mapping,
                                                   read_variable_mapping,
                                                   update_variable_mapping_file,
                                                   variable_mapping, write_variable_mapping)

__all__ = ['calibration_methods_latex', 'check_variable_mapping', 'default_variable_mapping',
           'module_sections', 'read_variable_mapping', 'update_variable_mapping_file',
           'variable_mapping', 'workflow_methods_latex', 'write_variable_mapping']
