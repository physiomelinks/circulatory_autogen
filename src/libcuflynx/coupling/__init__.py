'''
Coupling generated C++ 0D models to external Python models (e.g. FEniCS PDE models).

An external model is a row of the module array whose module config has an ``api`` block with
``"transport": "python"``. Generating the model as C++ (``model_type: cpp``) then also writes

* ``model0d_capi.cpp``, a C interface built as the shared library ``model0d_capi``, and
* ``external_models.json``, the exchange table and each external model's class and parameters.

``run_coupled(model_dir)`` (or the ``cuflynx-couple`` command) builds the library if needed,
creates the 0D model and every external model, and steps them together. See
``tutorial/docs/external-coupling`` for the whole workflow, and ``ExternalModel`` for the class
an external model implements.
'''

from libcuflynx.coupling.base import ExternalModel
from libcuflynx.coupling.model0d_lib import Model0dLibrary, Model0dError
from libcuflynx.coupling.runner import CoupledResult, run_coupled

__all__ = ['ExternalModel', 'Model0dLibrary', 'Model0dError', 'CoupledResult', 'run_coupled']
