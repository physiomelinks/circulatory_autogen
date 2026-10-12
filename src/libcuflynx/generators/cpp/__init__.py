'''
Template-based C++ generation (model_type: cpp).

``generator.CVS0DCppGenerator`` writes libCellML's C code for the model unmodified and renders
the wrapper (solver, output, external variables, driver, CMake build) from ``templates/``.
Couplings to other models are declared in module configs through ``api`` blocks (``api.py``).

Kept import-free so ``libcuflynx.generators.cpp.api`` can be imported by the parsers without
pulling in the generators.
'''
