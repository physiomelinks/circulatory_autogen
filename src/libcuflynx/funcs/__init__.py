"""Built-in cost, operation and modifier functions shipped with libcuflynx (issue #433).

These used to live in the repository's ``funcs_user/`` directory and were imported by bare
module name (``import cost_funcs_user``), which only worked from a source checkout with that
directory on ``sys.path``. They are library code -- a library must not ask its users to edit
files inside it -- so they now live here and are imported as ``libcuflynx.funcs.*``.

Adding *your own* funcs no longer means editing these modules. Write them in your own file and
point the config at it:

    cost_funcs_external_path: funcs_user/my_costs.py
    operation_funcs_external_path: funcs_user/my_ops.py
    modifier_funcs_external_path: funcs_user/my_modifiers.py

The external file's funcs are merged into the same registry as the built-ins, with their
``@is_MLE`` / ``@cost_combiner`` / ``@differentiable`` / ``@series_to_constant`` /
``@modifier_func`` decorators intact, and a name clash resolves in the external file's favour.
See ``funcs_user/README.md`` in the repository for worked templates.
"""
