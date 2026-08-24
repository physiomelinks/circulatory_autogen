"""Deprecated alias for :mod:`libcuflynx.generators`, removed in 0.6.0.

Importing this name emits a single ``DeprecationWarning`` and then hands back the real
``libcuflynx.generators`` package -- the same module object, not a copy -- so that
``isinstance`` checks and monkeypatching against classes reached through the old name
keep working. See ``libcuflynx/_deprecated_aliases.py`` for how, and why it takes a
meta path finder to make ``import generators.<submodule>`` identical too.
"""
from libcuflynx._deprecated_aliases import install_shim

install_shim(__name__)
