"""The pint UnitRegistry of utility_funcs: one per process."""
import pytest


@pytest.mark.unit
def test_unit_converters_share_one_unit_registry():
    """Building a pint UnitRegistry takes about a second, so a process builds one."""
    from libcuflynx.utilities.utility_funcs import UnitConverter, unit_registry
    assert UnitConverter().ureg is UnitConverter().ureg is unit_registry()
