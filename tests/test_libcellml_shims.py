"""The libCellML 0.6.x / 0.7.0 compatibility shims (issue #271).

`pyproject.toml` pins `libcellml>=0.6.3,<0.7.0`, so **the 0.7.0 branches of these shims cannot
execute in CI or in any user install**. They are dead code today, which means without these tests
they would first run at the exact moment the pin is lifted -- the worst possible time to discover
a mistake in them.

Both sides are exercised here against fakes rather than a real libCellML, so neither branch
depends on which version happens to be installed. This does not do the 0.7.0 migration (that is
still #271: the generated-Python output and unit flattening changed too); it removes the risk
that the *renames* are wrong when the pin comes off.
"""
import pytest

from libcuflynx.utilities.libcellml_helper_funcs import (
    _generator_is_pre_0_7, generate_implementation_code, generate_interface_code,
    get_analysed_model)


class _Analyser06:
    """0.6.x: the accessor is model()."""
    def model(self):
        return 'analysed-model-06'


class _Analyser07:
    """0.7.0: renamed to analyserModel()."""
    def analyserModel(self):
        return 'analysed-model-07'


class _Generator06:
    """0.6.x: configured via setModel()/setProfile(), then called with no arguments."""
    def __init__(self):
        self.model = None
        self.profile = None

    def setProfile(self, profile):
        self.profile = profile

    def setModel(self, model):
        self.model = model

    def implementationCode(self):
        return f'impl-06({self.model},{self.profile})'

    def interfaceCode(self):
        return f'iface-06({self.model},{self.profile})'


class _Generator07:
    """0.7.0: the setters are gone; the model and profile are arguments."""
    def implementationCode(self, analysed_model, profile):
        return f'impl-07({analysed_model},{profile})'

    def interfaceCode(self, analysed_model, profile):
        return f'iface-07({analysed_model},{profile})'


@pytest.mark.unit
def test_analysed_model_accessor_dispatches_both_ways():
    assert get_analysed_model(_Analyser06()) == 'analysed-model-06'
    assert get_analysed_model(_Analyser07()) == 'analysed-model-07'


@pytest.mark.unit
def test_the_version_probe_keys_off_the_setter_that_only_0_6_has():
    assert _generator_is_pre_0_7(_Generator06()) is True
    assert _generator_is_pre_0_7(_Generator07()) is False


@pytest.mark.unit
def test_implementation_code_passes_model_and_profile_on_both_versions():
    """The 0.6 branch must set *both* model and profile before calling, and the 0.7 branch must
    forward both as arguments. Dropping either would produce code generated against a default
    profile or an unset model, which fails late and confusingly."""
    gen06 = _Generator06()
    assert generate_implementation_code(gen06, 'MODEL', 'PROFILE') == 'impl-06(MODEL,PROFILE)'
    assert gen06.model == 'MODEL' and gen06.profile == 'PROFILE'

    assert generate_implementation_code(_Generator07(), 'MODEL', 'PROFILE') == \
        'impl-07(MODEL,PROFILE)'


@pytest.mark.unit
def test_interface_code_passes_model_and_profile_on_both_versions():
    gen06 = _Generator06()
    assert generate_interface_code(gen06, 'MODEL', 'PROFILE') == 'iface-06(MODEL,PROFILE)'
    assert gen06.model == 'MODEL' and gen06.profile == 'PROFILE'

    assert generate_interface_code(_Generator07(), 'MODEL', 'PROFILE') == 'iface-07(MODEL,PROFILE)'


@pytest.mark.unit
def test_a_0_7_generator_is_never_handed_the_0_6_call_shape():
    """Regression guard for the failure that would actually happen if the probe inverted: calling
    0.7's implementationCode() with no arguments raises TypeError, and calling 0.6's with two
    does the same. Assert the shim never produces either."""
    with pytest.raises(TypeError):
        _Generator07().implementationCode()
    with pytest.raises(TypeError):
        _Generator06().implementationCode('MODEL', 'PROFILE')
    # ...and the shim avoids both
    generate_implementation_code(_Generator07(), 'M', 'P')
    generate_implementation_code(_Generator06(), 'M', 'P')


@pytest.mark.unit
def test_the_pin_is_still_in_place_so_these_branches_remain_untested_in_the_wild():
    """If the pin is lifted, this test should be revisited alongside the rest of #271 -- the real
    migration (generated-Python output, unit flattening) is not covered by these fakes."""
    import pathlib
    pyproject = (pathlib.Path(__file__).resolve().parent.parent / 'pyproject.toml').read_text()
    if 'libcellml>=0.6.3,<0.7.0' not in pyproject:
        pytest.fail(
            "The libcellml <0.7.0 pin has changed. These shim tests cover the API renames only; "
            "confirm the generated-Python and unit-flattening migration in #271 is done, then "
            "update this test.")
