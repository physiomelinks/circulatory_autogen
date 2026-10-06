Modules the tests use that are not part of circulatory-autogen-modules: fixtures for generator
features (`test_modules`: ports and initial states) and the `Simple_ODE_Benchmark` model.
`tests/conftest.py` adds this directory to `CUFLYNX_MODULE_LIBRARY`, after the library itself.
