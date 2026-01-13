# Running Tests

This project uses pytest for testing. Since the project depends on OpenCOR's Python shell, tests must be run using OpenCOR's bundled Python interpreter.

## Quick Start

Run all tests using the provided script:

```bash
./run_pytest.sh
```

This script automatically:
1. Sources the OpenCOR Python shell path from `user_run_files/opencor_pythonshell_path.sh`
2. Installs pytest and related packages if needed
3. Runs pytest with the OpenCOR Python interpreter

## Running Specific Tests

You can pass any pytest arguments to the script:

```bash
# Run a specific test file
./run_pytest.sh tests/test_example.py

# Run tests matching a pattern
./run_pytest.sh -k "test_name_pattern"

# Run with coverage
./run_pytest.sh --cov=src --cov-report=html

# Run only unit tests (skip slow/integration tests)
./run_pytest.sh -m "not slow and not integration"

# Run with verbose output
./run_pytest.sh -v

# Run in parallel
./run_pytest.sh -n NUM_RANKS 

# Run in parallel and verbose (this is the standard way to test everything)

./run_pytest.sh -n NUM_RANKS -v -s

# Run only the optimiser method comparisons
./run_pytest.sh -n 8 tests/test_param_id.py::test_compare_optimisers -v -s
```


## Manual Setup

If you prefer to run pytest manually, you can:

1. Source the OpenCOR Python path:
   ```bash
   source user_run_files/opencor_pythonshell_path.sh
   ```

2. Install pytest in the OpenCOR Python environment:
   ```bash
   ${opencor_pythonshell_path} -m pip install pytest pytest-cov pytest-mpi pytest-xdist
   ```

3. Run pytest:
   ```bash
   ${opencor_pythonshell_path} -m pytest
   ```

## Test Structure

The test suite includes:
- `test_autogeneration.py` - Autogeneration tests (marked as `integration` and `slow`)
- `test_param_id.py` - Parameter identification tests (marked as `integration`, `slow`, and `mpi`)
- `test_sensitivity_analysis.py` - Sensitivity analysis tests (marked as `integration`, `slow`, and `mpi`)
- `conftest.py` - Shared fixtures and pytest configuration

All test files follow pytest conventions:
- Test files are named `test_*.py`
- Test functions are named `test_*` following the pattern `test_function_behavior()`
- Tests use parametrization to avoid repetition
- Tests use fixtures to eliminate side effects
- Tests are deterministic (random seeds are set automatically)
- Tests use proper assertions instead of print statements

Use pytest markers to categorize tests:
- `@pytest.mark.slow` - for slow-running tests
- `@pytest.mark.integration` - for integration tests
- `@pytest.mark.unit` - for unit tests
- `@pytest.mark.mpi` - for tests requiring MPI

## Running Specific Test Categories

```bash
# Run only fast tests (skip slow ones)
./run_pytest.sh -m "not slow"

# Run only unit tests
./run_pytest.sh -m "unit"

# Run only integration tests
./run_pytest.sh -m "integration"

# Skip MPI tests (if you don't have MPI set up)
./run_pytest.sh -m "not mpi"
```

## Fixtures

The `conftest.py` file provides:
- `project_root` - Returns the project root directory
- `opencor_python_path` - Returns the OpenCOR Python shell path (if available)

