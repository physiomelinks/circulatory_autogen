#_______python paths _________
# python_path is the interpreter run_pytest.sh runs the tests with: a venv with
# `pip install -e ".[dev]"` in it. Relative paths are from the repository root.
# (The launchers in this directory run the console commands on PATH instead; set
# CUFLYNX_PYTHON to pick an interpreter for them.)
python_path=venv/bin/python
