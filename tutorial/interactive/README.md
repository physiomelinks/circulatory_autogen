## Interactive Tutorials

These interactive tutorials show how generation, sensitivity analysis, and calibration can be done completely through python (and PhLynx for generation). 

- `generation_and_calibration.ipynb`: generates and simulates a cardiovascular model, then
  calibrates it as a calibration workflow (`WorkflowBuilder`): a sensitivity analysis, a
  reduction to the most influential parameters, and a calibration, each added and run in its own
  cell. It ends by writing `calibration_workflow.json`, which `cuflynx-calibration-workflow` and
  CUFLynx's Workflow tab run with the same results.
- `workflow_quickstart.ipynb`: every `WorkflowBuilder` call on the 3-compartment model, in a few
  minutes.
- `image_to_hemodynamics_model.ipynb`: a module array from a segmented image.

# TO RUN Jupyter Notebook 

In a venv with libcuflynx installed (`pip install -e ".[dev]"` from the repository root) and
Jupyter (`pip install jupyter`), run:

```
jupyter notebook tutorial/interactive/generation_and_calibration.ipynb
```

or `workflow_quickstart.ipynb`, `image_to_hemodynamics_model.ipynb` in the same folder.
