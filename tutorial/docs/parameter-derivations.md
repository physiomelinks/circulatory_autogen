# Deriving parameters from experimental data

Many parameters are calculated from measurements rather than measured: a bed's resistance from
the mean arterial and central venous pressures and its flow, a compliance from stroke volume and
pulse pressure. A **derivations file** records each calculation: its inputs, with their values,
uncertainties and sources, the equation, and the parameters it sets. `cuflynx-derive` evaluates it,
checks the values cached in the parameters file, and with `--write` updates them.

Derivations are for data only. They never run inside a simulation; a parameter you calibrate stays a
model parameter in `params_for_id`, and a derived value can be its starting value.

## The inputs: `input_items` in obs_data

A number a derivation reads that is not a calibration target goes in the obs_data's
`input_items`, beside `data_items` and `prediction_items`. Input items are never scored, plotted or
validated against.

```json
"input_items": [
  {"data_item_name": "CVP_Smith2001", "value": 5.0, "std": 1.0, "unit": "mmHg",
   "source": "Smith 2001, Table 2, supine adults", "data_reference": "Smith2001"},
  {"data_item_name": "CO_Rowell1986", "value": 1e-4, "std": 1e-5, "unit": "m3_per_s",
   "source": "Rowell 1986", "comment": "6 L/min"}
]
```

| Key | |
|---|---|
| `data_item_name` | Required. Unique across the file's data, prediction and input items. |
| `value`, `unit` | Required. One number and its unit (a CellML unit name). |
| `std` | Its standard deviation. |
| `source`, `data_reference`, `comment`, `species`, `location` | Where it comes from; not read. |
| `prob_dist_params` | The number as a distribution; not read yet. |

A derivation may also read a `data_item` with a constant value, when the same number is a
calibration target and an input (mean arterial pressure, say): then the target and the input are
one number.

## The derivations file

`<name>_derivations.json` sits beside the `<name>_parameters.json` it sets, with its CellML and
Python files next to it or below it. Paths in it are relative to it.

```json
{"derivations": [
  {"name": "R_T0",
   "inputs": {"MAP": {"data_item": "MAP"},
              "CVP": {"input_item": "CVP_Smith2001"},
              "Q":   {"parameter": "Systemic_T/Q_T0"}},
   "outputs": {"R": "Systemic_T/R_T0"},
   "cellml": "derivations/systemic_resistance.cellml",
   "comment": "the bed's resistance from its pressure drop and flow"},
  {"name": "Q_T0",
   "inputs": {"CO": {"input_item": "CO_Rowell1986"},
              "frac": {"parameter": "Systemic_T/frac"}},
   "outputs": {"Q": "Systemic_T/Q_T0"},
   "python": "derivations/derivation_funcs.py:flow_share"}]}
```

- **`inputs`** map each symbol to exactly one source: `parameter` (a path in the parameters file,
  as in `params_for_id`: `instance/variable`, `heart/lv/E_A`, `global/R`), `input_item` or
  `data_item`. Its units are the source's.
- **`outputs`** map each symbol to the parameter it sets. The parameter must be in the parameters
  file, with its units. Each parameter is set by one derivation at most.
- **Chains**: an input may be another derivation's output (above, `R_T0` reads `Q_T0`).
  Derivations run in the order their inputs need, whatever their order in the file; a loop is an
  error.
- **Method**: exactly one of `cellml` or `python`.

The structure is defined by `src/libcuflynx/schemas/derivations.schema.json`.

### A CellML derivation model (preferred)

The equation is a CellML 2.0 model. Its inputs are constants named as the input symbols (their
initial values are placeholders); its outputs are computed constants named as the output symbols.

```xml
<component name="derivation">
  <variable name="MAP" units="Pa" initial_value="0"/>
  <variable name="CVP" units="Pa" initial_value="0"/>
  <variable name="Q" units="m3_per_s" initial_value="1"/>
  <variable name="R" units="J_s_per_m6"/>
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><eq/><ci>R</ci>
      <apply><divide/><apply><minus/><ci>MAP</ci><ci>CVP</ci></apply><ci>Q</ci></apply>
    </apply>
  </math>
</component>
```

libCellML validates and analyses the model, and generates its Python code, which is evaluated
once. Any issue is an error, including units that do not balance in an equation. An input whose
units differ from its variable's but have the same dimension is converted (MAP in mmHg into `Pa`
above, if the model defines `mmHg`); so is an output whose parameter is in other units of the same
dimension. Units of different dimensions are an error. Implicit equations are refused: use a
Python function for those.

You can write the model in CellML Text and convert it with physiomelinks' `cellml-text-editor`, or
edit it in PhLynx's Instance Editor (the CellML Text and Math Editor tabs) and export the
component.

### A Python function

For a calculation CellML cannot state (an implicit equation, a network solve), name a function as
`file.py:function`:

```python
def flow_share(inputs, units):
    """The flow through a bed: its share of the cardiac output."""
    return {'Q': inputs['CO'] * inputs['frac']}
```

It is called with `inputs` (`{symbol: value}`, in each source's units) and `units` (`{symbol:
units}` for the inputs and outputs), and returns `{output symbol: value}` in the output parameters'
units: it converts units itself. A template is in `funcs_user/derivation_funcs_example.py`.

## The parameters file

A derived parameter keeps its value, as a cache every reader of the parameters file uses as it is,
and names its derivation:

```json
"R_T0": {"value": 146654199.99999994, "units": "J_s_per_m6",
         "data_reference": "derived; derivation R_T0", "derivation": "R_T0"}
```

## `cuflynx-derive`

```
cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json
cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json --write
```

The first evaluates every derivation and prints each parameter's cached and derived value. It exits
0 when every cached value matches within a relative tolerance of 1e-9 and 1 otherwise, so a test or
CI job can run it. A parameter that names a derivation that does not set it is reported too.
`--write` writes the derived values into the parameters file, and adds `derivation` to each output
parameter; every other key and parameter is kept.

`--obs` may be given several times; `--parameters` names the parameters file when it is not
`<name>_parameters.json` beside the derivations file. From Python:

```python
from libcuflynx.derivations import (check_derivations, evaluate_derivations, load_derivations,
                                    read_parameters)
doc = read_parameters('BVC_parameters.json')
results = evaluate_derivations(load_derivations('BVC_derivations.json'), doc, ['BVC_obs_data.json'])
print(check_derivations(results, doc))   # [] when the cache is current
```

Each result carries its inputs with their values, units, standard deviations and sources, for a
review page or a table.

## Packing

`pack_workflow` (and `cuflynx-calibration-workflow --pack`) copies the derivations file beside each
parameters file it packs, with the CellML and Python files it reads at the same paths relative to
it, so they go into the OMEX archive CUFLynx makes from the pack. A derivation's files must be in
the derivations file's directory or below it. A module version from a library is packed whole, with
its derivations.
