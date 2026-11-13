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
   "data_reference": "Smith2001", "comment": "Table 2, supine adults"},
  {"data_item_name": "CO_Rowell1986", "value": 1e-4, "std": 1e-5, "unit": "m3_per_s",
   "data_reference": "Rowell1986", "comment": "6 L/min"}
]
```

| Key | |
|---|---|
| `data_item_name` | Required. Unique across the file's data, prediction and input items. |
| `value`, `unit` | Required. One number and its unit (a CellML unit name). |
| `std` | Its standard deviation. |
| `data_reference` | Where it comes from (a citation key, DOI or reference); carried into each derivation's inputs. An input item does not take `source`: one with it is refused, naming `data_reference` (a data item's `source` is still read as its `data_reference`, with a warning). |
| `comment`, `species`, `location` | Free text; not read. |
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
- **`fixed_constants`** (`cellml` only): constants of the CellML model that keep their initial
  value from the CellML file, such as the gas constant. See below.

The structure is defined by `src/libcuflynx/schemas/derivations.schema.json`.

### A CellML derivation model (preferred)

The equation is a CellML 2.0 model. Its inputs are constants named as the input symbols (their
initial values are placeholders); its outputs are computed constants named as the output symbols.
Every constant of the model must be an input or a fixed constant: one the derivation leaves out
is an error, not its placeholder. The model may import units or components from other CellML files
beside it or below it.

A constant whose initial value is its value, such as a physical constant, is listed in the
derivation's `fixed_constants` instead of its inputs:

```json
{"name": "E_K",
 "inputs": {"T": {"input_item": "T_body"}, "K_o": {"input_item": "K_o"},
            "K_i": {"input_item": "K_i"}},
 "outputs": {"E": "membrane/E_K"},
 "cellml": "derivations/nernst.cellml",
 "fixed_constants": ["R", "F"]}
```

`R` and `F` keep the values and units the CellML file gives them. Every other constant must still
be an input, so a forgotten input is still an error. Naming a fixed constant that is not a
constant of the model (a computed constant, or no variable at all), or that is also an input, is
an error. Each derivation's result records its fixed constants with their values and units
(`fixed_constants` of the result: `{"R": {"value": 8.314, "units": "J_per_K_per_mol"}}`), and
`cuflynx-derive` prints them:

```
E_K (cellml)
  fixed constant R = 8.314 J_per_K_per_mol (from the CellML file)
  fixed constant F = 96485.0 C_per_mol (from the CellML file)
  membrane/E_K: cached -0.0897, derived -0.0897 V
```

A fixed constant may say what set its value, as `{"name": "R", "origin": "measured",
"data_reference": "[CODATA][2018][gas_constant]"}` (`origin`: `measured`, `inherited` or
`set_by_hand`; `set_by_hand` needs a `reason`). `cuflynx-derive --write` records each derived
parameter's fixed constants, with value, units and origin, in its `fixed_constants` key, and a
calibration workflow's provenance diagram draws them in a "Fixed constants of derivations" box
beside the derived values. A constant given as a bare name has no origin and is drawn
`unclassified`.

A number written in the equation itself (`<cn>`) needs no listing, but a named fixed constant
shows its value and units where the derivation is reviewed.

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
Python function for those. An output that is not a finite number (a division by zero flow, say) is
an error.

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
names its derivation, and has origin `derived` (a leaf with `derivation` and another origin, or
none, is refused):

```json
"R_T0": {"value": 146654199.99999994, "units": "J_s_per_m6",
         "data_reference": "derived; derivation R_T0", "origin": "derived", "derivation": "R_T0"}
```

The provenance diagram of a calibration workflow marks such a value `derived`.

### When a derived parameter is calibrated

A derived value is only a calibration's start value. When a calibration fits the parameter, the
calibrated leaf has origin `calibrated` and no `derivation`: `write_calibrated_parameters` (a
workflow's `<name>_calibrated_parameters.json`) and generation with fit parameters
(`cuflynx-generate True`) both write it so. `cuflynx-derive` prints a calibrated parameter's
derived value but neither checks nor writes it, so `--write` can never overwrite a calibrated
value.

The derived value is not lost:

- the parameters file the calibration started from still has it, with its `derivation`: a
  workflow writes only the calibrated file;
- the derivations file still calculates it: `cuflynx-derive` prints it beside the calibrated value;
- each workflow step's generated parameters file (`generated.parameters_path` in its result)
  holds it as the value that step started from.

## `cuflynx-derive`

```
cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json
cuflynx-derive BVC_derivations.json --obs BVC_obs_data.json --write
```

The first evaluates every derivation and prints each parameter's cached and derived value. It exits
0 when every cached value matches within a relative tolerance of 1e-9 and 1 otherwise, so a test or
CI job can run it. A parameter that names a derivation that does not set it is reported too.
`--write` writes the derived values into the parameters file, and adds `derivation` and origin
`derived` to each output parameter; every other key and parameter is kept, and a calibrated
parameter (origin `calibrated`) is left as it is. It exits 1 when a parameter still names a
derivation that does not set it, which `--write` cannot fix: remove that `derivation` key. A file or
derivation that cannot be read or evaluated is an error message and exit 2.

`--obs` may be given several times; `--parameters` names the parameters file when it is not
`<name>_parameters.json` beside the derivations file. From Python:

```python
from libcuflynx.derivations import (check_derivations, evaluate_derivations, load_derivations,
                                    read_parameters)
doc = read_parameters('BVC_parameters.json')
results = evaluate_derivations(load_derivations('BVC_derivations.json'), doc, ['BVC_obs_data.json'])
print(check_derivations(results, doc))   # [] when the cache is current
```

Each result carries its inputs with their values, units, standard deviations and sources, and its
fixed constants with their values and units, for a review page or a table.

## Packing

`pack_workflow` (and `cuflynx-calibration-workflow --pack`) copies the derivations file beside each
parameters file it packs, with the CellML and Python files it reads at the same paths relative to
it, so they go into the OMEX archive CUFLynx makes from the pack. A CellML derivation model brings
the CellML files it imports. A derivation's files must be in the derivations file's directory or
below it. A module version from a library is packed whole, with its derivations.

A study's working dir does the same: when the parameters file is copied into
`inputs/system/<date>/`, its derivations file and the files it reads are copied beside it, and an
edit to any of them makes a new dated copy.
