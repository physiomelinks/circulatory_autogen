import numpy as np
import math
import sys
import re
import pint

class Normalise_class:
    def __init__(self, param_mins, param_maxs, mod_first_variables=0, modVal = 1.0):
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.mod_first_variables=mod_first_variables
        self.modVal = modVal

    def normalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = (x - self.param_mins)/(self.param_maxs - self.param_mins)
        elif xDim == 2:
            y = (x - self.param_mins.reshape(-1, 1))/(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))
        elif xDim == 3:
            y = ((x.reshape(x.shape[0], -1) - self.param_mins.reshape(-1, 1)) /
                 (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))).reshape(x.shape[0], x.shape[1],
                                                                                            x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()

        return y

    def unnormalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = x * (self.param_maxs - self.param_mins) + self.param_mins
        elif xDim == 2:
            y = x * (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) + self.param_mins.reshape(-1, 1)
        elif xDim == 3:
            y = (x.reshape(x.shape[0], -1)*(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) +
                 self.param_mins.reshape(-1, 1)).reshape(x.shape[0], x.shape[1], x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()
        return y


def obj_to_string(obj, extra='    '):
    return str(obj.__class__) + '\n' + '\n'.join(
        (extra + (str(item) + ' = ' +
                  (obj_to_string(obj.__dict__[item], extra + '    ') if hasattr(obj.__dict__[item], '__dict__') else str(
                      obj.__dict__[item])))
         for item in sorted(obj.__dict__)))

def bin_resample(data, freq_1, freq_ds):

    new_len = len(freq_ds)
    new_data = np.zeros((new_len))
    new_count = 0 
    this_count = 0 
    addup = 0 
    for II in range(0, len(freq_1)):
        
        dist_behind = np.abs(freq_1[II] - freq_ds[new_count])
        dist_infront = np.abs(freq_1[II] - freq_ds[new_count+1])
        if dist_behind < dist_infront:
            addup += data[II]
            this_count += 1
        else:
            if new_count == 0:
                # overwrite with 0th entry of data
                # this ignores some data points directly after 0 frequency
                new_data[0] = data[0]
            else:
                new_data[new_count] = addup / this_count
            addup = data[II]
            this_count = 1 
            new_count += 1

        if new_count == len(freq_ds) - 1:
            # add all remaining data points to this new datapoint and average
            new_data[new_count] = np.sum(data[II+1:]) / len(data[II+1:])
            break

    return new_data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

class UnitConverter:
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        self.abbreviations = {
            'cm': 'centimeter',
            'mm': 'millimeter',
            'm': 'meter',
            'g': 'gram',
            'kg': 'kilogram',
            's': 'second',
            'ms': 'millisecond',
            'mol': 'mole',
            'L': 'liter',
            'J': 'joule',
            'N': 'newton',
            'A': 'ampere',
            'V': 'volt',
            'Hz': 'hertz',
            'W': 'watt',
            'dim': 'dimensionless'
        }
        self.compound_aliases = {
            'Js': 'joule*second',
            'Vs': 'volt*second',
            'As': 'ampere*second',
            'Ns': 'newton*second',
            'milliJs': 'millijoule*second',
        }

    def expand_compound_unit(self, token):
        # First, check for compound alias with exponent, e.g., Js2
        match = re.match(r'([A-Za-z]+)(\d+)$', token)
        if match and match.group(1) in self.compound_aliases:
            base = match.group(1)
            exponent = match.group(2)
            # Expand the compound alias, then apply exponent to the last unit
            expanded = self.compound_aliases[base]
            # Split expanded into its units (e.g., 'joule*second')
            units = expanded.split('*')
            # Apply exponent to the last unit
            units[-1] = f"{units[-1]}**{exponent}"
            return '*'.join(units)
        # If it's a plain compound alias
        if token in self.compound_aliases:
            return self.compound_aliases[token]
        # Otherwise, expand as before
        units = re.findall(r'[a-zA-Z]+(?:[a-zA-Z])?(?:\d*)', token)
        expanded_parts = []
        for u in units:
            match = re.match(r'([a-zA-Z]+)(\d*)$', u)
            if not match:
                raise ValueError(f"Could not parse unit token: {u}")
            base, exponent = match.groups()
            base_expanded = self.abbreviations.get(base, base)
            if exponent:
                expanded_parts.append(f"{base_expanded}**{exponent}")
            else:
                expanded_parts.append(base_expanded)
        return "*".join(expanded_parts)

    def parse_unit_string(self, unit_str):
        parts = unit_str.split('_per_')
        if len(parts) == 2:
            numerator = self.expand_compound_unit(parts[0])
            denominator = self.expand_compound_unit(parts[1])
            return f"{numerator} / {denominator}"
        else:
            return self.expand_compound_unit(unit_str)

    def get_scale_factor(self, from_unit_str, to_unit_str):
        try:
            from_unit = self.ureg(self.parse_unit_string(from_unit_str))
            to_unit = self.ureg(self.parse_unit_string(to_unit_str))
            factor = (1 * from_unit).to(to_unit).magnitude
            return factor
        except pint.DimensionalityError as e:
            raise ValueError(f"Incompatible units: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse or convert units: {e}")







