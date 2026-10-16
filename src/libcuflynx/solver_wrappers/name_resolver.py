"""
Unified variable name resolver for all simulation helpers.

All helpers accept names in CellML canonical form: 'component/variable'
(with '_module' stripped from the component, e.g. 'Lotka_Volterra/x').

Two resolution modes:

1. **Index mode** (python / casadi helpers)
   Initialised with STATE_INFO / VARIABLE_INFO lists from a libCellML-generated
   Python module. `resolve(name)` returns `(kind, index)`.

2. **Key mode** (opencor / myokit helpers)
   Static `resolve_key(name, *keyed_dicts, separator)` searches runtime
   string-keyed dicts (e.g. OpenCOR data.states(), Myokit qname_to_var).
   Returns `(kind, matched_key)`.

Canonical alias patterns tried (in order):
  'Comp/Var'              exact match (canonical)
  'Comp_bare/Var'         strip _module suffix from component
  'parameters/Var_Comp'   CellML parameter convention  (aortic_root/C → parameters/C_aortic_root)
  'parameters/Var_bare'   same, _module stripped
  'parameters_Comp/Var'   global-prefix convention     (global/x → parameters_global/x)
  'Comp_module/Var'       add _module to component
  hoisted names           HOISTED_PARAMETER_CONVENTIONS (utilities/parameter_names.py):
    'parameters/Comp_Var'          component first (soma_SN/c_ER -> parameters/soma_SN_c_ER)
    'parameters/Var'               name kept as-is (soma_SN/g_Na -> parameters/g_Na)
    'parameters_global/Var'        global constant (soma_SN/R -> parameters_global/R)
    'instance_parameters/Comp_Var' the same three in the components PhLynx uses from
    'instance_parameters/Var'      v0.3.0 on
    'global_parameters/Var'
  'Var'                   bare variable name (unambiguous only in index mode)

The hoisted names exist because a CellML model does not have to name its hoisted
parameters the way circulatory_autogen names them. PhLynx writes 'soma_SN_c_ER'
where CA writes 'c_ER_soma_SN', and drops the prefix entirely for a name that is
already unique in the model -- and Myokit's importer *merges* connected variables,
so after import only the defining copy in the parameters component survives and
'soma_SN.c_ER' is not a qname any more. Without these aliases a study built in PhLynx
resolves none of its own parameters (CUFLynx #300, issue #558). The same list is read
by model_qname_candidates, the variable mapping, methods and the calibrated-value writer.

For key mode with separator='.', every '/'-candidate is also tried with '/' → '.'
(so Myokit qnames like 'Lotka_Volterra_module.x' are matched automatically).
"""

import warnings

from libcuflynx.utilities.parameter_names import LEVEL_SEPARATOR, hoisted_names


class VariableNameResolver:

    # ------------------------------------------------------------------
    # Index mode (python / casadi helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def canonical(info: dict) -> str:
        """Return 'component/variable' from a STATE_INFO/VARIABLE_INFO dict."""
        comp = info.get("component", "")
        if comp.endswith("_module"):
            comp = comp[:-7]
        return f"{comp}/{info['name']}" if comp else info["name"]

    def __init__(self, state_info: list, variable_info: list):
        self._state_info = state_info
        self._variable_info = variable_info

        # canonical_name → ('state'|'var', index)
        self._map: dict = {}
        # short var name → canonical  (None = ambiguous)
        self._short: dict = {}

        for idx, info in enumerate(state_info):
            cname = self.canonical(info)
            self._map[cname] = ("state", idx)
            self._add_short(info["name"], cname)

        for idx, info in enumerate(variable_info):
            cname = self.canonical(info)
            self._map[cname] = ("var", idx)
            self._add_short(info["name"], cname)

    def _add_short(self, short: str, canonical: str):
        if short not in self._short:
            self._short[short] = canonical
        else:
            self._short[short] = None  # ambiguous

    def resolve(self, name: str):
        """Return (kind, index) or (None, None)."""
        name = str(name).strip()

        if name in self._map:
            return self._map[name]

        # Bare name: try unambiguous short lookup
        if "/" not in name and "." not in name:
            canon = self._short.get(name)
            if canon:
                return self._map[canon]
            return (None, None)

        for alias in self._aliases(name, "/"):
            if alias in self._map:
                return self._map[alias]
            if "/" not in alias:
                canon = self._short.get(alias)
                if canon:
                    return self._map[canon]

        legacy = self._legacy_match(name, [name] + self._aliases(name, "/"), [self._map])
        return self._map[legacy] if legacy else (None, None)

    def all_state_names(self) -> list:
        return [self.canonical(info) for info in self._state_info]

    def all_variable_names(self) -> list:
        return [self.canonical(info) for info in self._variable_info]

    def all_names(self) -> list:
        return self.all_state_names() + self.all_variable_names()

    # ------------------------------------------------------------------
    # Key mode (opencor / myokit helpers) — static interface
    # ------------------------------------------------------------------

    @classmethod
    def resolve_key(cls, name: str, keyed_dicts, separator: str = "/"):
        """
        Resolve `name` against one or more string-keyed dicts.

        Parameters
        ----------
        name : str
            Input name, e.g. 'Lotka_Volterra/x' or 'aortic_root/C'.
        keyed_dicts : iterable of (kind_label, dict-like)
            E.g. [('state', data.states()), ('const', data.constants())].
            Each dict-like is tested with ``key in d``.
        separator : str
            Key separator used internally by the backend.
            '/' for OpenCOR CellML names; '.' for Myokit qnames.

        Returns
        -------
        (kind_label, matched_key) or (None, None)
        """
        name = str(name).strip()
        keyed_dicts = list(keyed_dicts)

        def _check(candidate):
            for kind, d in keyed_dicts:
                if candidate in d:
                    return (kind, candidate)
            return (None, None)

        # Try the name as-is first
        kind, key = _check(name)
        if kind:
            return (kind, key)

        # Generate slash-based aliases, then optionally convert separator
        for slash_alias in cls._aliases(name, "/"):
            # Try with requested separator
            alias = slash_alias.replace("/", separator) if separator != "/" else slash_alias
            kind, key = _check(alias)
            if kind:
                return (kind, key)

        candidates = [name] + [a.replace("/", separator) for a in cls._aliases(name, "/")]
        legacy = cls._legacy_match(name, candidates, [d for _, d in keyed_dicts])
        return _check(legacy) if legacy else (None, None)

    @staticmethod
    def _legacy_match(name, candidates, keyed):
        """The one key whose nested levels, joined with "__", the candidates write with a single
        "_" as before #582 (``heart_lv/E`` for ``heart__lv``), with a FutureWarning; None when
        there is none. ValueError when there are several."""
        wanted = set(candidates)
        matches = sorted({key for d in keyed for key in d
                          if LEVEL_SEPARATOR in str(key)
                          and str(key).replace(LEVEL_SEPARATOR, "_") in wanted})
        if len(matches) > 1:
            raise ValueError(f'"{name}" could be any of {matches}; nested levels are joined '
                             f'with "{LEVEL_SEPARATOR}" now, so name one of those.')
        if matches:
            warnings.warn(f'"{name}" joins nested levels with "_"; it is read as '
                          f'"{matches[0]}" (levels are joined with "{LEVEL_SEPARATOR}").',
                          FutureWarning, stacklevel=3)
            return matches[0]
        return None

    # ------------------------------------------------------------------
    # Shared alias generation
    # ------------------------------------------------------------------

    @staticmethod
    def _aliases(name: str, sep: str) -> list:
        """
        Generate alias candidates for *name* using *sep* as the separator.
        Input may use '/' or '.'; output always uses *sep*.
        """
        name = str(name).strip()

        # Normalise input to use sep so bare-name fallback works for both / and .
        input_sep = "/" if "/" in name else ("." if "." in name else None)
        if input_sep and input_sep != sep:
            name = name.replace(input_sep, sep)

        if sep not in name:
            return []  # nothing to decompose

        comp, var = name.split(sep, 1)
        comp_bare = comp[:-7] if comp.endswith("_module") else comp
        comp_mod  = comp_bare + "_module"

        return [
            f"{comp_bare}{sep}{var}",       # strip _module
            f"parameters{sep}{var}_{comp}", # param convention
            f"parameters{sep}{var}_{comp_bare}",
            f"parameters_{comp}{sep}{var}", # global-prefix convention
            f"parameters_{comp_bare}{sep}{var}",
            f"{comp_mod}{sep}{var}",        # add _module
            # Hoisted-parameter conventions other generators use. Tried after the
            # CA-shaped ones above so a model that follows both is read the CA way,
            # and before the bare name so 'soma_SN/g_Na' cannot be answered by an
            # unrelated 'g_Na' in some other component.
            *hoisted_names(comp, var, sep),
            var,                            # bare variable name
        ]
