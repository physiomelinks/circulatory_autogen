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
  'Var'                   bare variable name (unambiguous only in index mode)

For key mode with separator='.', every '/'-candidate is also tried with '/' → '.'
(so Myokit qnames like 'Lotka_Volterra_module.x' are matched automatically).
"""


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

        return (None, None)

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

        return (None, None)

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
            var,                            # bare variable name
        ]
