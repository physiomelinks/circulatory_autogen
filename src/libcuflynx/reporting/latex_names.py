'''
The default rule that turns a model's variable names into LaTeX symbols.

The rule is deliberately simple and predictable, because its output is a starting point
for the user to edit (``<prefix>_variable_mapping.csv``), not a final answer:

* a name is split on ``_``: the first part is the symbol, the rest its subscripts
  (``g_leak_Na`` -> ``g_{\\mathrm{leak},\\mathrm{Na}}``);
* a part that is a Greek letter's name becomes that letter (``rho_M`` -> ``\\rho_{M}``,
  ``Delta`` -> ``\\Delta``);
* a one-character symbol or subscript stays italic, a longer one is set upright
  (``Nai`` -> ``\\mathrm{Nai}``); a trailing number on a one-letter symbol becomes a
  subscript (``V0`` -> ``V_{0}``);
* the module a variable belongs to is appended as one more subscript, so the same variable
  in two modules gets two symbols (``V`` of ``membrane`` -> ``V_{\\mathrm{membrane}}``).
'''

import re

GREEK = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota',
         'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi',
         'chi', 'psi', 'omega')
# capitals that LaTeX has a command for (the rest look like Latin capitals)
GREEK_UPPER = ('Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi',
               'Psi', 'Omega')

_SPECIAL = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '_': r'\_', '%': r'\%',
            '$': r'\$', '#': r'\#', '&': r'\&', '^': r'\^{}', '~': r'\~{}'}


def escape(text):
    '''``text`` safe to set in LaTeX text mode (``\\texttt{}``, ``\\mathrm{}``).'''
    return ''.join(_SPECIAL.get(ch, ch) for ch in str(text))


def _greek(token):
    if token in GREEK or token in GREEK_UPPER:
        return '\\' + token
    if token.lower() in GREEK and token[0].isupper() and token not in GREEK_UPPER:
        return '\\' + token.lower()      # 'Alpha' has no command: use the lower case
    return None


def _token(token):
    return _greek(token) or (token if len(token) == 1 else r'\mathrm{' + escape(token) + '}')


def symbol(name, extra_subscripts=()):
    '''The default LaTeX for a variable called ``name``, with ``extra_subscripts`` (already
    LaTeX) appended to its subscripts.'''
    tokens = [t for t in str(name).split('_') if t]
    if not tokens:
        return r'\mathrm{' + escape(name) + '}'
    head, rest = tokens[0], tokens[1:]
    subs = []
    match = re.fullmatch(r'([A-Za-z])(\d+)', head)
    if match:
        head_tex = match.group(1)
        subs.append(match.group(2))
    else:
        head_tex = _greek(head) or (head if len(head) == 1 else r'\mathrm{' + escape(head) + '}')
    subs += [_token(t) for t in rest] + [s for s in extra_subscripts if s]
    return head_tex + ('_{' + ','.join(subs) + '}' if subs else '')


def module_subscript(module_name):
    '''A module (vessel) name as a subscript: upright, underscores kept.'''
    return r'\mathrm{' + escape(module_name) + '}'
