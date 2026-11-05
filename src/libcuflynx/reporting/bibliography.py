'''
BibTeX for the methods write-up: reading module ``*_references.bib`` files, turning the keys
in a parameter's ``data_reference`` into ``\\cite{}``, and writing one ``.bib`` for the document.

A ``data_reference`` is free text that names its sources by BibTeX key, conventionally first
(``martin2023modelling; Methods (p. 16): ...``) but anywhere in the text. A word is cited only
when it is a key of the bibliography given, so ``definitional`` or ``user_defined`` stay text.
'''
import os
import re

from libcuflynx.reporting.latex_names import escape

_ENTRY = re.compile(r'@\s*(\w+)\s*\{\s*([^,\s]+)\s*,', re.S)
_WORD = re.compile(r'[A-Za-z][\w:.\-]*\w|[A-Za-z]')


def read_bib(paths):
    '''``{key: entry text}`` from the ``.bib`` files ``paths``; the first file defining a key
    wins. ``@comment``, ``@string`` and ``@preamble`` blocks are skipped.'''
    entries = {}
    for path in paths:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        for match in _ENTRY.finditer(text):
            if match.group(1).lower() in ('comment', 'string', 'preamble'):
                continue
            depth, end = 0, None
            for i in range(text.index('{', match.start()), len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is not None:
                entries.setdefault(match.group(2), text[match.start():end])
    return entries


def cite(text, keys):
    '''``text`` as LaTeX, with every word that is one of ``keys`` written ``\\cite{key}``.'''
    out, last = [], 0
    for match in _WORD.finditer(text or ''):
        if match.group(0) in keys:
            out.append(escape(text[last:match.start()]))
            out.append(r'\cite{' + match.group(0) + '}')
            last = match.end()
    out.append(escape((text or '')[last:]))
    return ''.join(out)


def cited_keys(text, keys):
    return [m.group(0) for m in _WORD.finditer(text or '') if m.group(0) in keys]


def write_bib(entries, path):
    '''Write ``entries`` (``{key: entry text}``) as one ``.bib`` file.'''
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(entries[k] for k in sorted(entries)) + '\n')
    return path


def module_bib_files(config_paths):
    '''The ``*_references*.bib`` files beside each module config in ``config_paths`` (a
    module library version keeps its references there), in order, without repeats.'''
    found = []
    for config in config_paths:
        directory = os.path.dirname(os.path.abspath(config))
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith('.bib') and '_references' in name and path not in found:
                found.append(path)
    return found
