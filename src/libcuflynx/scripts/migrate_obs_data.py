"""Rewrite an obs_data.json into the vocabulary introduced by #466.

    cuflynx-migrate-obs-data resources/            # every obs_data file under a directory
    cuflynx-migrate-obs-data my_obs_data.json --dry-run

``variable`` and ``name_for_plotting`` each named two different things, and the split is not
purely mechanical, which is why this exists rather than a note in the changelog:

* ``variable`` becomes ``data_item_name``, and on a ``prediction_item`` it also seeds
  ``operands`` -- there was no ``operands`` key there at all, the model variable *was*
  ``variable``.
* ``name_for_plotting`` becomes ``trace_name_for_plotting``.
* ``data_item_name`` has to be unique across ``data_items`` and ``prediction_items``, because it
  is what an ``operation_kwargs`` reference to another item resolves against. Files that named
  one variable once per feature -- the mean and the max of a trace, or one variable measured
  across experiments -- now collide, so a colliding name is qualified by whatever actually
  distinguishes the items: the operation first, then the experiment and sub-experiment.

Edits are textual, so each file keeps its own hand formatting and the diff shows only the keys
and names that moved. An ``operation_kwargs`` value that referenced a renamed item is followed
through, so a difference-of-two-observables item keeps pointing at the right pair.
"""
import argparse
import collections
import json
import os
import re
import sys

#: ``"data_item_name": "..."`` wherever it sits on the line, so a file that puts the key inline
#: after an opening brace is treated the same as one that gives it a line of its own.
_NAME = re.compile(r'"data_item_name"(\s*):(\s*)"((?:[^"\\]|\\.)*)"')
_LEGACY_KEYS = (('variable', 'data_item_name'),
                ('name_for_plotting', 'trace_name_for_plotting'))


def _operation_of(item):
    """What reduces this item's trace to its value, however the file spells it.

    Falls back to the deprecated ``obs_type``, which older files use in place of ``operation``.
    It is usually the only thing distinguishing several items on one variable, so without it the
    disambiguated names come out as ``x``, ``x 2``, ``x 3`` rather than ``max x``, ``min x``.
    """
    for key in ('operation', 'obs_type'):
        op = item.get(key)
        if op not in (None, '', 'Null', 'None', 'null', 'none', 'nan',
                      'series', 'frequency', 'constant'):
            return str(op)
    return None


def _items_in_file_order(raw, doc):
    """Every entry paired with whether it is a prediction, in the order they appear in the text.

    The order matters because the rewrite walks the file's ``data_item_name`` occurrences in
    sequence, and a document may put ``prediction_items`` before ``data_items``.
    """
    if isinstance(doc, list):
        return [(item, False) for item in doc if isinstance(item, dict)]
    data = [i for i in (doc.get('data_items') or doc.get('data_item') or []) if isinstance(i, dict)]
    preds = [i for i in (doc.get('prediction_items') or []) if isinstance(i, dict)]
    if not preds:
        return [(i, False) for i in data]
    at_data = raw.find('"data_items"')
    if at_data < 0:
        at_data = raw.find('"data_item"')
    at_pred = raw.find('"prediction_items"')
    if 0 <= at_pred < at_data:
        return [(i, True) for i in preds] + [(i, False) for i in data]
    return [(i, False) for i in data] + [(i, True) for i in preds]


def _unique_name(base, item, is_prediction, taken):
    """A name for ``item`` that nothing else answers to, changing as little as possible."""
    op, exp = _operation_of(item), item.get('experiment_idx', 0)
    sub = item.get('subexperiment_idx', 0)
    candidates = []
    if op:
        candidates += [f'{op} {base}', f'{op} {base} exp{exp}', f'{op} {base} exp{exp} sub{sub}']
    candidates += [f'{base} exp{exp}', f'{base} exp{exp} sub{sub}']
    if is_prediction:
        candidates = [f'{base} prediction'] + [f'{c} prediction' for c in candidates]
    for candidate in candidates:
        if candidate not in taken:
            return candidate
    suffix = 2
    while f'{base} {suffix}' in taken:
        suffix += 1
    return f'{base} {suffix}'


def plan_file(path):
    """What migrating ``path`` would change: ``(new_text, [(what, detail), ...])``.

    Returns ``(None, [])`` for a file that is not an obs_data document, and ``(text, [])`` for
    one that is already migrated.
    """
    with open(path, encoding='utf-8-sig') as fh:
        raw = fh.read()
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, []
    probe = doc if isinstance(doc, list) else (doc.get('data_items') or doc.get('data_item'))
    if not (isinstance(probe, list) and probe and isinstance(probe[0], dict)):
        return None, []

    changes = []
    for legacy, current in _LEGACY_KEYS:
        pattern = re.compile(rf'"{legacy}"(\s*):')
        if pattern.search(raw):
            raw = pattern.sub(rf'"{current}"\1:', raw)
            changes.append((f'{legacy} -> {current}', ''))
    doc = json.loads(raw)

    ordered = _items_in_file_order(raw, doc)
    counts = collections.Counter(str(i.get('data_item_name')) for i, _ in ordered)
    taken = {name for name, n in counts.items() if n == 1}
    renames, steps = {}, []
    for item, is_prediction in ordered:
        base = str(item.get('data_item_name'))
        new = base
        if counts[base] > 1:
            new = _unique_name(base, item, is_prediction, taken)
            taken.add(new)
            renames[base] = renames.get(base, []) + [new]
            changes.append(('renamed', f'{base!r} -> {new!r}'))
        needs_operands = bool(is_prediction and not item.get('operands'))
        if needs_operands:
            changes.append(('operands added', f'{base!r} -> ["{base}"]'))
        steps.append((base, new, needs_operands))

    index = [0]

    def replace(match):
        space_a, space_b, value = match.groups()
        base, new, add_operands = steps[index[0]]
        index[0] += 1
        out = f'"data_item_name"{space_a}:{space_b}"{new}"'
        if add_operands:
            out += f', "operands"{space_a}:{space_b}["{base}"]'
        return out

    raw = _NAME.sub(replace, raw)
    if index[0] != len(steps):
        raise RuntimeError(f'{path}: matched {index[0]} names, expected {len(steps)}')

    # An operation_kwargs value naming a renamed item has to follow it. Only safe where the old
    # name resolved to exactly one new one; a name that split into several is reported instead.
    for old, news in renames.items():
        if len(news) != 1:
            changes.append(('REVIEW', f'{old!r} split into {news}; check any operation_kwargs '
                                      f'that referenced it'))
            continue
        pattern = re.compile(rf'(:\s*)"{re.escape(old)}"')
        raw, n = pattern.subn(rf'\g<1>"{news[0]}"', raw)
        if n:
            changes.append(('reference updated', f'{old!r} -> {news[0]!r} ({n} place(s))'))

    json.loads(raw)
    return raw, changes


def _candidate_files(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if name.endswith('.json') and 'obs' in name:
                        yield os.path.join(root, name)
        else:
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='cuflynx-migrate-obs-data',
        description=__doc__.split('\n\n')[0],
        epilog=('Unlike the pipeline stages, this command takes paths on the command line and '
                'does not read user_inputs.yaml. Run it on the obs_data files a study owns; '
                'it rewrites them in place unless --dry-run is given.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', metavar='PATH',
                        help='obs_data JSON files, or directories to search for *obs*.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would change without writing anything')
    args = parser.parse_args(argv)

    changed = unchanged = skipped = failed = 0
    review = []
    for path in _candidate_files(args.paths):
        try:
            new_text, changes = plan_file(path)
        except Exception as exc:                                     # noqa: BLE001
            print(f'FAILED   {path}\n           {type(exc).__name__}: {exc}')
            failed += 1
            continue
        if new_text is None:
            skipped += 1
            continue
        if not changes:
            unchanged += 1
            continue
        changed += 1
        print(f'{"would migrate" if args.dry_run else "migrated"}  {path}')
        for what, detail in changes:
            print(f'    {what}{": " + detail if detail else ""}')
            if what == 'REVIEW':
                review.append((path, detail))
        if not args.dry_run:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(new_text)

    print(f'\n{changed} file(s) {"would be " if args.dry_run else ""}migrated, '
          f'{unchanged} already current, {skipped} not obs_data, {failed} failed')
    if review:
        print('\nNeeds a look -- one name became several, so a reference to it is ambiguous:')
        for path, detail in review:
            print(f'  {path}: {detail}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
