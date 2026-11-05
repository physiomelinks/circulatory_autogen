"""Reading saved Sobol indices and ranking parameters by them.

The indices CSV (``all_outputs_n<N>_Sobol_indices.csv``) has a ``Parameter`` column and an
``S1_<output>`` and ``ST_<output>`` column per output.
"""

import pandas as pd

CRITERIA = ('max', 'mean')


def read_sobol_indices(path):
    """``{'S1': {output: {parameter: value}}, 'ST': {...}}`` from a saved indices CSV."""
    df = pd.read_csv(path).set_index('Parameter')
    results = {'S1': {}, 'ST': {}}
    for col in df.columns:
        for index_type in results:
            if col.startswith(f'{index_type}_'):
                results[index_type][col[len(index_type) + 1:]] = df[col].to_dict()
    return results


def rank_sobol_parameters(indices, top_n=5, index_type='ST', criterion='max', threshold=None):
    """The ``top_n`` parameters with the highest score, highest first, as
    ``[(parameter, score)]``.

    A parameter's score is the ``criterion`` (``'max'``, ``'mean'`` or a callable) of its
    ``index_type`` index over the outputs. With ``threshold``, parameters scoring below it
    are dropped.
    """
    data = indices[index_type.upper()]
    per_parameter = {}
    for values in data.values():
        for name, value in values.items():
            per_parameter.setdefault(name, []).append(value)
    if criterion == 'max':
        score = max
    elif criterion == 'mean':
        score = lambda x: sum(x) / len(x)  # noqa: E731
    elif callable(criterion):
        score = criterion
    else:
        raise ValueError(f"criterion is one of {CRITERIA} or a callable, not {criterion!r}")
    scores = {name: score(values) for name, values in per_parameter.items()}
    if threshold is not None:
        scores = {name: s for name, s in scores.items() if s >= threshold}
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
