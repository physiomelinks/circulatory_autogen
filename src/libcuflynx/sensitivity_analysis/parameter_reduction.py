"""Reducing the parameters to calibrate from a Sobol sensitivity analysis.

:class:`SobolReduction` keeps the parameters with the highest Sobol index and writes those
entries of the analysed params_for_id, so a later calibration fits only them::

    reduction = SobolReduction(top_n=3, index_type='ST', criterion='max', threshold=0.01)
    kept = reduction.reduce('sa/all_outputs_n64_Sobol_indices.csv',
                            'candidates_params_for_id.json', 'reduced_params_for_id.json')
    # [('aortic_root/C', 0.81), ('heart/E_max', 0.42), ...], highest first

A calibration workflow's ``reduction`` step is one of these.
"""

from libcuflynx.sensitivity_analysis.sobol_ranking import (CRITERIA, rank_sobol_parameters,
                                                           read_sobol_indices)

INDEX_TYPES = ('ST', 'S1')


class SobolReduction:
    """Keep the ``top_n`` parameters by Sobol index.

    Args:
        top_n: How many parameters to keep, highest score first.
        index_type: ``'ST'`` (total, default) or ``'S1'`` (first order).
        criterion: A parameter's score over the outputs: ``'max'`` (default) or ``'mean'``.
        threshold: Parameters scoring below it are dropped, even within the top_n.
    """

    def __init__(self, top_n, index_type='ST', criterion='max', threshold=None):
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError(f'top_n is the number of parameters to keep, a positive '
                             f'integer, not {top_n!r}.')
        if str(index_type).upper() not in INDEX_TYPES:
            raise ValueError(f'index_type is one of {INDEX_TYPES}, not {index_type!r}.')
        if criterion not in CRITERIA:
            raise ValueError(f'criterion is one of {CRITERIA}, not {criterion!r}.')
        if threshold is not None and (isinstance(threshold, bool)
                                      or not isinstance(threshold, (int, float))
                                      or threshold < 0):
            raise ValueError(f'threshold is a number >= 0, not {threshold!r}.')
        self.top_n = top_n
        self.index_type = str(index_type).upper()
        self.criterion = criterion
        self.threshold = threshold

    @classmethod
    def from_options(cls, options):
        """From a workflow's ``reduction`` dict (``top_n``, ``index_type``, ``criterion``,
        ``threshold``)."""
        return cls(**dict(options))

    def as_dict(self):
        return {'top_n': self.top_n, 'index_type': self.index_type,
                'criterion': self.criterion, 'threshold': self.threshold}

    def rank(self, indices):
        """``[(parameter, score)]`` of the kept parameters, highest first. ``indices`` is a
        saved indices CSV or ``read_sobol_indices``' dict."""
        if isinstance(indices, str):
            indices = read_sobol_indices(indices)
        return rank_sobol_parameters(indices, self.top_n, self.index_type, self.criterion,
                                     self.threshold)

    @staticmethod
    def write_params_for_id(source_path, out_path, keep):
        """Write the entries of the params_for_id ``source_path`` whose parameter is in
        ``keep``, whole and in its order, to ``out_path``. Returns the number written."""
        from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser,
                                                         param_entry_labels)
        from libcuflynx.utilities import params_for_id_file as pfi
        doc = pfi.read_params_for_id(source_path, keep_text=True)
        entries = list(doc.get('params') or [])
        labels = param_entry_labels(ObsAndParamDataParser().get_param_id_info(source_path))
        if len(labels) != len(entries):
            raise ValueError(f'{source_path} has {len(entries)} entries but {len(labels)} '
                             f'parameters; a reduction keeps whole entries, one parameter each.')
        unknown = sorted(set(keep) - set(labels))
        if unknown:
            raise ValueError(f'{source_path} has no parameter {unknown}.')
        kept = [entry for entry, label in zip(entries, labels) if label in keep]
        pfi.write_params_for_id(out_path, dict(doc, params=kept))
        return len(kept)

    def reduce(self, indices, params_for_id_path, out_path):
        """Rank ``indices`` and write the kept entries of ``params_for_id_path`` to
        ``out_path``. Returns the ranking (see :meth:`rank`); a ValueError when nothing
        scores the threshold."""
        ranked = self.rank(indices)
        if not ranked:
            raise ValueError(f'no parameter scores {self.threshold} or more.')
        self.write_params_for_id(params_for_id_path, out_path, [name for name, _ in ranked])
        return ranked
