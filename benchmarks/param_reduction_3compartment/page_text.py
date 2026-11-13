"""Text of the result page that comes from a run's files rather than from the script."""


def pca_options_text(report):
    """The options and informative-mode count of a ``param_reduction.json`` (the reduce_pca step's
    report), as a sentence for the provenance diagram."""
    options = report['reduction']
    shown = ', '.join(f'{name.replace("_", " ")} {options[name]}'
                      for name in ('min_mode_std', 'variance_fraction', 'min_index')
                      if name in options)
    return f'{shown}. Found {report["n_informative_modes"]} informative modes'
