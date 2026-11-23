"""Exceptions libcuflynx raises on input it cannot use.

Library code raises these rather than calling ``exit()``: ``SystemExit`` derives from
``BaseException``, so it passes through the ``except Exception`` of a caller that runs
libcuflynx in-process (a notebook, the CUFLynx server, a calibration workflow step) and
takes the message with it. The console entry points (``libcuflynx.scripts``) turn an
exception into a printed traceback and a non-zero exit status.
"""


class CAInputError(ValueError):
    """A configuration, model array, parameters file or obs_data that libcuflynx refuses.

    The message says what is wrong and where, so a caller can show it as it is. Catch this
    to handle bad user input apart from other failures; it is a ``ValueError``, so code
    that catches those catches it too.
    """
