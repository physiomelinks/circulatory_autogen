"""Builders that exist so code *outside* this repository can test against a real CA run.

Everything here is library code by necessity rather than by ambition: the wheel
ships no ``tests/``, so anything CUFLynx (or any other consumer) has to import in
order to check itself against CA cannot live in CA's test suite. It is imported by
tests on both sides of that boundary and by nothing an ordinary run reaches.

Kept apart from :mod:`libcuflynx.checks`, which validates a *user's* model during
generation -- a different job with a different audience.
"""
