"""forgedge.playground — analysis abstractions over ``ForgeResult`` (R).

Read-only helpers that manipulate the *results* of one or more ``forge()``
runs — never re-run the pipeline. Each function returns a long-format
``pandas.DataFrame`` (one row per elementary observation) so callers can
compose their own ``groupby``/aggregation instead of the function deciding
it for them. See ``docs/analysis/`` issue #237 for the running checklist of
use cases this module implements.

Usage::

    from forgedge.playground import *
"""

from .m2 import discard_reasons_by_grade, undetermined_direction_by_family

__all__ = [
    "discard_reasons_by_grade",
    "undetermined_direction_by_family",
]
