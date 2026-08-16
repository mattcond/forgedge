"""``UNSET`` — the sentinel that separates "the user chose this" from "default".

FORGE's configuration is spread across seventeen dataclasses and 158 fields, and
the same quantity is materialised several times over — the same arrival rate
lives in four fields, the same bar duration in five, the same column name in
four (see ``docs/analysis/pipeline_parameter_coherence.md``).  Keeping those
copies consistent requires answering one question for every field: *did the
caller choose this value, or is it the class default?*

A plain default cannot answer it.  ``AlphaConfig.timestamp_col == "open_dt"`` is
identical whether the user wrote it or inherited it, so a resolver has no way to
know which copies it may fill in and which it must leave alone.  The only place
in the codebase that needed the distinction so far — the hourly-horizon-grid
warning in ``forge()`` — worked around it by comparing against
``AlphaConfig.__dataclass_fields__["horizon_grid"].default``, a technique that
does not generalise past one field.

``UNSET`` is that answer.  A field left at ``UNSET`` means *"the resolver
decides"*; any other value, including one equal to the historical default, means
*"this was chosen"* and is never overwritten.

Deliberate design points
------------------------
* **Falsy.** ``bool(UNSET)`` is ``False``, so ``if cfg.field:`` reads as
  "absent", which is what an unresolved field is.
* **Not arithmetic.** No comparison or arithmetic operators are defined, so
  ``UNSET < 3`` or ``UNSET * 2`` raise ``TypeError`` rather than silently
  producing a number.  An unresolved value that reaches a computation fails
  loudly at the point of use.
* **Singleton, copy-stable.** ``copy``/``deepcopy``/``pickle`` all return the
  same object, so ``is UNSET`` stays a valid test after a config is copied —
  which ``resolve()`` does on every call.

Typing note: fields annotated ``float`` or ``str`` may hold ``UNSET`` before
resolution.  This is intentional and untyped; the alternative (``Union[float,
_Unset]`` on 158 fields) would make every signature unreadable for a state that
exists only between construction and ``resolve()``.
"""
from __future__ import annotations

from typing import Any

__all__ = ["UNSET", "is_set", "coalesce"]


class _Unset:
    """Singleton type of :data:`UNSET`.  Never instantiate directly."""

    _instance = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False

    # Copy-stable: `is UNSET` must survive copy/deepcopy/pickle, because
    # resolve() works on copies of the caller's configs.
    def __copy__(self) -> "_Unset":
        return self

    def __deepcopy__(self, memo: dict) -> "_Unset":
        return self

    def __reduce__(self):
        return (_Unset, ())


#: The "resolver decides" sentinel.  See the module docstring.
UNSET = _Unset()


def is_set(value: Any) -> bool:
    """``True`` when ``value`` is an actual choice rather than :data:`UNSET`."""
    return value is not UNSET


def coalesce(*values: Any, default: Any = None) -> Any:
    """Return the first argument that is not :data:`UNSET`, else ``default``.

    Handy at module boundaries where a field may legitimately still be unset
    (standalone use with no context supplied).
    """
    for value in values:
        if value is not UNSET:
            return value
    return default
