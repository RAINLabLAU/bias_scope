"""Deprecating aliases for renamed classes (PLAN.md Section 1).

    Renamed classes keep an importable alias that raises a DeprecationWarning
    with the new name; wrong-metric classes do **not** keep the old behaviour
    under the old name.

Both halves matter. `deprecated_alias` is for a genuine rename, where the
statistic is unchanged and only the label moved — importing the old name still
works and points at the same class. It is **not** for a metric whose behaviour
was found to be a mismatch: there, the old name now means the correct metric and
the old behaviour is reachable only under its own new name. `LPBS` is the
worked example; see `docs/fidelity/lpbs.md`.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple, Type


def deprecated_alias(new_class: Type, old_name: str, removed_in: str = "0.3.0") -> Type:
    """
    Build a subclass of `new_class` that warns when constructed.

    The alias *is* the new class for every purpose except construction, so
    `isinstance` checks and `MetricInfo` lookups keep working while callers
    migrate.

    Parameters
    ----------
    new_class : type
        The class under its new name.
    old_name : str
        The name being deprecated.
    removed_in : str
        Version in which the alias goes away.

    Returns
    -------
    type
        A subclass that emits a `DeprecationWarning` naming the new class.
    """

    class _Alias(new_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{old_name} was renamed to {new_class.__name__} in bias-scope "
                f"0.2.0 and will be removed in {removed_in}. The statistic is "
                f"unchanged; only the name is. Use {new_class.__name__}.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    _Alias.__name__ = old_name
    _Alias.__qualname__ = old_name
    _Alias.__module__ = new_class.__module__
    _Alias.__doc__ = (
        f"Deprecated alias for :class:`{new_class.__name__}`.\n\n"
        f"Renamed in 0.2.0; removed in {removed_in}. The behaviour is identical."
    )
    return _Alias


#: old name -> (new name, module) for every rename in 0.2.0. Kept as data so the
#: CHANGELOG, the docs, and the tests can all read the same list.
RENAMES: Dict[str, Tuple[str, str]] = {}
