"""Aggregator: importing this registers every problem catalog.

Importing a catalog module runs its module-level register_problem() calls.
We auto-discover every ``search/problems_*.py`` module present in this repo
and import each one tolerantly (missing/legacy modules are skipped, not
fatal). ``problems_taxonomy_extras`` is imported LAST so its taxonomy-exact
names win any collision with legacy _chal/_edge_chal variants.
"""
import importlib
import os
import pkgutil

import search.problems  # noqa  base problems (defines register_problem/PROBLEMS)

_PKG_DIR = os.path.dirname(__file__)
_LAST = "search.problems_taxonomy_extras"
_SELF = "search.problems_all_catalogs"

# Discover all sibling problems_*.py modules deterministically (sorted), so
# import order is stable across runs and across the 7 ranks.
_mods = sorted(
    f"search.{name}"
    for _, name, _ in pkgutil.iter_modules([_PKG_DIR])
    if name.startswith("problems_") and f"search.{name}" not in (_LAST, _SELF)
)

for _m in _mods:
    try:
        importlib.import_module(_m)
    except ModuleNotFoundError:
        # optional dependency of a legacy catalog not present in this repo
        pass

# taxonomy parametric extras LAST (taxonomy-exact names authoritative)
importlib.import_module(_LAST)
