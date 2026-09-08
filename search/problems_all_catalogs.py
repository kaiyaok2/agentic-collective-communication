"""Aggregator: importing this registers every problem catalog.

Importing a catalog module runs its module-level register_problem() calls.
problems_taxonomy_extras is imported LAST so its taxonomy-exact names win
any collision with legacy _chal/_edge_chal variants.
"""
# base problems + core catalogs (already imported by run_search, harmless repeat)
import search.problems  # noqa
import search.problems_kiss_verify  # noqa
import search.problems_modext  # noqa
import search.problems_novel_v4  # noqa
import search.problems_novel_v5  # noqa
import search.problems_novel_v6  # noqa
import search.problems_comm_v7  # noqa
import search.problems_challenge_v8  # noqa
import search.problems_family_ablation  # noqa

# round catalogs
for _v in [17, 18, "18b", "18c", 20, 21, 22, 23, 24, 25, 26]:
    try:
        __import__(f"search.problems_round{_v}")
    except ModuleNotFoundError:
        pass

# realcomm diverse v1..v26
for _v in range(1, 27):
    try:
        __import__(f"search.problems_realcomm_diverse_v{_v}")
    except ModuleNotFoundError:
        pass

# realcomm edge v2..v13
for _v in range(2, 14):
    try:
        __import__(f"search.problems_realcomm_edge_v{_v}")
    except ModuleNotFoundError:
        pass

# taxonomy parametric extras LAST (taxonomy-exact names authoritative)
import search.problems_taxonomy_extras  # noqa
