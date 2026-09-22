"""Round 59b -- FAMILY-1 top-up. The r59 batch showed that overlay folds the
lighter payloads (part 256/512), dragging the bootstrap CI_lo to exactly 1.0 even
when the median ratio is ~1.9; only the HEAVY payload (part=1024, depth 8) recipe
r59_su_a4_d8_p1024 survived the strict CI_lo>1.0 gate. These top-ups replicate that
winning recipe (deep d8 + heavy payload) with DISTINCT (a-pattern, payload) tuples so
the reference outputs differ from every existing family-1 problem.
"""
import torch  # noqa: F401
from .problems_diverge_r59 import _mk


def register_all():
    _mk("r59b_su_a4_d8_p2048", 2048, 8, "1.0 + 0.6*(r % 4)",
        lambda r: 1.0 + 0.6 * (r % 4), "1.0+0.6*(r%4)")
    _mk("r59b_su_a5_d8_p1024", 1024, 8, "1.0 + 0.4*(r % 5)",
        lambda r: 1.0 + 0.4 * (r % 5), "1.0+0.4*(r%5)")
    _mk("r59b_su_a6_d8_p1024", 1024, 8, "1.0 + 0.35*(r % 6)",
        lambda r: 1.0 + 0.35 * (r % 6), "1.0+0.35*(r%6)")


register_all()
