"""Round 68b -- FAMILY-7 TOP-UP (rank-indexed bidiagonal inter-shard coupling), toward 10.

Family-7 stands at 3/10 (b02_d8_p2048, b03_d8_p2048, b03_d8_p1024) -- the first
rank-DEPENDENT off-diagonal family, which confirms reliably (like fam-1/2) with strong
ratios. This battery adds fresh rank-indexed b-patterns at the proven d8 depth and the
p2048/p1024 payloads where the confirmations clustered. Reuses the EXACT r68 mechanism +
telescoping via import (out[shard r] = s[shard r] + b[r]*s[shard r+1], unit-upper-bidiagonal
M invertible by local back-substitution); every b-pattern keeps |b[r]|<1 for a
well-conditioned inverse, and each distinct (b-pattern, payload) triple -> distinct reference
md5 from every registered problem (prescreen guard). Depth/payload is NOT distinctness on its
own; the rank-indexed coupling coefficient is.
"""
from .problems_diverge_r68 import _mk  # noqa: reuse bidiagonal mechanism + telescoping


def register_all():
    _mk("r68b_bidi_b015m3_d8_p2048", 2048, 8, "0.15 + 0.1*(r % 3)",
        lambda r: 0.15 + 0.1 * (r % 3), "0.15+0.1*(r%3)")
    _mk("r68b_bidi_b035m4_d8_p2048", 2048, 8, "0.35 + 0.1*(r % 4)",
        lambda r: 0.35 + 0.1 * (r % 4), "0.35+0.1*(r%4)")
    _mk("r68b_bidi_b02m4_d8_p2048", 2048, 8, "0.2 + 0.12*(r % 4)",
        lambda r: 0.2 + 0.12 * (r % 4), "0.2+0.12*(r%4)")
    _mk("r68b_bidi_b04m3_d8_p2048", 2048, 8, "0.4 + 0.08*(r % 3)",
        lambda r: 0.4 + 0.08 * (r % 3), "0.4+0.08*(r%3)")
    _mk("r68b_bidi_b025m5_d8_p2048", 2048, 8, "0.25 + 0.12*(r % 5)",
        lambda r: 0.25 + 0.12 * (r % 5), "0.25+0.12*(r%5)")
    _mk("r68b_bidi_b03m6_d8_p2048", 2048, 8, "0.3 + 0.08*(r % 6)",
        lambda r: 0.3 + 0.08 * (r % 6), "0.3+0.08*(r%6)")
    _mk("r68b_bidi_b015m3_d8_p1024", 1024, 8, "0.15 + 0.1*(r % 3)",
        lambda r: 0.15 + 0.1 * (r % 3), "0.15+0.1*(r%3)")
    _mk("r68b_bidi_b035m4_d8_p1024", 1024, 8, "0.35 + 0.09*(r % 4)",
        lambda r: 0.35 + 0.09 * (r % 4), "0.35+0.09*(r%4)")
    _mk("r68b_bidi_b022m5_d8_p1024", 1024, 8, "0.22 + 0.11*(r % 5)",
        lambda r: 0.22 + 0.11 * (r % 5), "0.22+0.11*(r%5)")
    _mk("r68b_bidi_b018m4_d8_p1024", 1024, 8, "0.18 + 0.13*(r % 4)",
        lambda r: 0.18 + 0.13 * (r % 4), "0.18+0.13*(r%4)")


register_all()
