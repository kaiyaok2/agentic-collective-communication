"""Round 65b -- FAMILY-5 TOP-UP (multiplicative global normalization), toward 10 problems.

Reuses the EXACT gnorm mechanism/telescoping from r65 (out = s / (1 + beta*mean(|s|)); import
guarantees identical fold, no divergence risk) and widens the (beta, payload) net around the
three confirmed winners (b0.5_p1024, b0.5_p2048, b2.0_p2048). Family-5's confirmations cluster
at beta away from 1.0 (b1.0 failed both payloads), so this battery targets betas near 0.5 and
2.0 at the p2048/p1024 payloads. All new combos are md5-distinct from registered (prescreen
guard). Depth is NOT distinctness.
"""
from .problems_diverge_r65 import _mk  # noqa: reuse gnorm mechanism + telescoping


def register_all():
    _mk("r65b_gnorm_b0p3_p2048_d8", 2048, 8, 0.3)
    _mk("r65b_gnorm_b0p4_p2048_d8", 2048, 8, 0.4)
    _mk("r65b_gnorm_b0p6_p2048_d8", 2048, 8, 0.6)
    _mk("r65b_gnorm_b0p7_p2048_d8", 2048, 8, 0.7)
    _mk("r65b_gnorm_b1p5_p2048_d8", 2048, 8, 1.5)
    _mk("r65b_gnorm_b2p5_p2048_d8", 2048, 8, 2.5)
    _mk("r65b_gnorm_b3p0_p2048_d8", 2048, 8, 3.0)
    _mk("r65b_gnorm_b0p5_p512_d8",  512,  8, 0.5)
    _mk("r65b_gnorm_b0p6_p1024_d8", 1024, 8, 0.6)
    _mk("r65b_gnorm_b2p0_p1024_d8", 1024, 8, 2.0)


register_all()
