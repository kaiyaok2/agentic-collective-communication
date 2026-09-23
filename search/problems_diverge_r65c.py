"""Round 65c -- FAMILY-5 TOP-UP #2 (multiplicative global normalization), close to 10.

Family-5 stands at 9/10 after r65b confirmed 6. This battery adds four fresh betas at the
proven p2048 payload (where every r65b confirmation clustered) to secure the final slot.
Reuses the EXACT r65 gnorm mechanism/telescoping via import (out = s / (1 + beta*mean(|s|)));
new betas -> distinct reference md5 from every registered gnorm problem (prescreen guard).
Depth is NOT distinctness.
"""
from .problems_diverge_r65 import _mk  # noqa: reuse gnorm mechanism + telescoping


def register_all():
    _mk("r65c_gnorm_b0p8_p2048_d8", 2048, 8, 0.8)
    _mk("r65c_gnorm_b0p9_p2048_d8", 2048, 8, 0.9)
    _mk("r65c_gnorm_b1p2_p2048_d8", 2048, 8, 1.2)
    _mk("r65c_gnorm_b2p2_p2048_d8", 2048, 8, 2.2)


register_all()
