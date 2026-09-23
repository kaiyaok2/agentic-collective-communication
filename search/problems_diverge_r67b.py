"""Round 67b -- FAMILY-6 TOP-UP (dirB self-coupling rank-1 coupling), toward 10.

Family-6 stands at 5/10 (2 Walsh + 3 self). This battery adds fresh betas at the proven
p2048 payload for the self-coupling direction (u=v, full Sherman-Morrison inverse
1/(1+beta)). Reuses the EXACT r67 mechanism/telescoping via import (out = s + beta*v*mean(v*s));
new betas -> distinct reference md5 from every registered problem (prescreen guard). Depth is
NOT distinctness.
"""
from .problems_diverge_r67 import _mk  # noqa: reuse self-coupling rank-1 mechanism + telescoping


def register_all():
    _mk("r67b_vself_b0p3_p2048_d8", 2048, 8, 0.3)
    _mk("r67b_vself_b0p4_p2048_d8", 2048, 8, 0.4)
    _mk("r67b_vself_b0p7_p2048_d8", 2048, 8, 0.7)
    _mk("r67b_vself_b1p5_p2048_d8", 2048, 8, 1.5)
    _mk("r67b_vself_b3p0_p2048_d8", 2048, 8, 3.0)


register_all()
