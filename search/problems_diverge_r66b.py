"""Round 66b -- FAMILY-6 TOP-UP (dirA orthogonal-Walsh rank-1 coupling), toward 10.

Family-6 stands at 5/10 (2 Walsh + 3 self). This battery adds fresh betas at the proven
p2048 payload (where the confirmations clustered) for the Walsh direction. Reuses the EXACT
r66 mechanism/telescoping via import (out = s + beta*u*mean(v*s), v=Walsh(1), u=Walsh(2),
v perpendicular u -> trivial-denominator inverse); new betas -> distinct reference md5 from
every registered problem (prescreen guard). Depth is NOT distinctness.
"""
from .problems_diverge_r66 import _mk  # noqa: reuse Walsh rank-1 mechanism + telescoping


def register_all():
    _mk("r66b_walsh_b0p3_p2048_d8", 2048, 8, 0.3)
    _mk("r66b_walsh_b0p4_p2048_d8", 2048, 8, 0.4)
    _mk("r66b_walsh_b0p7_p2048_d8", 2048, 8, 0.7)
    _mk("r66b_walsh_b1p5_p2048_d8", 2048, 8, 1.5)
    _mk("r66b_walsh_b3p0_p2048_d8", 2048, 8, 3.0)


register_all()
