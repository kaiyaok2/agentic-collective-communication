"""Round 79 -- FAMILY-6 TOP-UP (dirB self-coupling rank-1, full Sherman-Morrison
inverse 1/(1+beta), out = s + beta*v*mean(v*s)). Reuses the EXACT r67 mechanism/
telescoping via import. The confirmed fam6 survivor sits at beta=0.4/p2048; this
battery sweeps betas immediately around it (0.35-0.45) at p2048 to top up the family.
New betas -> distinct reference md5 (prescreen guard). Depth is NOT distinctness.
"""
from .problems_diverge_r67 import _mk  # reuse self-coupling rank-1 mechanism + telescoping


def register_all():
    # Only beta=0.35 survived the fair sim+RT gate (best 1.053 / p 0.0011 /
    # CI-lo 1.0014; warm-cache RT 1.12x Sorcar>Overlay at 224 ranks). The
    # 0.38/0.42/0.45 sweep siblings did not clear both gates and were pruned.
    _mk("r79_vself_b0p35_p2048_d8", 2048, 8, 0.35)


register_all()
