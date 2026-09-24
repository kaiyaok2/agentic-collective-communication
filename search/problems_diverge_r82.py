"""Round 82 -- FAMILY-6 TOP-UP #2 (dirB self-coupling rank-1, full Sherman-Morrison
inverse 1/(1+beta), out = s + beta*v*mean(v*s)). Reuses the EXACT r67 mechanism/
telescoping via import. The confirmed fam6 survivors sit at beta=0.35/0.40 (p2048);
this battery sweeps the productive neighborhood BELOW/AROUND 0.35 (0.30-0.37) at
p2048_d8 to top up the family further. Betas 0.35/0.38/0.40/0.42/0.45 are already
registered (r67b/r79) -> these four (0.30,0.32,0.33,0.37) are all new, distinct
reference md5 (prescreen guard). Depth is NOT distinctness.
"""
from .problems_diverge_r67 import _mk  # reuse self-coupling rank-1 mechanism + telescoping


def register_all():
    # Only beta=0.30 survived the fair sim gate (best 1.278 / p 0.0003 /
    # CI-lo 1.0012). The 0.32/0.33/0.37 sweep siblings did not clear both
    # gates (CI-lo pinned at/near 1.0) and were pruned.
    _mk("r82_vself_b0p30_p2048_d8", 2048, 8, 0.30)


register_all()
