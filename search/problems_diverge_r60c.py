"""Round 60c -- FAMILY-2 REDESIGN. The r60 (B=5/6, tiny payload) batch tied almost
everywhere: overlay also finds the count-fold when the depth-D chain sits near the sim
dispatch floor, so median ratios were ~1.0. This batch returns to the archival CONFIRMING
structure (r40: B=8, contiguous length-L window, start=(rank+2) mod B, depth 8) but with
HEAVY payloads (part 2048/4096) so that whenever overlay's k=5 search misses the count-fold
the ratio is large, and adds larger block counts B in {9,10,11,12} for non-uniform counts.

Distinctness: for B=8 the count at W=224 is the uniform constant W*L/B, so different L gives
a different constant AND different payload gives a different vector length -> distinct md5
from r40 (B8,L3,part256/1024) and r43 (B9,part256). Depth is NOT used for distinctness
(the count reference is depth-independent).
"""
import torch  # noqa: F401
from .problems_diverge_r40 import _mk as _mk40  # (name, part, depth, L, off, count_cue)
from .problems_diverge_r60 import _mk as _mk60  # (name, spec, depth, cue)


def register_all():
    # B=8, r40 structure (off=2), heavy payloads, varied L (distinct uniform constants).
    _mk40("r60c_b8_L3_p2048_d8", 2048, 8, 3, 2, True)
    _mk40("r60c_b8_L4_p2048_d8", 2048, 8, 4, 2, True)
    _mk40("r60c_b8_L2_p2048_d8", 2048, 8, 2, 2, True)
    _mk40("r60c_b8_L5_p2048_d8", 2048, 8, 5, 2, True)
    _mk40("r60c_b8_L3_p4096_d8", 4096, 8, 3, 2, True)
    # Larger block counts (non-uniform counts at W=224), heavy payload.
    _mk60("r60c_b9_L3_p2048_d8",  {"shape": "contig", "B": 9,  "off": 2, "L": 3, "stride": 1, "part": 2048}, 8, True)
    _mk60("r60c_b10_L4_p2048_d8", {"shape": "contig", "B": 10, "off": 2, "L": 4, "stride": 1, "part": 2048}, 8, True)
    _mk60("r60c_b11_L4_p2048_d8", {"shape": "contig", "B": 11, "off": 2, "L": 4, "stride": 1, "part": 2048}, 8, True)
    _mk60("r60c_b12_L5_p2048_d8", {"shape": "contig", "B": 12, "off": 2, "L": 5, "stride": 1, "part": 2048}, 8, True)


register_all()
