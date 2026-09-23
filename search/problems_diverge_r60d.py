"""Round 60d -- FAMILY-2 TOP-UP (non-duplicate): rank-INDEXED ROUTING -> per-block
count c[b] diagonal collapse. Same structure that CONFIRMED in r60c (B blocks, contiguous
length-L keep-window starting at (rank+2) mod B, depth 8, HEAVY payload part=2048), but
NEW (B, L) combinations not previously registered. r60c confirmed at (B,L) in
{(8,2),(9,3),(10,4),(12,5)} and tied at {(8,3),(8,4),(8,5),(11,4)} and p4096; this batch
fills in nearby non-duplicate (B, L) pairs at the proven p2048 payload.

Distinctness: the family-2 reference is the per-block count c[b] (how many ranks route
into block b), which depends on (B, L, off, stride, part). Every (B, L) pair here is new
vs r40/r43/r60/r60c. Depth is NOT used for distinctness. The prescreen md5 cross-check
(vs all registered problems) is the guard.
"""
import torch  # noqa: F401
from .problems_diverge_r60 import _mk as _mk60  # (name, spec, depth, cue)


def _spec(B, L, part=2048, off=2, stride=1):
    return {"shape": "contig", "B": B, "off": off, "L": L, "stride": stride, "part": part}


def register_all():
    _mk60("r60d_b9_L2_p2048_d8",  _spec(9, 2),  8, True)
    _mk60("r60d_b10_L3_p2048_d8", _spec(10, 3), 8, True)
    _mk60("r60d_b10_L5_p2048_d8", _spec(10, 5), 8, True)
    _mk60("r60d_b13_L3_p2048_d8", _spec(13, 3), 8, True)
    _mk60("r60d_b13_L4_p2048_d8", _spec(13, 4), 8, True)
    _mk60("r60d_b14_L4_p2048_d8", _spec(14, 4), 8, True)
    _mk60("r60d_b16_L5_p2048_d8", _spec(16, 5), 8, True)


register_all()
