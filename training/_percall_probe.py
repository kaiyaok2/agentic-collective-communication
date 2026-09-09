"""Per-call timing probes — late-phase .item()-based variant.

Three env vars:
  PERCALL_PROBE=1            : enable probing at all
  PROBE_START_STEP=<int>     : enable probes only from this step onward (default 0)
  PROBE_END_STEP=<int>       : disable probes after this step (default 999999)

Late-phase probing avoids the NRT_RESOURCE / ENC_MAX_COMM_N runtime limits
that .item()-sync probes hit when applied across all steps of an
OLMoE-scale training run: by running the first ~150 steps WITHOUT probes,
the NEFF cache is fully populated and stabilised; the next ~50 steps run
WITH .item() probes, adding only a bounded number of additional NEFFs.
The probed window provides accurate per-call execution latency.
"""
import os
import time as _t
import json
import contextlib

ENABLED = os.environ.get("PERCALL_PROBE", "0") == "1"
PROBE_START_STEP = int(os.environ.get("PROBE_START_STEP", "0"))
PROBE_END_STEP = int(os.environ.get("PROBE_END_STEP", "999999"))
_current_step = 0
_BUFFERS = {}


def set_step(s):
    """Training loop calls this before each step's collectives so probes can
    self-gate by step index."""
    global _current_step
    _current_step = int(s)


def in_window():
    return ENABLED and PROBE_START_STEP <= _current_step <= PROBE_END_STEP


def record(name, ms):
    if ENABLED:
        _BUFFERS.setdefault(name, []).append(float(ms))


def record_after_step(name, start_t):
    """Backward-compat: callback for xm.add_step_closure. Not recommended
    for late-phase work (use .item() instead)."""
    if ENABLED:
        _BUFFERS.setdefault(name, []).append((_t.time() - start_t) * 1000.0)


@contextlib.contextmanager
def timed(name):
    if not ENABLED:
        yield
        return
    t0 = _t.time()
    try:
        yield
    finally:
        _BUFFERS.setdefault(name, []).append((_t.time() - t0) * 1000.0)


def buffers():
    return _BUFFERS


def reset():
    _BUFFERS.clear()


def dump(path, extra=None):
    if not ENABLED:
        return
    out = dict(_BUFFERS)
    if extra:
        out["_meta"] = extra
    out["_meta"] = out.get("_meta", {})
    out["_meta"]["probe_start_step"] = PROBE_START_STEP
    out["_meta"]["probe_end_step"] = PROBE_END_STEP
    with open(path, "w") as f:
        json.dump(out, f)
