"""Patch search/agent_simulator_config.py so Phase-1 tools answer from a LIVE probe table when OVERLAY_HW_TABLE points at a
train_hwprobe_7node.py JSON (derived_table), instead of the shipped static snapshot. Keys missing from the live table
(collective_permute latency, overlap_effects) fall back to the snapshot. Usage: python3 live_probe_table.py <repo>"""
import sys, re
p = sys.argv[1] + "/search/agent_simulator_config.py"; s = open(p).read()
anchor = "def _test_primitive_compilation(primitive):"
assert anchor in s and "OVERLAY_HW_TABLE" not in s
inject = '''
# --- rebuttal: live Phase-1 table -------------------------------------------------------------------------------
# If OVERLAY_HW_TABLE is set, replace the static snapshot above with the on-device probe results
# (training/train_hwprobe_7node.py -> hw_probe_measured.json, key "derived_table"). The probe writes the same
# schema; keys it could not measure keep the snapshot values so every tool still answers.
import os as _os, json as _json
_LIVE = _os.environ.get("OVERLAY_HW_TABLE")
if _LIVE:
    _live = _json.load(open(_LIVE)).get("derived_table", {})
    for _k, _v in _live.items():
        if isinstance(_v, dict) and _k in _HARDWARE_MEASUREMENTS and isinstance(_HARDWARE_MEASUREMENTS[_k], dict):
            _HARDWARE_MEASUREMENTS[_k] = {**_HARDWARE_MEASUREMENTS[_k], **{kk: vv for kk, vv in _v.items() if vv is not None}}
        else:
            _HARDWARE_MEASUREMENTS[_k] = _v
    _HARDWARE_MEASUREMENTS["_source"] = f"live probe {_LIVE}"
    print(f"[Phase 1] hardware table: LIVE probe ({_LIVE})")
else:
    _HARDWARE_MEASUREMENTS["_source"] = "static snapshot"
    print("[Phase 1] hardware table: STATIC SNAPSHOT (set OVERLAY_HW_TABLE to use live probes)")

'''
s = s.replace(anchor, inject + anchor, 1); open(p, "w").write(s); print("patched", p)
