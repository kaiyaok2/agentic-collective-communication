import json, os, sys
sys.path.insert(0, "/home/ubuntu/scripts"); import programs
C = "/home/ubuntu/corpus"; n = 0
for s in programs.grid_collective():
    p = s["params"]
    if p["numel"] == 16777216 or (p["count"] == 8 and p["kind"] in ("reduce_scatter", "ag_rs")):
        d = f"{C}/{s['id']}"
        if os.path.exists(f"{d}/meta.json") or os.path.exists(f"{d}/spec.json"): continue
        os.makedirs(d, exist_ok=True); json.dump(s, open(f"{d}/spec.json", "w"))
        json.dump({"spec": s, "ok": False, "error": "SKIPPED: time budget", "modules": []}, open(f"{d}/meta.json", "w")); n += 1
print("collective specs deferred:", n)
