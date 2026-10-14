#!/bin/bash
# queue33 (after queue32): measure the deployed AllToAllV runtimes with the helper routed through evolved_alltoallv
# (the shipped codegen helper is a hard-coded AG+T+RS, so the plain measurement does not exercise the deployed algorithm).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE32 DONE" $Q; do sleep 20; done
stage "QUEUE33 wrapped-helper measurement of deployed AllToAllV runtimes (snapshot, live, v4live), 3 repeats"
python3 - <<'PY'
import json, os, subprocess
man=[]
for a in ("snapshot","live","v4live"):
    src=f"/home/ubuntu/livecal/alltoallv_{a}_runtime.py"; dst=f"/home/ubuntu/livecal/alltoallv_{a}_runtime_wrapped.py"
    if os.path.exists(src):
        r=subprocess.run(["python3","/home/ubuntu/ap7/wrap_a2av_helper.py",src,dst]); 
        if r.returncode==0: man.append({"problem":"alltoallv","cand":f"livecal_{a}_wrapped","kind":"file","path":dst})
json.dump(man, open("/home/ubuntu/ap7/manifest_livecal_wrapped.json","w"), indent=1); print(len(man),"wrapped runtimes")
PY
for r in 1 2 3; do MANIFEST=$AP/manifest_livecal_wrapped.json SWEEP_OUT=/home/ubuntu/sweep_livecal_wrapped_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE33 DONE"
