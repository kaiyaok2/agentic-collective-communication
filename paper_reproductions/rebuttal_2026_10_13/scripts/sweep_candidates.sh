#!/bin/bash
# Sim-vs-measured sweep (A-Q2 / C-Q1): measure every manifest candidate at training scope on the
# full cluster with the per-problem harness (training/train_<p>_7node.py), 300 steps each.
# Run ON master:  bash sweep_candidates.sh [problem-filter-regex]
# Results: /home/ubuntu/sweep_results/<problem>__<cand>.json  (+ sweep.log)
set -uo pipefail
FILTER=${1:-.}
REPO=/home/ubuntu/OverlayCCL
AP=/home/ubuntu/ap7
OUT=${SWEEP_OUT:-/home/ubuntu/sweep_results}; mkdir -p $OUT
export RESULTS_DIR=$OUT/raw; mkdir -p $RESULTS_DIR
STEPS=${STEPS:-300}
source /home/ubuntu/venv.env
python3 - "$FILTER" <<'PY' > /tmp/sweep_plan.txt
import json,sys,re
flt=re.compile(sys.argv[1])
import os
for c in json.load(open(os.environ.get('MANIFEST','/home/ubuntu/ap7/manifest.json'))):
    if flt.search(c['problem']): print(c['problem'], c['cand'], c['kind'], c.get('path') or c.get('name') or c.get('code') or '-')
PY
echo "=== sweep: $(wc -l < /tmp/sweep_plan.txt) runs, $STEPS steps each ===" | tee -a $OUT/sweep.log
while read -r P C K X; do
  TAG="${P}__${C}"
  if [ -f $OUT/$TAG.json ] && python3 -c "import json,sys; d=json.load(open('$OUT/$TAG.json')); sys.exit(0 if d.get('measured_ms') is not None else 1)"; then echo "skip $TAG (done)"; continue; fi
  rm -f $OUT/$TAG.json
  echo "--- [$(date -u +%T)] $TAG" | tee -a $OUT/sweep.log
  case $K in
    file)     python3 $AP/deploy_candidate.py $P --file $X; BACKEND=agent ;;
    builtin)  python3 $AP/deploy_candidate.py $P --builtin $X; BACKEND=agent ;;
    code)     python3 $AP/deploy_candidate.py $P --code $X; BACKEND=agent ;;
    baseline) python3 $AP/deploy_candidate.py $P --restore; BACKEND=baseline ;;
  esac
  # clear stale outputs of this backend on all nodes (every candidate of a backend writes the same file name)
  CLEAR="rm -f $RESULTS_DIR/*${BACKEND}_percall.json /tmp/tp_search/*${BACKEND}_percall.json $RESULTS_DIR/${P}_${BACKEND}_step.json $RESULTS_DIR/${P}_7node_${BACKEND}.json $RESULTS_DIR/ua2a_7node_${BACKEND}.json /tmp/tp_search/${P}_${BACKEND}.json"
  eval "$CLEAR"; for ip in $WORKERS; do ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip "mkdir -p /tmp/tp_search $RESULTS_DIR; $CLEAR" </dev/null; done
  HARNESS=training/train_${P}_7node.py
  if [ "$P" = pp_send_recv ]; then HARNESS=training/train_pp_send_recv_7node_s512.py; [ -f $REPO/$HARNESS ] || sed -E "s/^S *= *1024/S = 512/" $REPO/training/train_pp_send_recv_7node.py > $REPO/$HARNESS; fi
  timeout ${RUN_TIMEOUT:-2400} bash $AP/run_ap7.sh $HARNESS $BACKEND $STEPS $TAG >> $OUT/sweep.log 2>&1
  RC=$?
  # harvest immediately from master or any worker, any of the four naming conventions
  F=""
  for cand in "$RESULTS_DIR/${P}_${BACKEND}_step.json" "$RESULTS_DIR/${P}_7node_${BACKEND}.json" "$RESULTS_DIR/ua2a_7node_${BACKEND}.json" "/tmp/tp_search/${P}_${BACKEND}.json"; do
    [ -f "$cand" ] && { F=$cand; break; }
    for ip in $WORKERS; do
      scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$cand $OUT/raw/harvest_${TAG}.json 2>/dev/null && { F=$OUT/raw/harvest_${TAG}.json; break; }
    done
    [ -n "$F" ] && break
  done
  # also harvest the per-call probe json (collective-only time per call) from master or any worker
  for pc in "$RESULTS_DIR/${P}_${BACKEND}_percall.json" "$RESULTS_DIR/${P}_7node_${BACKEND}_percall.json" "$RESULTS_DIR/ua2a_7node_${BACKEND}_percall.json" "/tmp/tp_search/${P}_${BACKEND}_percall.json"; do
    if [ -f "$pc" ]; then cp "$pc" $OUT/${TAG}_percall.json; break; fi
    for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$pc $OUT/${TAG}_percall.json 2>/dev/null && break 2; done
  done
  if [ -n "$F" ]; then
    python3 - "$F" "$OUT/$TAG.json" "$P" "$C" "$K" "$RC" <<'PY2'
import json,sys,statistics,os
d=json.load(open(sys.argv[1])); d.update(problem=sys.argv[3], cand=sys.argv[4], kind=sys.argv[5], rc=int(sys.argv[6]))
# normalized measured_ms: prefer steady median/mean, else recompute from per-step list, else avg
m=None; src="none"
ts=d.get("all_ms") or d.get("step_times_ms")
PS=int(os.environ.get("PROBE_START_STEP","280"))
if ts and len(ts)>=PS:          # per-step list available: median over the clean steady window, excluding the probe window
    m=statistics.median(ts[len(ts)//2:PS]); src=f"median(steps {len(ts)//2}-{PS})"
elif ts and len(ts)>60:
    m=statistics.median(ts[len(ts)//2:]); src="median(second half)"
else:
    for k in ("steady_median_ms","steady_ms","steady_mean_ms"):
        if d.get(k) is not None: m=float(d[k]); src=k; break
    if m is None and d.get("avg_ms") is not None: m=float(d["avg_ms"]); src="avg_ms"
d["measured_ms"]=m; d["measured_src"]=src
d["steady_window_ms"]=[round(x,2) for x in (ts[len(ts)//2:PS] if ts and len(ts)>=PS else (ts or []))][:200]
for k in ("all_ms","step_times_ms","losses"): d.pop(k,None)
json.dump(d, open(sys.argv[2],'w'), indent=1)
print(f"   measured_ms={m}  (avg_ms={d.get('avg_ms')} steady={d.get('steady_ms') or d.get('steady_median_ms')}) rc={d['rc']}")
PY2
  else
    echo "   NO RESULT (rc=$RC)" | tee -a $OUT/sweep.log
    echo "{\"problem\":\"$P\",\"cand\":\"$C\",\"kind\":\"$K\",\"rc\":$RC,\"error\":\"no result json\"}" > $OUT/$TAG.json
  fi
  python3 $AP/deploy_candidate.py $P --restore >/dev/null
done < /tmp/sweep_plan.txt
echo "=== sweep done $(date -u +%T) ===" | tee -a $OUT/sweep.log
