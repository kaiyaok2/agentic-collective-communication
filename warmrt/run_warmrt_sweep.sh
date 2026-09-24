#!/bin/bash
# Full warm-cache RT sweep over every staged confirmed problem: for each, run the
# median (middle) seed's Sorcar code and Overlay code across all 7 nodes and record
# MS_PER_ITER for both. Idempotent: skips a (problem,pipeline) already in results.
# Runs ON the master after bootstrap. Usage: run_warmrt_sweep.sh <MASTER_IP> "<W1..W6>" [N_ITERS] [WARM_REPS]
set -u
MASTER_IP=$1; WORKERS=$2; N_ITERS=${3:-50}; WARM_REPS=${4:-3}
BASE=/home/ubuntu/agentic-collective-communication/warmrt
STAGE=$BASE/stage
OUT=/home/ubuntu/warmrt_results
mkdir -p "$OUT"
RES=$OUT/results.jsonl
MAN=$STAGE/master_manifest.json

# problems = every fam/prob dir under stage that has both files
mapfile -t KEYS < <(cd "$STAGE" && for f in */*/sorcar.py; do d=$(dirname "$f"); [ -f "$STAGE/$d/overlay.py" ] && echo "$d"; done | sort)
echo "sweep: ${#KEYS[@]} problems x {sorcar,overlay} x ${WARM_REPS} reps @ ${N_ITERS} iters"

for KEY in "${KEYS[@]}"; do
  FAM=$(dirname "$KEY"); PROB=$(basename "$KEY")
  for PIPE in sorcar overlay; do
    F=$STAGE/$KEY/$PIPE.py
    [ -f "$F" ] || { echo "[skip] no $F"; continue; }
    # median of WARM_REPS (first rep warms the Neuron cache; keep all, report median)
    declare -a MSV=()
    for rep in $(seq 1 "$WARM_REPS"); do
      MS=$(bash "$BASE/rt_7node.sh" "$F" "$PROB" "$MASTER_IP" "$WORKERS" "$N_ITERS" 2>/dev/null | grep '^MS_PER_ITER=' | cut -d= -f2)
      echo "  [$FAM/$PROB $PIPE rep$rep] MS_PER_ITER=${MS:-FAIL}"
      [ -n "${MS:-}" ] && [ "$MS" != "FAIL" ] && MSV+=("$MS")
    done
    # median of the collected reps (warm-cache: drop rep1 if >=3 reps)
    if [ "${#MSV[@]}" -gt 0 ]; then
      WARMV=("${MSV[@]}"); [ "${#MSV[@]}" -ge 3 ] && WARMV=("${MSV[@]:1}")
      MED=$(printf '%s\n' "${WARMV[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
    else MED="null"; fi
    printf '{"family":"%s","problem":"%s","pipeline":"%s","ms_median":%s,"reps":[%s]}\n' \
      "$FAM" "$PROB" "$PIPE" "$MED" "$(IFS=,; echo "${MSV[*]:-}")" >> "$RES"
    unset MSV
  done
done
echo "=== sweep done -> $RES ==="
# quick per-problem sorcar-vs-overlay speedup table
/opt/aws_neuronx_venv_pytorch_2_9/bin/python - "$RES" <<'PY'
import json,sys,collections
rows=collections.defaultdict(dict)
for ln in open(sys.argv[1]):
    try: r=json.loads(ln)
    except: continue
    if r.get("ms_median") not in (None,"null"): rows[(r["family"],r["problem"])][r["pipeline"]]=r["ms_median"]
print(f"{'family/problem':44s} {'sorcar_ms':>10s} {'overlay_ms':>10s} {'speedup(ov/sc)':>14s}")
for (fam,prob),d in sorted(rows.items()):
    sc,ov=d.get("sorcar"),d.get("overlay")
    sp=f"{ov/sc:.3f}x" if (sc and ov and sc>0) else "-"
    print(f"{fam+'/'+prob:44s} {str(sc):>10s} {str(ov):>10s} {sp:>14s}")
PY
