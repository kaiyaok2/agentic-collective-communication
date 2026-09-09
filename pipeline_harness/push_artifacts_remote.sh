#!/bin/bash
# Push Sorcar/strat generated code + cost numbers. Runs on the restore instance
# against the mounted snapshot volume. PAT passed as $1 (never echoed).
set -u
PAT="${1:?usage: push_artifacts_remote.sh <PAT>}"
BASE=/mnt/acc/home/ubuntu
REPO=$BASE/acc
BRANCH="taxonomy-3col-rt-2026-09-09"
OUT="$REPO/taxonomy_3col_results"
cd "$REPO" || exit 1

# git needs to write; ensure ownership is ours on the mount
sudo chown -R ubuntu:ubuntu "$REPO" 2>/dev/null

mkdir -p "$OUT/sorcar_code" "$OUT/strat_code" "$OUT/rt_json"

# 1) top-level result docs/data
cp "$BASE/three_col.json"                 "$OUT/" 2>/dev/null
cp "$BASE/rt_three_col.json"              "$OUT/" 2>/dev/null
cp "$REPO/RT_THREE_COL_RESULTS.json"      "$OUT/" 2>/dev/null
cp "$BASE"/rt_sweep/*.json                "$OUT/rt_json/" 2>/dev/null

# 2) Sorcar winner code + per-problem search cost (kiss_summary.json)
for d in "$BASE"/sorcar_sweep/*/; do
  nm=$(basename "$d")
  [ -f "$d/best_code.py" ]     && cp "$d/best_code.py"     "$OUT/sorcar_code/${nm}.py" 2>/dev/null
  [ -f "$d/kiss_summary.json" ] && cp "$d/kiss_summary.json" "$OUT/sorcar_code/${nm}.cost.json" 2>/dev/null
done

# 3) strat winner code (deployed runtime) + strat sweep result json (cost/timing)
for d in "$BASE"/strat_sweep/*/; do
  nm=$(basename "$d")
  f=$(ls "$d"/results_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" "$OUT/strat_code/${nm}.result.json" 2>/dev/null
done
cp "$REPO"/runtime/trainium_*_7node.py "$OUT/strat_code/" 2>/dev/null

# 4) compact cost summaries (Sorcar search cost + strat winner sim time)
python3 - <<PY
import json, glob, os
base="$BASE"
# Sorcar search cost
srows=[]
for f in glob.glob(base+'/sorcar_sweep/*/kiss_summary.json'):
    nm=os.path.basename(os.path.dirname(f))
    try:
        d=json.load(open(f))
        srows.append({'problem':nm,'n_score_calls':d.get('n_score_calls'),
                      'wall_seconds':d.get('wall_seconds'),
                      'baseline_sim_us':d.get('baseline_sim_time_us'),
                      'best_sim_us':d.get('best_sim_time_us'),
                      'best_name':d.get('best_name'),'hit_target':d.get('hit_target')})
    except Exception as e:
        srows.append({'problem':nm,'error':str(e)[:60]})
json.dump({'n':len(srows),'rows':sorted(srows,key=lambda r:r['problem'])},
          open("$OUT/sorcar_cost_summary.json",'w'),indent=2)
# strat winner (best cost_score row per problem)
trows=[]
for f in glob.glob(base+'/strat_sweep/*/results_*.json'):
    nm=os.path.basename(os.path.dirname(f))
    try:
        arr=json.load(open(f))
        best=min(arr,key=lambda r:r.get('cost_score',float('inf')))
        trows.append({'problem':nm,'winner':best.get('name'),
                      'sim_time_us':best.get('sim_time_us'),
                      'num_all_reduce':best.get('num_all_reduce'),
                      'num_all_gather':best.get('num_all_gather'),
                      'num_collective_permute':best.get('num_collective_permute'),
                      'n_templates':len(arr)})
    except Exception as e:
        trows.append({'problem':nm,'error':str(e)[:60]})
json.dump({'n':len(trows),'rows':sorted(trows,key=lambda r:r['problem'])},
          open("$OUT/strat_cost_summary.json",'w'),indent=2)
print('sorcar cost rows',len(srows),'strat cost rows',len(trows))
PY

echo "sorcar_code files: $(ls $OUT/sorcar_code | wc -l)"
echo "strat_code files:  $(ls $OUT/strat_code | wc -l)"
echo "rt_json files:     $(ls $OUT/rt_json | wc -l)"

# 5) orphan commit (author Kaiyao Ke, committer kaiyaok2, no Claude attribution)
git checkout --orphan "$BRANCH" 2>/dev/null || git checkout "$BRANCH" 2>/dev/null
git reset >/dev/null 2>&1
git add taxonomy_3col_results
GIT_AUTHOR_NAME="Kaiyao Ke" GIT_AUTHOR_EMAIL="kaiyaoke@berkeley.edu" \
GIT_COMMITTER_NAME="kaiyaok2" GIT_COMMITTER_EMAIL="kaiyaoke@berkeley.edu" \
git commit -q -m "3-column taxonomy: baseline / measured-strat / Sorcar + warm-cache RT

143/143 both controllers on the single current==OverlayCCL Phase-1 sim, 7 nodes.
Sorcar 56 sim wins >5%, 0 strat wins; strat==baseline on every divergence.
Warm-cache RT (224 ranks) of the 56 divergent: 46 Sorcar RT wins, 0 strat,
8 dispatch-floor ties, 2 Sorcar sim-pass/HW-abort (reduce_scatter shard_count=224
on non-divisible tensors). Includes per-problem Sorcar/strat winner code and
search cost numbers (score calls, sim times)."
echo "commit rc=$? sha=$(git rev-parse --short HEAD 2>/dev/null)"

# 6) push (PAT only in URL; scrub any echo)
REMOTE="https://${PAT}@github.com/kaiyaok2/agentic-collective-communication.git"
git push -f "$REMOTE" "$BRANCH" 2>&1 | sed -E 's/ghp_[A-Za-z0-9]+/ghp_***/g'
echo "push rc=${PIPESTATUS[0]}"
