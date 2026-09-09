#!/bin/bash
set -uo pipefail
# Full 3-column warm-cache RT over the divergent taxonomy set.
# For each problem: baseline (builtin template) / strat (runtime/trainium_<nm>_7node.py)
# / sorcar (sorcar_sweep/<nm>/best_code.py). Runs sequentially (shared 224 ranks).
NAMES_FILE=${1:-/home/ubuntu/divergent_56.txt}
RT_OUTDIR=${RT_OUTDIR:-/home/ubuntu/rt_sweep}
CODE=/home/ubuntu/rt_code
mkdir -p "$RT_OUTDIR" "$CODE"
export RT_OUTDIR
BASE_PORT=${BASE_PORT:-32700}

# 1) extract baseline templates for all problems in one python pass
/opt/aws_neuronx_venv_pytorch_2_8/bin/python - "$NAMES_FILE" "$CODE" <<'PYEOF'
import os, sys
sys.path.insert(0, "/home/ubuntu/acc")
import search.problems_all_catalogs
from search.problems import get_problem
names_file, code_dir = sys.argv[1], sys.argv[2]
os.makedirs(code_dir, exist_ok=True)
for line in open(names_file):
    nm = line.strip()
    if not nm:
        continue
    try:
        p = get_problem(nm)
        k, code = next(iter(p.builtin_templates.items()))
        open(os.path.join(code_dir, nm + ".baseline.py"), "w").write(code)
    except Exception as e:
        print("BASELINE_FAIL", nm, type(e).__name__, str(e)[:80])
print("baselines extracted")
PYEOF

port=$BASE_PORT
while IFS= read -r nm <&3; do
  nm=$(echo "$nm" | tr -d '[:space:]'); [ -z "$nm" ] && continue
  base_f="$CODE/${nm}.baseline.py"
  strat_f="/home/ubuntu/acc/runtime/trainium_${nm}_7node.py"
  sorcar_f="/home/ubuntu/sorcar_sweep/${nm}/best_code.py"
  for tag_f in "baseline:$base_f" "strat:$strat_f" "sorcar:$sorcar_f"; do
    tag="${tag_f%%:*}"; f="${tag_f#*:}"
    if [ ! -f "$f" ]; then echo "[rt-sweep] SKIP $nm/$tag (no file $f)"; continue; fi
    if [ -f "$RT_OUTDIR/${nm}.${tag}.json" ]; then echo "[rt-sweep] SKIP $nm/$tag (done)"; continue; fi
    port=$((port+1))
    echo "[rt-sweep] $(date -u) === $nm / $tag (port=$port) ==="
    PORT=$port bash /home/ubuntu/rt_launch.sh "$nm" "$tag" "$f" </dev/null 2>&1 | tail -3
  done
done 3< "$NAMES_FILE"
echo "[rt-sweep] ALL DONE $(date -u)"
