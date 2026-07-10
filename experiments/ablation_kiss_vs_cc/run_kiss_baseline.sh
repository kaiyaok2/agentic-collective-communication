#!/bin/bash
# kiss-sorcar-react driver: runs the 8-problem Phase-3 ablation. Reads each
# problem's cc-react `results_$P.json` to compute a per-problem ±5% target
# and passes it to kiss via --target-score so kiss exits early once the
# score plateaus within the accept band.
#
# Environment (all optional; sensible defaults):
#   ACC_REPO       Path to the acc repo (default: two dirs up from this file).
#   ABLATION_WORK  Output root (default: $SCRIPT_DIR/outputs). Must contain
#                  cc_react/$P/results_$P.json from a prior cc-react run.
#   KISS_PY        Python interpreter of the kiss venv (default:
#                  /home/ubuntu/kiss/.venv/bin/python).
#   MAX_BUDGET     Kiss agent max-budget in dollars (default: 5.0).
#   MAX_STEPS      Kiss agent max-steps (default: 30).
# Requires ANTHROPIC_API_KEY exported (or in ~/.bashrc).

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACC_REPO="${ACC_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORK="${ABLATION_WORK:-$SCRIPT_DIR/outputs}"
KISS_PY="${KISS_PY:-/home/ubuntu/kiss/.venv/bin/python}"
MAX_BUDGET="${MAX_BUDGET:-5.0}"
MAX_STEPS="${MAX_STEPS:-30}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  eval "$(grep -E '^export ANTHROPIC_API_KEY=' ~/.bashrc 2>/dev/null || true)"
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"

export ACC_REPO
mkdir -p "$WORK/kiss"

PROBLEMS=("alltoallv:moe" "uniform_a2a:moe" "ring_kv:moe" "dxe:moe"
          "pp_send_recv:moe" "tp_mlp:moe"
          "fsdp_prefetch:moe" "llama_block_ar:moe")

echo "[kiss-sorcar-react ablation: 8 problems, sonnet, 1-node]"
echo "  ACC_REPO=$ACC_REPO"
echo "  WORK=$WORK  KISS_PY=$KISS_PY"
date -Is

for entry in "${PROBLEMS[@]}"; do
  P="${entry%:*}"; PAT="${entry#*:}"
  OUT="$WORK/kiss/$P"
  mkdir -p "$OUT"
  TLOG="$OUT/tokens.jsonl"
  rm -f "$TLOG"

  # Compute per-problem ±5% target from cc-react's saved results.
  RJ="$WORK/cc_react/$P/results_$P.json"
  TARG_ARG=""
  if [ -s "$RJ" ]; then
    BEST=$(python3 - <<PY
import json
d = json.load(open("$RJ"))
vals = []
def walk(x):
    if isinstance(x, dict):
        if 'sim_time_us' in x:
            vals.append(x['sim_time_us'])
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)
walk(d)
print(min(vals) if vals else "")
PY
)
    if [ -n "$BEST" ]; then
      LIMIT=$(python3 -c "print(float('$BEST')*1.05)")
      TARG_ARG="--target-score $LIMIT"
      echo
      echo "== $P (pattern=$PAT) cc_best=$BEST target=$LIMIT =="
    else
      echo
      echo "== $P (pattern=$PAT) [no cc target] =="
    fi
  else
    echo
    echo "== $P (pattern=$PAT) [no cc_react result yet] =="
  fi

  START=$(date +%s)
  # shellcheck disable=SC2086  # deliberate splitting of $TARG_ARG
  ABLATION_TOKEN_LOG="$TLOG" \
    "$KISS_PY" "$SCRIPT_DIR/kiss_phase3.py" \
      --problem "$P" --pattern "$PAT" \
      --output-dir "$OUT" \
      --max-budget "$MAX_BUDGET" --max-steps "$MAX_STEPS" \
      $TARG_ARG 2>&1 \
      | tee "$OUT/run.log" | tail -20
  END=$(date +%s)
  WALL=$((END - START))
  echo "  wall_seconds=$WALL"
  if [ -s "$TLOG" ]; then
    python - <<PY
import json
ts = [json.loads(l) for l in open("$TLOG")]
ti = sum(t['input_tokens'] for t in ts)
to = sum(t['output_tokens'] for t in ts)
cr = sum(t['cache_read_input_tokens'] for t in ts)
cc = sum(t['cache_creation_input_tokens'] for t in ts)
cost = (ti/1e6)*3.0 + (to/1e6)*15.0 + (cr/1e6)*0.3 + (cc/1e6)*3.75
print(f"  tokens: in={ti:,} out={to:,} cache_read={cr:,} cache_create={cc:,}")
print(f"  sonnet 4.5 cost \$={cost:.4f}  api_calls={len(ts)}")
PY
  fi
done

echo
echo "[done $(date -Is)]"
