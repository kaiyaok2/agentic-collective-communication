#!/bin/bash
# Re-run the 4 Llama-side problems (pp_send_recv, tp_mlp, fsdp_prefetch,
# llama_block_ar) for both cc-react and kiss with `--pattern moe`. The
# original ablation used the (invalid) `--pattern llama` for these problems
# and aborted; run_search.py's --pattern options are moe / uniform / skewed
# / sparse / random / increasing / locality only. The Llama-side problems
# work fine under `moe` send-count generation.
#
# Environment: same defaults as run_cc_baseline.sh and run_kiss_baseline.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACC_REPO="${ACC_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORK="${ABLATION_WORK:-$SCRIPT_DIR/outputs}"
NEURON_VENV="${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
KISS_PY="${KISS_PY:-/home/ubuntu/kiss/.venv/bin/python}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_BUDGET="${MAX_BUDGET:-5.0}"
MAX_STEPS="${MAX_STEPS:-30}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  eval "$(grep -E '^export ANTHROPIC_API_KEY=' ~/.bashrc 2>/dev/null || true)"
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"

# shellcheck disable=SC1091
source "$NEURON_VENV/bin/activate"
export PYTHONUNBUFFERED=1
export ACC_REPO

PROBLEMS=("pp_send_recv" "tp_mlp" "fsdp_prefetch" "llama_block_ar")

cd "$ACC_REPO"

echo "[CC-REACT Llama retry]"
date -Is
for P in "${PROBLEMS[@]}"; do
  OUT="$WORK/cc_react/$P"
  mkdir -p "$OUT"
  TLOG="$OUT/tokens.jsonl"
  rm -f "$TLOG" "$OUT/run.log"
  echo
  echo "== cc $P =="
  START=$(date +%s)
  TOKEN_LOG="$TLOG" SEARCH_OUTPUT_DIR="$OUT" SEARCH_TAG="cc_react_$P" \
    python "$SCRIPT_DIR/run_with_route_patch.py" \
      --problem "$P" --pattern moe \
      --phase3-style cc-react --llm-model sonnet --num-nodes 1 \
      --max-rounds "$MAX_ROUNDS" --output-dir "$OUT" 2>&1 \
      | tee "$OUT/run.log" \
      | grep -E "Simulator winner|cc:final|Generated" | tail -5
  END=$(date +%s); WALL=$((END - START))
  echo "  wall_seconds=$WALL"
  if [ -s "$TLOG" ]; then
    python - <<PY
import json
ts = [json.loads(l) for l in open("$TLOG")]
ti = sum(t['input_tokens'] for t in ts)
to = sum(t['output_tokens'] for t in ts)
cr = sum(t['cache_read_input_tokens'] for t in ts)
cc = sum(t['cache_creation_input_tokens'] for t in ts)
cost = (ti/1e6)*3 + (to/1e6)*15 + (cr/1e6)*0.3 + (cc/1e6)*3.75
print(f"  tokens: in={ti:,} out={to:,} cache_read={cr:,} cache_create={cc:,}")
print(f"  cost \$={cost:.4f}  api_calls={len(ts)}")
PY
  fi
done

echo
echo "[KISS Llama retry]"
date -Is
for P in "${PROBLEMS[@]}"; do
  OUT="$WORK/kiss/$P"
  mkdir -p "$OUT"
  TLOG="$OUT/tokens.jsonl"
  rm -f "$TLOG" "$OUT/run.log"

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
    fi
  fi

  echo
  echo "== kiss $P (target $TARG_ARG) =="
  START=$(date +%s)
  # shellcheck disable=SC2086
  ABLATION_TOKEN_LOG="$TLOG" \
    "$KISS_PY" "$SCRIPT_DIR/kiss_phase3.py" \
      --problem "$P" --pattern moe \
      --output-dir "$OUT" --max-budget "$MAX_BUDGET" \
      --max-steps "$MAX_STEPS" $TARG_ARG 2>&1 \
      | tee "$OUT/run.log" | tail -20
  END=$(date +%s); WALL=$((END - START))
  echo "  wall_seconds=$WALL"
  if [ -s "$TLOG" ]; then
    python - <<PY
import json
ts = [json.loads(l) for l in open("$TLOG")]
ti = sum(t['input_tokens'] for t in ts)
to = sum(t['output_tokens'] for t in ts)
cr = sum(t['cache_read_input_tokens'] for t in ts)
cc = sum(t['cache_creation_input_tokens'] for t in ts)
cost = (ti/1e6)*3 + (to/1e6)*15 + (cr/1e6)*0.3 + (cc/1e6)*3.75
print(f"  tokens: in={ti:,} out={to:,} cache_read={cr:,} cache_create={cc:,}")
print(f"  cost \$={cost:.4f}  api_calls={len(ts)}")
PY
  fi
done

echo
echo "[done $(date -Is)]"
