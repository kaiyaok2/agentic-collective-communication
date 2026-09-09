#!/bin/bash
# cc-react baseline driver: runs the 8-problem Phase-3 ablation.
#
# Environment (all optional; sensible defaults):
#   ACC_REPO       Path to the acc repo (default: two dirs up from this file).
#   ABLATION_WORK  Output root (default: $SCRIPT_DIR/outputs).
#   NEURON_VENV    Neuron venv root (default: /opt/aws_neuronx_venv_pytorch_2_9).
#   MAX_ROUNDS     cc-react --max-rounds (default: 8).
# Requires ANTHROPIC_API_KEY exported (or in ~/.bashrc).

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACC_REPO="${ACC_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORK="${ABLATION_WORK:-$SCRIPT_DIR/outputs}"
NEURON_VENV="${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"

# Pick up ANTHROPIC_API_KEY from ~/.bashrc if not already exported.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  eval "$(grep -E '^export ANTHROPIC_API_KEY=' ~/.bashrc 2>/dev/null || true)"
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"

# shellcheck disable=SC1091
source "$NEURON_VENV/bin/activate"
export PYTHONUNBUFFERED=1
export ACC_REPO

mkdir -p "$WORK/cc_react"
cd "$ACC_REPO"

PROBLEMS=("alltoallv:moe" "uniform_a2a:moe" "ring_kv:moe" "dxe:moe"
          "pp_send_recv:moe" "tp_mlp:moe"
          "fsdp_prefetch:moe" "llama_block_ar:moe")

echo "[cc-react ablation: 8 problems, sonnet, max-rounds $MAX_ROUNDS, 1-node]"
echo "  ACC_REPO=$ACC_REPO"
echo "  WORK=$WORK"
date -Is

for entry in "${PROBLEMS[@]}"; do
  P="${entry%:*}"; PAT="${entry#*:}"
  OUT="$WORK/cc_react/$P"
  mkdir -p "$OUT"
  TLOG="$OUT/tokens.jsonl"
  rm -f "$TLOG"
  echo
  echo "== $P (pattern=$PAT) =="
  START=$(date +%s)
  TOKEN_LOG="$TLOG" SEARCH_OUTPUT_DIR="$OUT" SEARCH_TAG="cc_react_$P" \
    python "$SCRIPT_DIR/run_with_route_patch.py" \
      --problem "$P" --pattern "$PAT" \
      --phase3-style cc-react --llm-model sonnet --num-nodes 1 \
      --max-rounds "$MAX_ROUNDS" --output-dir "$OUT" 2>&1 \
      | tee "$OUT/run.log" \
      | grep -E "Simulator winner|cc:final|Phase 3|Generated" | tail -10
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
# Sonnet 4.5 pricing per 1M tokens: input \$3, output \$15,
# cache_read \$0.30, cache_create \$3.75.
cost = (ti/1e6)*3.0 + (to/1e6)*15.0 + (cr/1e6)*0.3 + (cc/1e6)*3.75
print(f"  tokens: in={ti:,} out={to:,} cache_read={cr:,} cache_create={cc:,}")
print(f"  sonnet 4.5 cost \$={cost:.4f}  api_calls={len(ts)}")
PY
  fi
done

echo
echo "[done $(date -Is)]"
