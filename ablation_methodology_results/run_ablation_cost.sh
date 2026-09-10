#!/usr/bin/env bash
# Faithful ablation: measure search-controller cost with the research-discovery
# loop + adversarial-testing methodology ENABLED (full) vs DISABLED (ablated).
# Same problems, same scorer, same model; only the KISSAgent SYSTEM.md differs.
set -u
KP=/home/ubuntu/kiss/src/kiss
KISS_PY=/home/ubuntu/kiss/.venv/bin/python
export ANTHROPIC_API_KEY="$(cat /home/ubuntu/.anthropic_env_val)"
export ACC_REPO=/home/ubuntu/acc
export NEURON_PY=/opt/aws_neuronx_venv_pytorch_2_8/bin/python
export KISS_HOME=/home/ubuntu/.kiss
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
MAXSTEPS="${MAXSTEPS:-30}"; MAXBUDGET="${MAXBUDGET:-5.0}"; PAR="${PAR:-8}"
PROBS=(sixtyfourinline eightyaltsum nine_ar_same_input_chal sequential_ar_chain_edge_chal perslice3dM96 perrowM64N4K mixmaxmin eightslab)

run_arm() {
  local arm="$1" sysfile="$2"
  local OUT=/home/ubuntu/ablation_cost/$arm
  mkdir -p "$OUT" "$KISS_HOME"
  cp "$sysfile" "$KP/SYSTEM.md"
  echo "[arm $arm] SYSTEM.md <- $sysfile ($(wc -l < $KP/SYSTEM.md) lines)"
  run_one() {
    local nm="$1" arm="$2" OUT="$3"
    [ -f "$OUT/$nm/kiss_summary.json" ] && { echo "SKIP $arm/$nm"; return 0; }
    cd /home/ubuntu/acc/experiments/ablation_kiss_vs_cc
    ABLATION_TOKEN_LOG="$OUT/$nm/tokens.jsonl" \
    PYTHONPATH=/home/ubuntu/acc \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
    /home/ubuntu/kiss/.venv/bin/python sorcar_phase3_7node.py \
      --problem "$nm" --pattern moe --output-dir "$OUT/$nm" \
      --max-steps "$MAXSTEPS" --max-budget "$MAXBUDGET" \
      > "$OUT/$nm.log" 2>&1
    echo "done $arm/$nm rc=$?"
  }
  export -f run_one; export MAXSTEPS MAXBUDGET
  printf "%s\n" "${PROBS[@]}" | xargs -I{} -P "$PAR" bash -c "run_one {} $arm $OUT"
  echo "[arm $arm] COMPLETE"
}

run_arm full    "$KP/SYSTEM_FULL.md"
run_arm ablated "$KP/SYSTEM_ABLATED.md"
# restore pristine
cp "$KP/SYSTEM_FULL.md" "$KP/SYSTEM.md"
echo "ALL ARMS COMPLETE; SYSTEM.md restored"
