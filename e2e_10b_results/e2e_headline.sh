#!/bin/bash
# E2E 10B-TP HEADLINE run on the live 7-node cluster: full L=48, N_MB=16,
# fused ZeRO-1, 12 steps (warmup 3), all 3 backends on real wikitext-103.
# Target: reproduce >=2x sorcar/strat with baseline==strat bit-identical loss.
# Runs ON THE MASTER (e2e_launch.sh SSHes the 6 workers).
#   Usage:  bash e2e_headline.sh [llama|gpt] [nmb]   (defaults: llama 16)
set -u
ARCH=${1:-llama}; NMB=${2:-16}
case "$ARCH" in
  llama) SCRIPT=/home/ubuntu/train_llama10b_tp_families.py ;;
  gpt)   SCRIPT=/home/ubuntu/train_gpt10b_tp_families.py ;;
  *) echo "unknown arch: $ARCH"; exit 2 ;;
esac
[ -f "$SCRIPT" ] || { echo "MISSING $SCRIPT"; exit 2; }
[ -f /home/ubuntu/wiki.train.raw ] || { echo "MISSING wiki.train.raw"; exit 2; }
LAUNCH=/home/ubuntu/e2e_launch.sh
CACHE=${CACHE:-/home/ubuntu/neuron_cache_e2e_headline}
OUT=/home/ubuntu/e2e_headline_out
mkdir -p "$OUT" "$CACHE"
COMMON="--nmb $NMB --warmup 3 --fuse"   # L defaults to 48 in the script

declare -A MED
declare -A FL
for BK in baseline strat sorcar; do
  echo "=== HEADLINE $ARCH nmb=$NMB backend=$BK $(date -u) ==="
  STEPS=${STEPS:-12} CACHE="$CACHE" RUN_TIMEOUT=7200 bash "$LAUNCH" "$SCRIPT" "$BK" \
    "hl_${ARCH}_nmb${NMB}_${BK}" $COMMON 2>&1 | tee "$OUT/${ARCH}_nmb${NMB}_${BK}.log"
  RJ=$(grep -h "RESULT_JSON" "$OUT/${ARCH}_nmb${NMB}_${BK}.log" | tail -1 | sed 's/^.*RESULT_JSON //')
  if [ -n "$RJ" ]; then
    MED[$BK]=$(echo "$RJ" | python3 -c "import sys,json;print(json.load(sys.stdin)['median_ms_per_step'])")
    FL[$BK]=$(echo "$RJ" | python3 -c "import sys,json;print(repr(json.load(sys.stdin)['final_loss']))")
    echo "  -> $BK median_ms=${MED[$BK]} final_loss=${FL[$BK]}"
  else
    MED[$BK]="FAIL"; FL[$BK]="FAIL"
    echo "  -> $BK NO RESULT_JSON (see $OUT/${ARCH}_nmb${NMB}_${BK}.log)"
  fi
done

echo "=== HEADLINE SUMMARY ($ARCH nmb=$NMB) ==="
for BK in baseline strat sorcar; do echo "  $BK: ${MED[$BK]:-?} ms  final_loss=${FL[$BK]:-?}"; done
if [ "${MED[sorcar]:-FAIL}" != "FAIL" ] && [ "${MED[baseline]:-FAIL}" != "FAIL" ]; then
  python3 - <<PY
b=${MED[baseline]}; s=${MED[strat]:-$b}; k=${MED[sorcar]}
print("  sorcar/baseline = %.3fx" % (b/k))
print("  sorcar/strat    = %.3fx" % (s/k))
print("  strat/baseline  = %.3fx" % (s/b))
print("  >=2x OVER STRAT: %s" % ("YES" if s/k>=2.0 else "NO (%.3fx)"%(s/k)))
PY
  echo "  loss parity: baseline=${FL[baseline]} strat=${FL[strat]} sorcar=${FL[sorcar]}"
fi
echo "E2E_HEADLINE_DONE ($ARCH nmb=$NMB)"
