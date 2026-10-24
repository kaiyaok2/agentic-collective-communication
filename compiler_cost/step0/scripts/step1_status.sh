#!/bin/bash
# progress of the Step 1 campaign
R=/home/ubuntu/results; C=/home/ubuntu/corpus
echo "$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvailable/{printf "%.1fG",$2/1e6}' /proc/meminfo)"
for g in sweep__ synth_norm synth_layout synth_slice synth_mask synth_scatter synth_conv2 synth_einsum synth_rnn synth_logits synth_cast acc_cand__; do
  n=$(ls -d $C/${g}* 2>/dev/null | wc -l); d=$(ls $C/${g}*/meta.json 2>/dev/null | wc -l); f=$(grep -l '"ok": false' $C/${g}*/meta.json 2>/dev/null | wc -l)
  [ $n -gt 0 ] && printf "  %-14s claimed=%-4s done=%-4s failed=%s\n" $g $n $d $f
done
tail -3 $R/step1.log 2>/dev/null; for f in $R/bench_step1_default_s*.jsonl $R/bench_versions_full_s*.jsonl $R/bench_flags_full_s*.jsonl; do [ -f $f ] && echo "  $(basename $f): $(wc -l < $f) rows"; done
tail -1 $R/capture_step1_a.log 2>/dev/null | cut -c1-150; tail -1 $R/capture_step1_b.log 2>/dev/null | cut -c1-150; tail -1 $R/capture_step1_acc.log 2>/dev/null | cut -c1-150
