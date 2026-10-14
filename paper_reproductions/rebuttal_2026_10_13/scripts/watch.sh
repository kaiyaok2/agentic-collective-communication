#!/bin/bash
# watch.sh <max_minutes> : poll master queue.log every 60s; exit when a run finishes (new 'rc=' line) or a new stage starts.
MAXMIN=${1:-90}
base=$(bash /tmp/ap7/m.sh "grep -cE 'rc=|^###' /home/ubuntu/queue.log" </dev/null 2>/dev/null | tr -dc 0-9); base=${base:-0}
for i in $(seq 1 $MAXMIN); do
  sleep 60
  n=$(bash /tmp/ap7/m.sh "grep -cE 'rc=|^###' /home/ubuntu/queue.log" </dev/null 2>/dev/null | tr -dc 0-9); n=${n:-0}
  [ "$n" -gt "$base" ] && break
done
echo "=== $(date -u +%T) queue.log tail ==="
bash /tmp/ap7/m.sh "grep -vE 'Permanently added|^\s*$' /home/ubuntu/queue.log | tail -14; echo '--- sim scoring:'; tail -3 /home/ubuntu/sweep_results_sim.log 2>/dev/null; echo '--- llama runs.log:'; cat /home/ubuntu/llama7b_results/runs.log 2>/dev/null" </dev/null 2>/dev/null | grep -v "Permanently added"
