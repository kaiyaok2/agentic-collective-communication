#!/bin/bash
# exit when a new '###' stage line appears in master queue.log (or after MAXMIN minutes)
MAXMIN=${1:-120}
base=$(bash /tmp/ap7/m.sh "grep -c '^###' /home/ubuntu/queue.log" </dev/null 2>/dev/null | tr -dc 0-9); base=${base:-0}
for i in $(seq 1 $MAXMIN); do sleep 60; n=$(bash /tmp/ap7/m.sh "grep -c '^###' /home/ubuntu/queue.log" </dev/null 2>/dev/null | tr -dc 0-9); n=${n:-0}; [ "$n" -gt "$base" ] && break; done
echo "=== $(date -u +%T) queue.log tail ==="
bash /tmp/ap7/m.sh "grep -E 'rc=|\[bench\]|^###|^=== \[|NO RESULT|steady_ms' /home/ubuntu/queue.log | tail -20; echo '--- llama runs.log:'; cat /home/ubuntu/llama7b_results/runs.log 2>/dev/null" </dev/null 2>/dev/null | grep -v "Permanently added"
