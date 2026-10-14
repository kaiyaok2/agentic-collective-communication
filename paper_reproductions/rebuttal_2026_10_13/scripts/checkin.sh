#!/bin/bash
# checkin.sh <minutes>: sleep, then pull results and print sweep progress + a quick analysis
sleep $(( ${1:-30} * 60 ))
echo "=== $(date -u +%T) check-in ==="
bash /tmp/ap7/pull_results.sh 2>/dev/null | tail -1
bash /tmp/ap7/m.sh "grep -E '^---|steady_ms=|NO RESULT|^###|rc=' /home/ubuntu/sweep_results/sweep.log /home/ubuntu/queue.log 2>/dev/null | grep -vE '^/home/ubuntu/queue.log:  rc' | tail -24; echo '--- done files:'; ls /home/ubuntu/sweep_results/*.json 2>/dev/null | wc -l" </dev/null 2>/dev/null | grep -v "Permanently added"
echo "--- analysis so far:"; python3 /tmp/ap7/analyze_sweep.py /tmp/ap7/pull/sweep_results 2>&1 | head -40
