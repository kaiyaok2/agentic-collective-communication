#!/bin/bash
sleep $(( ${1:-30} * 60 ))
echo "=== $(date -u +%T) check-in ==="
bash /tmp/ap7/pull_results.sh 2>/dev/null | tail -1
bash /tmp/ap7/m.sh "grep -E '^###|^=== \[|  rc=|Steady step|Final loss|\[bench\]|measured_ms=|NO RESULT|pp512' /home/ubuntu/queue.log | tail -28" </dev/null 2>/dev/null | grep -v "Permanently added"
echo "analysis so far"
