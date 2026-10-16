#!/bin/bash
# kill the two synth/real/mutation capture drivers at the deadline so the chain proceeds (collective -> acc -> manifest -> bench)
sleep ${1:-6000}
pkill -f "scripts/capture.py --grids synth,real,mutatio[n]"; sleep 2; pkill -9 -f "scripts/capture_one.p[y]"
echo "### [$(date -u +%T)] TIMEBOX: capture drivers killed" >> /home/ubuntu/results/capture3.log
