#!/bin/bash
# Pre-armed: wait until the follow-on Capacity Block is active, then launch + bootstrap + start queue30 (transition_ap7.sh).
export AWS_PAGER=""
CR=cr-08c76b9a932768c9e
# the current block ends 11:30 UTC; the new one starts 11:30 UTC. Poll (up to ~5.5 h) for state=active.
for i in $(seq 1 1000); do
  ST=$(aws ec2 describe-capacity-reservations --region ap-southeast-2 --capacity-reservation-ids $CR --query 'CapacityReservations[0].[State,AvailableInstanceCount]' --output text 2>/dev/null)
  echo "$(date -u +%T) $ST"
  case "$ST" in active*) break;; esac
  sleep 20
done
case "$ST" in active*) ;; *) echo "CB not active after wait; state=$ST"; exit 1;; esac
sleep 45
bash /tmp/ap7/transition_ap7.sh $CR
