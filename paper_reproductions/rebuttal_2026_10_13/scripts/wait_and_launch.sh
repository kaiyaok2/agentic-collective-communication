#!/bin/bash
export AWS_PAGER=""
CR=cr-0e125b5367e87c67f
for i in $(seq 1 90); do
  ST=$(aws ec2 describe-capacity-reservations --region ap-southeast-2 --capacity-reservation-ids $CR --query 'CapacityReservations[0].State' --output text 2>/dev/null)
  echo "$(date -u +%T) state=$ST"
  case "$ST" in *active*) break;; esac
  sleep 20
done
case "$ST" in *active*) ;; *) echo "CB not active after wait; state=$ST"; exit 1;; esac
sleep 30
bash /tmp/ap7/launch_ap7.sh $CR
