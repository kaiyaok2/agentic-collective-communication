#!/bin/bash
# Purchase the EARLIEST 7x trn1.32xlarge 24h Capacity Block in ap-southeast-2 (Sydney).
# Usage: bash /tmp/ap7/purchase.sh            -> earliest (immediate start, ~27.5h, ~$1,838)
#        bash /tmp/ap7/purchase.sh 1          -> second-earliest (11:30 UTC start, 24h, $1,601)
#        REGION=ap-south-1 bash /tmp/ap7/purchase.sh   -> Mumbai instead
set -euo pipefail
export AWS_PAGER=""
REGION=${REGION:-ap-southeast-2}
IDX=${1:-0}
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
END=$(date -u -v+3d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '+3 days' +%Y-%m-%dT%H:%M:%SZ)

echo "offerings in $REGION:"
aws ec2 describe-capacity-block-offerings --region "$REGION" \
  --instance-type trn1.32xlarge --instance-count 7 --capacity-duration-hours 24 \
  --start-date-range "$START" --end-date-range "$END" \
  --query 'sort_by(CapacityBlockOfferings,&StartDate)[].[CapacityBlockOfferingId,AvailabilityZone,StartDate,EndDate,UpfrontFee]' \
  --output text

OID=$(aws ec2 describe-capacity-block-offerings --region "$REGION" \
  --instance-type trn1.32xlarge --instance-count 7 --capacity-duration-hours 24 \
  --start-date-range "$START" --end-date-range "$END" \
  --query "sort_by(CapacityBlockOfferings,&StartDate)[$IDX].CapacityBlockOfferingId" \
  --output text)
echo "purchasing offering $OID ..."
aws ec2 purchase-capacity-block --region "$REGION" \
  --capacity-block-offering-id "$OID" --instance-platform Linux/UNIX \
  --query 'CapacityReservation.[CapacityReservationId,State,StartDate,EndDate,AvailabilityZone,TotalInstanceCount]' \
  --output text | tee /tmp/ap7/cr.txt
echo "saved to /tmp/ap7/cr.txt"
