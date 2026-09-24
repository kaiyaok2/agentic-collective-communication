#!/bin/bash
# Launch 7x trn1.32xlarge into CB cr-0e709fd8e24812597 (us-east-1c), one master + 6 workers,
# each with 8 EFA NICs. Associate the EIP to the master's card-0 ENI. Prints IPs.
# Run once the CB is ACTIVE. Requires AWS_PROFILE=kaiyao. NIC config: nic_cb7.json (same dir).
set -euo pipefail
export AWS_PROFILE=kaiyao
REGION=us-east-1
CR=cr-0e709fd8e24812597
AMI=ami-0fd664467b3cf8dfd
ITYPE=trn1.32xlarge
KEY=Kaiyao
EIP_ALLOC=eipalloc-02bc3f30121513aa3
NIC=$(dirname "$0")/nic_cb7.json
OUTF=/tmp/warmrt_cluster.env

launch_one () {
  local name=$1
  aws ec2 run-instances --region $REGION \
    --image-id $AMI --instance-type $ITYPE --key-name $KEY --count 1 \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR}" \
    --instance-market-options 'MarketType=capacity-block' \
    --network-interfaces "file://$NIC" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
    --query 'Instances[0].InstanceId' --output text
}

echo "launching master..."; MASTER_ID=$(launch_one warmrt-master)
WORKER_IDS=()
for i in $(seq 1 6); do
  echo "launching worker-$i..."; WORKER_IDS+=("$(launch_one warmrt-worker-$i)")
done
ALL_IDS=("$MASTER_ID" "${WORKER_IDS[@]}")
echo "instances: ${ALL_IDS[*]}"

echo "waiting for running (each id separately)..."
for id in "${ALL_IDS[@]}"; do aws ec2 wait instance-running --region $REGION --instance-ids "$id"; echo "  running: $id"; done

# master card-0 ENI + associate EIP
MENI=$(aws ec2 describe-instances --region $REGION --instance-ids "$MASTER_ID" \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId' --output text)
echo "master card-0 ENI=$MENI"
aws ec2 associate-address --region $REGION --allocation-id $EIP_ALLOC --network-interface-id "$MENI" >/dev/null
MASTER_PUB=$(aws ec2 describe-addresses --region $REGION --allocation-ids $EIP_ALLOC --query 'Addresses[0].PublicIp' --output text)

get_priv () { aws ec2 describe-instances --region $REGION --instance-ids "$1" \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].PrivateIpAddress' --output text; }
MASTER_PRIV=$(get_priv "$MASTER_ID")
WORKER_PRIVS=(); for id in "${WORKER_IDS[@]}"; do WORKER_PRIVS+=("$(get_priv "$id")"); done

{
  echo "MASTER_ID=$MASTER_ID"
  echo "WORKER_IDS=\"${WORKER_IDS[*]}\""
  echo "MASTER_PUB=$MASTER_PUB"
  echo "MASTER_PRIV=$MASTER_PRIV"
  echo "WORKER_PRIVS=\"${WORKER_PRIVS[*]}\""
} | tee "$OUTF"
echo "=== wrote $OUTF; ssh: ssh -i ~/.ssh/Kaiyao.pem ubuntu@$MASTER_PUB ==="
