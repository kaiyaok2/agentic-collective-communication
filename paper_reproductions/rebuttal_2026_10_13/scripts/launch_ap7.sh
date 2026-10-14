#!/bin/bash
# Launch 7x trn1.32xlarge into a Capacity Block in ap-southeast-2c (rebuttal cluster),
# 1 master + 6 workers, 8 EFA NICs each, 500G gp3 root, EIP on master card-0 ENI.
# Usage: bash launch_ap7.sh <cr-id> [ami-id]
set -euo pipefail
export AWS_PAGER=""
CR=${1:?capacity reservation id (cr-...)}
AMI=${2:-ami-0fac1ab96d92cae78}   # Deep Learning AMI Neuron (Ubuntu 24.04) 20260126, ap-southeast-2
REGION=ap-southeast-2
ITYPE=trn1.32xlarge
KEY=Kaiyao
EIP_ALLOC=eipalloc-07fe33b881fef0ffa    # <MASTER-EIP>
NIC=/tmp/nic_ap7.json
OUTF=/tmp/ap7/cluster.env

launch_one () {
  local name=$1
  aws ec2 run-instances --region $REGION \
    --image-id $AMI --instance-type $ITYPE --key-name $KEY --count 1 \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR}" \
    --instance-market-options 'MarketType=capacity-block' \
    --network-interfaces "file://$NIC" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
    --query 'Instances[0].InstanceId' --output text
}

echo "launching master..."; MASTER_ID=$(launch_one rebuttal-master)
WORKER_IDS=""
for i in 1 2 3 4 5 6; do
  echo "launching worker-$i..."; WORKER_IDS="$WORKER_IDS $(launch_one rebuttal-worker-$i)"
done
echo "instances: $MASTER_ID $WORKER_IDS"
for id in $MASTER_ID $WORKER_IDS; do aws ec2 wait instance-running --region $REGION --instance-ids "$id"; echo "  running: $id"; done

MENI=$(aws ec2 describe-instances --region $REGION --instance-ids "$MASTER_ID" \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId' --output text)
aws ec2 associate-address --region $REGION --allocation-id $EIP_ALLOC --network-interface-id "$MENI" --allow-reassociation >/dev/null
MASTER_PUB=$(aws ec2 describe-addresses --region $REGION --allocation-ids $EIP_ALLOC --query 'Addresses[0].PublicIp' --output text)

get_priv () { aws ec2 describe-instances --region $REGION --instance-ids "$1" \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].PrivateIpAddress' --output text; }
MASTER_PRIV=$(get_priv "$MASTER_ID")
WORKER_PRIVS=""; for id in $WORKER_IDS; do WORKER_PRIVS="$WORKER_PRIVS $(get_priv "$id")"; done
WORKER_PRIVS=$(echo $WORKER_PRIVS)

{
  echo "REGION=$REGION"
  echo "CR=$CR"
  echo "MASTER_ID=$MASTER_ID"
  echo "WORKER_IDS=\"$(echo $WORKER_IDS)\""
  echo "MASTER_PUB=$MASTER_PUB"
  echo "MASTER_PRIV=$MASTER_PRIV"
  echo "WORKER_PRIVS=\"$WORKER_PRIVS\""
  echo "LAUNCHED_UTC=$(date -u +%FT%TZ)"
} | tee "$OUTF"
echo "=== wrote $OUTF; ssh -i ~/.ssh/Kaiyao.pem ubuntu@$MASTER_PUB ==="
