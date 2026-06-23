#!/usr/bin/env bash
# create_gke_tpu_cluster.sh
# This script creates a custom VPC network, custom subnet, internal firewall rules,
# a standard GKE regional cluster, a single-host TPU v6e-8 node pool with Flex-start,
# and the results telemetry Cloud Storage bucket.

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================
PROJECT_ID="<your-gcp-project-id>"
CLUSTER_NAME="<cluster-name>"
REGION="<region>"
ZONE="<zone>"
NODE_POOL_NAME="<node-pool-name>"
TPU_MACHINE_TYPE="ct6e-standard-8t"
RESULTS_BUCKET="<results-bucket>"

# Networking parameters
NETWORK_NAME="<network-name>"
SUBNET_NAME="<subnet-name>"
FIREWALL_NAME="<firewall-name>"
PRIMARY_RANGE="172.25.0.0/16" # Primary IP address block for GKE nodes

echo "=============================================================================="
echo "1. Ensuring Custom VPC Network exists: ${NETWORK_NAME}"
echo "=============================================================================="
if ! gcloud compute networks describe "${NETWORK_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud compute networks create "${NETWORK_NAME}" \
        --project="${PROJECT_ID}" \
        --subnet-mode=custom
    echo "Successfully created custom VPC network: ${NETWORK_NAME}"
else
    echo "Custom VPC network ${NETWORK_NAME} already exists."
fi

echo "=============================================================================="
echo "2. Ensuring Custom Subnet exists: ${SUBNET_NAME}"
echo "=============================================================================="
if ! gcloud compute networks subnets describe "${SUBNET_NAME}" --project="${PROJECT_ID}" --region="${REGION}" &>/dev/null; then
    gcloud compute networks subnets create "${SUBNET_NAME}" \
        --project="${PROJECT_ID}" \
        --network="${NETWORK_NAME}" \
        --region="${REGION}" \
        --range="${PRIMARY_RANGE}" \
        --enable-private-ip-google-access
    echo "Successfully created custom subnet: ${SUBNET_NAME}"
else
    echo "Custom subnet ${SUBNET_NAME} already exists."
fi

echo "=============================================================================="
echo "3. Ensuring Internal Firewall Rules exist: ${FIREWALL_NAME}"
echo "=============================================================================="
if ! gcloud compute firewall-rules describe "${FIREWALL_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud compute firewall-rules create "${FIREWALL_NAME}" \
        --project="${PROJECT_ID}" \
        --network="${NETWORK_NAME}" \
        --allow=tcp,udp,icmp \
        --source-ranges="${PRIMARY_RANGE}"
    echo "Successfully created internal firewall rule: ${FIREWALL_NAME}"
else
    echo "Firewall rule ${FIREWALL_NAME} already exists."
fi

echo "=============================================================================="
echo "4. Ensuring Standard GKE Regional Cluster exists: ${CLUSTER_NAME}"
echo "=============================================================================="
if ! gcloud container clusters describe "${CLUSTER_NAME}" --project="${PROJECT_ID}" --location="${REGION}" &>/dev/null; then
    gcloud container clusters create "${CLUSTER_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --network="${NETWORK_NAME}" \
        --subnetwork="${SUBNET_NAME}" \
        --release-channel="rapid" \
        --workload-pool="${PROJECT_ID}.svc.id.goog"
    echo "Successfully created GKE cluster: ${CLUSTER_NAME}"
else
    echo "GKE cluster ${CLUSTER_NAME} already exists."
fi

echo "=============================================================================="
echo "5. Ensuring Single-Host TPU v6e (Trillium) Node Pool exists: ${NODE_POOL_NAME}"
echo "=============================================================================="
if ! gcloud container node-pools describe "${NODE_POOL_NAME}" --project="${PROJECT_ID}" --cluster="${CLUSTER_NAME}" --location="${REGION}" &>/dev/null; then
    gcloud container node-pools create "${NODE_POOL_NAME}" \
        --project="${PROJECT_ID}" \
        --cluster="${CLUSTER_NAME}" \
        --location="${REGION}" \
        --node-locations="${ZONE}" \
        --machine-type="${TPU_MACHINE_TYPE}" \
        --num-nodes=0 \
        --enable-autoscaling \
        --min-nodes=0 \
        --max-nodes=2 \
        --reservation-affinity=none \
        --flex-start
    echo "Successfully created single-host TPU v6e node pool: ${NODE_POOL_NAME}"
else
    echo "TPU node pool ${NODE_POOL_NAME} already exists."
fi

echo "=============================================================================="
echo "6. Ensuring Telemetry Cloud Storage Bucket exists: gs://${RESULTS_BUCKET}"
echo "=============================================================================="
if ! gcloud storage buckets describe "gs://${RESULTS_BUCKET}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud storage buckets create "gs://${RESULTS_BUCKET}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --uniform-bucket-level-access
    echo "Successfully created Cloud Storage bucket: gs://${RESULTS_BUCKET}"
else
    echo "Cloud Storage bucket gs://${RESULTS_BUCKET} already exists."
fi

echo "=============================================================================="
echo "Network, Firewall, Cluster, TPU v6e node pool, and GCS Bucket verification complete!"
echo "Verify ready nodes using: kubectl get nodes -o wide"
echo "=============================================================================="
