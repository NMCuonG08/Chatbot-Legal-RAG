#!/bin/bash
# ============================================================================
# infra_status.sh — one-shot inventory of everything terraform created
# ----------------------------------------------------------------------------
# Run from the terraform dir (or anywhere with AWS creds + this repo). Answers
# "what is terraform managing right now + did the console drift + what's
# running + what would it cost to forget". Pairs with `terraform destroy` for
# clean teardown (destroy removes EVERYTHING in state — nothing forgotten).
#
# Usage:  ./deployment/scripts/infra_status.sh
# Requires: terraform, aws CLI (configured), jq (optional, for pretty JSON).
# ============================================================================
set -uo pipefail

TF_DIR="$(cd "$(dirname "$0")/../terraform" && pwd)"
REGION="${AWS_REGION:-ap-southeast-1}"
TAG_KEY="Name"
TAG_PREFIX="legal"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
hr()   { printf "\n%s\n" "----------------------------------------"; }

hr; bold "1) Terraform-managed resources (state list)"
if [ -f "$TF_DIR/terraform.tfstate" ] || [ -d "$TF_DIR/.terraform" ]; then
  ( cd "$TF_DIR" && terraform state list 2>/dev/null ) || echo "  (terraform not init'd — run: cd $TF_DIR && terraform init)"
else
  echo "  (no state yet — terraform not applied)"
fi

hr; bold "2) Drift check (terraform plan -detailed-exitcode)"
# exit 0 = clean, 2 = drift, 1 = error. We only report, never fail the script.
( cd "$TF_DIR" && terraform plan -detailed-exitcode -input=false -lock=false -no-color >/tmp/tfplan.txt 2>&1 )
case $? in
  0) echo "  CLEAN — state matches AWS." ;;
  2) echo "  DRIFT — AWS console changed something. Diff:"; sed -n '1,40p' /tmp/tfplan.txt ;;
  *) echo "  (plan skipped — terraform not init'd or no creds)" ;;
esac

hr; bold "3) All AWS resources tagged $TAG_KEY=$TAG_PREFIX-* (catch orphans)"
aws resourcegroupstaggingapi get-resources \
  --tag-filters "Key=$TAG_KEY,Values=$TAG_PREFIX" \
  --region "$REGION" --output text --query 'ResourceTagMappingList[].{arn:ResourceARN,name:ResourceTagMappingList[?Key==`Name`].Value|[0]}' \
  2>/dev/null | sed 's/^/  /' || echo "  (aws CLI not configured)"

hr; bold "4) Running EC2 instances (cost accumulates while up)"
aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:$TAG_KEY,Values=$TAG_PREFIX*" "Name=instance-state-name,Values=running,pending,stopping,shutting-down" \
  --query 'Reservations[].Instances[].{id:InstanceId,state:State.Name,type:InstanceType,ip:PublicIpAddress,name:Tags[?Key==`Name`].Value|[0]}' \
  --output table 2>/dev/null || echo "  (aws CLI not configured)"

hr; bold "5) S3 buckets owned by this stack"
aws s3api list-buckets --region "$REGION" \
  --query "Buckets[?starts_with(Name,\`$TAG_PREFIX\`)].Name" --output text 2>/dev/null \
  | tr '\t' '\n' | sed 's/^/  /' || echo "  (aws CLI not configured)"

hr; bold "6) Elastic IPs (charged even when unattached!)"
aws ec2 describe-addresses --region "$REGION" \
  --filters "Name=tag:$TAG_KEY,Values=$TAG_PREFIX*" \
  --query 'Addresses[].{ip:PublicIp,inst:InstanceId,state:AssociationId}' --output table 2>/dev/null \
  || echo "  (aws CLI not configured)"

hr; bold "7) Cost to date (this month, by service)"
START="$(date -u -d 'first day of this month' +%Y-%m-%d 2>/dev/null || date -u +%Y-%m-01)"
END="$(date -u +%Y-%m-%d)"
aws ce get-cost-and-usage --region "$REGION" \
  --time-period Start="$START",End="$END" \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[].Groups[?Metrics.UnblendedCost.Amount>`0`].{svc:Keys[0],usd:Metrics.UnblendedCost.Amount}' \
  --output table 2>/dev/null || echo "  (Cost Explorer not enabled or no data — enable in AWS Billing console)"

hr; bold "Teardown (cost-conscious: removes EVERYTHING in state, no orphans)"
echo "  cd $TF_DIR && terraform destroy -var-file=terraform.tfvars"
echo "  Verify empty after:  ./deployment/scripts/infra_status.sh  (sections 3-6 should be empty)"
echo