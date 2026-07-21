#!/bin/bash
# ============================================================================
# Legal Chatbot — nightly backup to S3
# ----------------------------------------------------------------------------
# Run on the EC2 host (crontab installed by user_data.sh). Dumps MariaDB +
# tars Qdrant/Redis/Grafana data, uploads to S3_BACKUP_BUCKET, prunes local
# copies. Uses the EC2 instance role for S3 creds (no static AWS keys on disk)
# — see aws_iam_role_policy.legal_s3_backup in main.tf.
#
# Schedule (user_data.sh): 17 3 * * *  /home/ubuntu/app/deployment/scripts/backup.sh
# Manual:  sudo -u ubuntu /home/ubuntu/app/deployment/scripts/backup.sh
# ============================================================================
set -euo pipefail

APP_DIR="/home/ubuntu/app"
ENV_FILE="$APP_DIR/backend/.env"
DATA_DIR="$APP_DIR/data"
LOCAL_BK="$DATA_DIR/backups"

# --- Load env (MYSQL creds, S3 bucket, region) ------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "[backup] ERROR: $ENV_FILE missing — fill backend/.env first." >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

: "${MYSQL_PASSWORD:?MYSQL_PASSWORD missing in backend/.env}"
: "${S3_BACKUP_BUCKET:?S3_BACKUP_BUCKET missing in backend/.env}"
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
MYSQL_USER="${MYSQL_USER:-legal}"
MYSQL_DB="${MYSQL_DB:-legal_db}"
MARIADB_CONTAINER="${MARIADB_CONTAINER:-mariadb}"

TS="$(date -u +%Y%m%d-%H%M%S)"
BK="$LOCAL_BK/$TS"
mkdir -p "$BK"

echo "[backup] $TS -> s3://$S3_BACKUP_BUCKET/$TS/"

# --- 1. MariaDB logical dump (via docker exec to avoid installing client) ----
echo "[backup] dumping MariaDB..."
docker exec "$MARIADB_CONTAINER" \
  mysqldump -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" --single-transaction \
  --routines --triggers "$MYSQL_DB" > "$BK/mariadb.sql"

# --- 2. Volume data (Qdrant + Redis + Grafana) ------------------------------
echo "[backup] tarring volume data..."
tar czf "$BK/data.tar.gz" -C "$DATA_DIR" qdrant redis grafana 2>/dev/null || {
  echo "[backup] WARN: some data dirs missing — partial tar." >&2
}

# --- 3. Upload to S3 (instance role) ----------------------------------------
echo "[backup] uploading to S3..."
aws s3 cp "$BK/mariadb.sql" "s3://$S3_BACKUP_BUCKET/$TS/mariadb.sql" \
  --region "$AWS_REGION" --no-progress
aws s3 cp "$BK/data.tar.gz" "s3://$S3_BACKUP_BUCKET/$TS/data.tar.gz" \
  --region "$AWS_REGION" --no-progress

# --- 4. Prune local copies > 7 days -----------------------------------------
find "$LOCAL_BK" -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {} +

# --- 5. S3 lifecycle handles remote retention (set bucket lifecycle:        ---
# ---    noncurrent + delete-after 30d). Documented in SECRETS_CHECKLIST.    ---
echo "[backup] done. local: $BK  remote: s3://$S3_BACKUP_BUCKET/$TS/"