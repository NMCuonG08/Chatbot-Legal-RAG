# Secrets Checklist — Where to Get Every Key

> Mục đích: biết chính xác key nào lấy ở đâu, free/paid, scope quyền tối thiểu.
> Fill vào `backend/.env` (copy từ `.env.prod.aws.example`). KHÔNG commit file đã fill.
> Verify sau fill: `grep -i "PLACEHOLDER\|change-me\|your_" backend/.env` → phải 0 dòng.

---

## Tier A — Bắt buộc app chạy

| Key | Lấy ở đâu | Free | Scope tối thiểu | Dùng cho |
|---|---|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys | Có (quota) | default | LLM chính + judge |
| `JWT_SECRET` | `python -c "import secrets;print(secrets.token_urlsafe(32))"` | — | — | Mã hóa session |
| `SEED_ADMIN_PASSWORD` | Tự tạo mạnh ≥16 ký tự | — | — | Bootstrap admin |
| `DATABASE_URL` pass | Tự tạo mạnh (không "legal") | — | — | MariaDB |
| `MCP_API_KEY` | `secrets.token_urlsafe(32)` | — | — | Bearer MCP HTTP |
| `GF_SECURITY_ADMIN_PASSWORD` | Tự tạo mạnh | — | — | Grafana |

## Tier B — Nên có (chất lượng RAG)

| Key | Lấy ở đâu | Free | Scope tối thiểu | Dùng cho |
|---|---|---|---|---|
| `TAVILY_API_KEY` | https://app.tavily.com/api-key | 1k req/tháng | default | Web search tool |
| `COHERE_API_KEY` | https://dashboard.cohere.com/api-keys | Trial | rerank | Reranker (không có = passthrough, giảm chất lượng) |

## Tier C — Tracing + alerting (optional, debug)

| Key | Lấy ở đâu | Free | Dùng cho |
|---|---|---|---|
| `LANGCHAIN_API_KEY` | https://smith.langchain.com/settings | 5k trace/tháng | LangSmith trace. Set `LANGCHAIN_TRACING_V2=true` |
| `HF_TOKEN` | https://huggingface.co/settings/tokens | Có | BGE-M3 download (model public — token chỉ để tránh rate limit 429). Bỏ trống OK. |
| `ALERT_WEBHOOK_URL` | Slack/Teams/Discord incoming webhook | Có | Alertmanager receiver. Bỏ trống = alert đi vào void (placeholder 127.0.0.1:9999). |

## Tier D — AWS (bắt buộc deploy EC2)

| Key | Lấy ở đâu | Scope tối thiểu | Lưu ý |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | IAM Console → Users → Security credentials | IAM user riêng, KHÔNG root | Gắn policy tối thiểu (dưới) |
| `ROUTE53_HOSTED_ZONE_ID` | Route 53 → Hosted zones | read/write Records | Cần domain đã mua + hosted zone ($0.50/tháng) |
| `EC2_KEY_PAIR_NAME` | EC2 → Key Pairs → Create | — | Private `.pem` giữ máy, KHÔNG commit, chmod 600 |
| `S3_BACKUP_BUCKET` | S3 → Create bucket | 5GB free, sau $0.023/GB | Block public access ON, versioning ON |

### IAM policy tối thiểu (deploy user)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["ec2:*"],"Resource":"*"},
    {"Effect":"Allow","Action":["s3:*"],"Resource":"*"},
    {"Effect":"Allow","Action":["route53:*"],"Resource":"*"},
    {"Effect":"Allow","Action":["iam:PassRole"],"Resource":"*"},
    {"Effect":"Allow","Action":["secretsmanager:*"],"Resource":"*"}
  ]
}
```
> Prod thật: giới hạn `Resource` = ARN cụ thể. Đây là starter.

### EC2 instance role (terraform-managed, KHÔNG cần key tĩnh trên EC2)
`main.tf` tạo `legal-chatbot-instance-role` + gắn vào EC2. Role này chỉ có
`s3:PutObject/GetObject/DeleteObject/ListBucket` trên `S3_BACKUP_BUCKET` —
để `deployment/scripts/backup.sh` chạy `aws s3 cp` mà không cần AWS key trên
EC2. KHÔNG dùng key này cho CI/terraform — tách biệt hoàn toàn.
- Backup: cron nightly 03:17 UTC (user_data.sh). Local prune 7 ngày, S3 lifecycle 30 ngày (set bucket lifecycle rule).
- Verify role gắn: `aws ec2 describe-instances --instance-ids <i-xxx> --query 'Reservations[0].Instances[0].IamInstanceProfile'`

## Tier E — Container registry (GHCR recommend, free)

| Key | Lấy ở đâu | Scope | Dùng cho |
|---|---|---|---|
| `REGISTRY_PASSWORD` (GHCR PAT) | https://github.com/settings/tokens (classic) | `write:packages`, `read:packages` | GitHub Actions push image |

> Hoặc **AWS ECR**: Console → ECR → Create repo. Thêm policy `ecr:*`. GHCR đơn giản hơn cho app 1 người.

---

## Quy tắc bảo mật (KHÔNG thương lượng)

1. **Root AWS key** = KHÔNG BAO GIỜ dùng. IAM user riêng + policy tối thiểu.
2. **Mọi password** ≥16 ký tự random. KHÔNG "admin", "legal", "123".
3. **`.pem` key pair** giữ máy cá nhân, chmod 600. Mất = mất truy cập EC2.
4. **`.env` đã fill** KHÔNG commit, KHÔNG gửi Slack/Discord, KHÔNG dán chat AI.
5. **Rotate** key 3-6 tháng/lần, đặc biệt sau khi ai rời team.
6. **AWS Secrets Manager**: nâng cao — app pull secret runtime thay file `.env`. Chuyển khi team ≥2.
7. **CloudTrail** bật ở AWS account → audit mọi API call (free).

## Thứ tự lấy key (1 buổi)

```
1. Groq key (2 phút)         ← app chạy demo được
2. Tự tạo JWT/admin/MCP/Grafana (5 phút)
3. AWS: IAM user + access key + key pair + S3 bucket (30 phút)
4. Route 53: mua domain + hosted zone (nếu chưa có, 15 phút)
5. Cohere + Tavily (10 phút/cái) ← RAG chất lượng lên
6. GHCR PAT (5 phút)        ← CI/CD push image
```

Sau fill: `grep -i "PLACEHOLDER\|change-me\|your_" backend/.env` → 0 dòng. Còn dòng = chưa fill.