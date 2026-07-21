# Deployment — Legal Chatbot (AWS)

Folder gom artifact deploy production lên AWS + LLMOps Tier 1. Đọc theo thứ tự dưới.

## Cấu trúc

```
deployment/
├── README.md                  ← bạn đang đọc (index)
├── LLMOps_ROADMAP.md          ← roadmap 4 tuần lên LLMOps production-grade
├── docker-compose.prod.yml    ← prod compose (GHCR images + EBS volumes)
├── env/
│   ├── .env.prod.aws.example  ← template env đầy đủ (AWS keys) — copy → backend/.env
│   └── SECRETS_CHECKLIST.md   ← key nào lấy ở đâu, scope quyền tối thiểu
├── terraform/                 ← IaC: EC2 + EBS + EIP + SG + S3
│   ├── main.tf  variables.tf  terraform.tfvars.example  user_data.sh  README.md
├── github-actions/
│   └── deploy.yml             ← CI/CD: build+push GHCR + SSH deploy EC2
├── model_registry/            ← LLMOps Tier 1.1: version model/prompt
│   ├── models.yaml.example  model_registry.py  README.md
└── prompts/                   ← LLMOps Tier 1.2: prompt tách file + version
    ├── prompt_loader.py  agent_system.v1.txt  README.md
```

## Bắt đầu nhanh (thứ tự)

```
1. Lấy key:   đọc env/SECRETS_CHECKLIST.md → fill backend/.env (copy từ env/.env.prod.aws.example)
2. Infra:     cd terraform && terraform apply   (tạo EC2)
3. SSH vào EC2, clone repo, docker compose -f deployment/docker-compose.prod.yml up -d
4. CI/CD:     cp github-actions/deploy.yml → .github/workflows/  (deploy tự động khi git tag)
5. LLMOps:    làm theo LLMOps_ROADMAP.md tuần 1-4
```

## Tài liệu liên quan (ngoài folder này)

- `DEPLOY_AWS.md` (repo root) — architecture EC2 + Nginx + SSL step-by-step
- `RUN.md` — ops entry `scripts/dev.sh`
- `CODEBASE_GUIDE.md` — architecture layer
- `EVAL_SYSTEM_SUMMARY.md` — harness 7 phase (đã có, vũ khí chính)
- `docs/AWS_DEPLOY_PLAYBOOK.md` — tư duy senior AI dev + 8 readiness gates

## Bạn cần kiếm key về (ưu tiên)

→ Mở `env/SECRETS_CHECKLIST.md`. Tier A (Groq + tự tạo secret) đủ app chạy demo. Tier D (AWS IAM + key pair + S3) đủ deploy EC2. Tier E (GHCR PAT) đủ CI/CD.

Verify sau fill: `grep -i "PLACEHOLDER\|change-me\|your_" backend/.env` → phải 0 dòng.