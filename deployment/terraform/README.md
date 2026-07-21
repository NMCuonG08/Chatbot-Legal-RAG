# Terraform — Legal Chatbot AWS Single-EC2

Tạo hạ tầng: EC2 `t3.large` + EBS gp3 + Elastic IP + Security Group (chỉ 80/443 public, 22 admin IP) + 2 S3 bucket (backup + corpus, block public).

## Yêu cầu trước
1. **AWS CLI** + credentials (IAM user, KHÔNG root). Test: `aws sts get-caller-identity`.
2. **Terraform** ≥ 1.5. Cài: https://developer.hashicorp.com/terraform/downloads
3. **SSH key** local (terraform register pubkey vào AWS, không cần Console):
   ```bash
   cd deployment/terraform
   ssh-keygen -t ed25519 -f legal-prod-key -N "" -C legal-prod
   ```
   `legal-prod-key` (private) giữ máy — dùng SSH + GitHub secret `EC2_SSH_KEY`.
   `legal-prod-key.pub` (public) commit, terraform đọc tại apply.
4. **Public IP** của bạn (admin_cidr): https://ifconfig.me → `<IP>/32`.

## Chạy
```bash
cd deployment/terraform
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars          # fill admin_cidr, s3 bucket names (key_pair tự tạo)
terraform init
terraform plan -var-file=terraform.tfvars      # preview
terraform apply -var-file=terraform.tfvars     # nhập yes
```

Output cho: `ec2_public_ip`, `ssh_command`, `s3_backup_bucket`.

## Sau apply
```bash
ssh -i <key>.pem ubuntu@<ec2_public_ip>
cd app
cp deployment/env/.env.prod.aws.example backend/.env
nano backend/.env              # fill key (xem env/SECRETS_CHECKLIST.md)
docker compose up -d --build
```

## Domain + SSL
```bash
# Trong EC2, sau khi app up:
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com -d grafana.yourdomain.com
# Route53: A record → ec2_public_ip (hoặc uncomment aws_route53_record trong main.tf)
```

## Hủy (CẨN THẬN — mất EBS data)
```bash
terraform destroy -var-file=terraform.tfvars
```

## Lưu ý
- `admin_cidr` default `0.0.0.0/0` = nguy hiểm. BẮT BUỘC đổi thành IP/32 của bạn.
- EBS encrypted=true (mặc định).
- S3 block public access ON, versioning ON cho backup.
- `terraform.tfvars` chứa IP cá nhân → gitignore. Thêm vào root `.gitignore` nếu chưa: `deployment/terraform/terraform.tfvars`.
- Upgrade multi-AZ/RDS: dùng module `terraform-aws-modules/vpc/aws` + RDS. Overkill cho MVP.