# ============================================================================
# Legal Chatbot — AWS single-EC2 production infra
# ----------------------------------------------------------------------------
# Provision:  EC2 (t3.large) + EBS gp3 + Elastic IP + Security Group + S3 buckets
# Usage:
#   terraform init
#   terraform plan -var-file=terraform.tfvars
#   terraform apply -var-file=terraform.tfvars
# Destroy:    terraform destroy -var-file=terraform.tfvars  (CAREFUL — loses EBS data)
# ============================================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# ---------- Security Group: chỉ mở 80/443 public, 22 restrict IP -------------
resource "aws_security_group" "legal_sg" {
  name        = "legal-chatbot-sg"
  description = "Legal chatbot prod - only 80/443 public, 22 admin IP"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH admin"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr] # YOUR_IP/32 — NOT 0.0.0.0/0
  }

  ingress {
    description = "HTTP (redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "legal-chatbot-sg" }
}

# ---------- EC2 SSH key pair (registered from a local public key) -------------
# Generate locally once:  ssh-keygen -t ed25519 -f legal-prod-key -N "" -C legal-prod
# Terraform registers ONLY the public key with AWS. The private key
# (legal-prod-key) stays on your machine — it NEVER enters terraform state.
# Use legal-prod-key as the .pem for SSH + as GitHub secret EC2_SSH_KEY.
resource "aws_key_pair" "legal" {
  key_name   = "legal-prod-key"
  public_key = file("${path.module}/legal-prod-key.pub")
  tags       = { Name = "legal-prod-key" }
}

# ---------- EC2 instance ------------------------------------------------------
resource "aws_instance" "legal_app" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  key_name                    = aws_key_pair.legal.key_name
  vpc_security_group_ids      = [aws_security_group.legal_sg.id]
  associate_public_ip_address = true
  monitoring                  = true
  iam_instance_profile        = aws_iam_instance_profile.legal_instance_profile.name

  user_data = templatefile("${path.module}/user_data.sh", {
    github_repo_url = var.github_repo_url
    branch          = var.branch
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = var.ebs_size_gb
    encrypted   = true
  }

  tags = { Name = "legal-chatbot-prod" }
}

# ---------- IAM instance role (least-privilege S3 backup access) -------------
# Lets the EC2 host run `aws s3 cp` for backups WITHOUT static AWS creds on disk.
# Backup script (deployment/scripts/backup.sh) relies on this role.
resource "aws_iam_role" "legal_instance_role" {
  name = "legal-chatbot-instance-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_instance_profile" "legal_instance_profile" {
  name = "legal-chatbot-instance-profile"
  role = aws_iam_role.legal_instance_role.name
}

# Least privilege: only the backup bucket, only the actions backup.sh needs.
resource "aws_iam_role_policy" "legal_s3_backup" {
  name = "legal-s3-backup"
  role = aws_iam_role.legal_instance_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads"
        ]
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
      }
    ]
  })
}

# ---------- Elastic IP (static IP for Route53) --------------------------------
resource "aws_eip" "legal_eip" {
  instance = aws_instance.legal_app.id
  domain   = "vpc"
  tags     = { Name = "legal-chatbot-eip" }
}

# ---------- S3: backup + corpus (block public access) -------------------------
resource "aws_s3_bucket" "backups" {
  bucket = var.s3_backup_bucket
  tags   = { Name = "legal-backups" }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket                  = aws_s3_bucket.backups.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "corpus" {
  bucket = var.s3_corpus_bucket
  tags   = { Name = "legal-corpus" }
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  bucket                  = aws_s3_bucket.corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------- Billing budget (email alert before cost surprises) ----------------
# AWS emails budget_email at 80% actual spend + 100% forecast. The #1 defense
# against "forgot a resource, got a big bill". Tune budget_limit_amount in tfvars.
resource "aws_budgets_budget" "legal" {
  name         = "legal-chatbot-monthly"
  budget_type  = "COST"
  limit_amount = var.budget_limit_amount
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  # Adjust the start month if you apply in a later month. Format: YYYY-MM-DD_HH:MM
  time_period_start = "2026-07-01_00:00"

  notification {
    notification_type          = "ACTUAL"
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    subscriber_email_addresses = [var.budget_email]
  }
  notification {
    notification_type          = "FORECASTED"
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    subscriber_email_addresses = [var.budget_email]
  }
}

# ---------- Route53 A record (uncomment after EIP known + hosted zone ready) --
# resource "aws_route53_record" "app" {
#   zone_id = var.route53_hosted_zone_id
#   name    = var.domain
#   type    = "A"
#   ttl     = 60
#   records = [aws_eip.legal_eip.public_ip]
# }

# ---------- Outputs -----------------------------------------------------------
output "ec2_public_ip" { value = aws_eip.legal_eip.public_ip }
output "ec2_public_dns" { value = aws_instance.legal_app.public_dns }
output "s3_backup_bucket" { value = aws_s3_bucket.backups.id }
output "ssh_command" { value = "ssh -i ${aws_key_pair.legal.key_name} ubuntu@${aws_eip.legal_eip.public_ip}" }