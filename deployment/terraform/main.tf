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
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# ---------- Security Group: chỉ mở 80/443 public, 22 restrict IP -------------
resource "aws_security_group" "legal_sg" {
  name        = "legal-chatbot-sg"
  description = "Legal chatbot prod — only 80/443 public, 22 admin IP"
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

# ---------- EC2 instance ------------------------------------------------------
resource "aws_instance" "legal_app" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  key_name                    = var.key_pair_name
  vpc_security_group_ids      = [aws_security_group.legal_sg.id]
  associate_public_ip_address = true
  monitoring                  = true

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
  block_public_policy      = true
  ignore_public_acls       = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "corpus" {
  bucket = var.s3_corpus_bucket
  tags   = { Name = "legal-corpus" }
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  bucket                  = aws_s3_bucket.corpus.id
  block_public_acls       = true
  block_public_policy      = true
  ignore_public_acls       = true
  restrict_public_buckets = true
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
output "ec2_public_ip"   { value = aws_eip.legal_eip.public_ip }
output "ec2_public_dns"  { value = aws_instance.legal_app.public_dns }
output "s3_backup_bucket" { value = aws_s3_bucket.backups.id }
output "ssh_command"     { value = "ssh -i ${var.key_pair_name}.pem ubuntu@${aws_eip.legal_eip.public_ip}" }