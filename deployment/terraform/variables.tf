variable "aws_region" {
  type        = string
  default     = "ap-southeast-1"
  description = "AWS region for EC2 + S3"
}

variable "instance_type" {
  type        = string
  default     = "t3.large"
  description = "EC2 type. t3.large=8GB (load BGE-M3). t3.medium=4GB if only Groq API."
}

variable "ebs_size_gb" {
  type        = number
  default     = 50
  description = "EBS gp3 size (GB). Qdrant + MariaDB + logs grow here."
}

variable "admin_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "CIDR allowed SSH. MUST be YOUR_IP/32 in prod. NEVER leave 0.0.0.0/0."
}

variable "s3_backup_bucket" {
  type        = string
  description = "Globally-unique S3 bucket name for MariaDB + Qdrant backups."
}

variable "s3_corpus_bucket" {
  type        = string
  description = "Globally-unique S3 bucket name for legal corpus snapshots."
}

variable "budget_limit_amount" {
  type        = string
  default     = "5"
  description = "Monthly USD budget ceiling. AWS emails budget_email at 80% (actual) + 100% (forecast)."
}

variable "budget_email" {
  type        = string
  default     = "admin@example.com"
  description = "Email address for AWS billing alerts (budget threshold)."
}

variable "github_repo_url" {
  type        = string
  default     = "https://github.com/NMCuong08/Chatbot-Legal-RAG.git"
  description = "Repo to clone on EC2 at first boot."
}

variable "branch" {
  type    = string
  default = "main"
}

# Uncomment with Route53 record when domain ready
# variable "route53_hosted_zone_id" { type = string }
# variable "domain" { type = string }