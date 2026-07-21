#!/bin/bash
# Cloud-init: runs once at EC2 first boot. Installs Docker, clones repo.
# Template vars (injected by Terraform templatefile): ${github_repo_url}, ${branch}
set -eux

apt-get update -y
apt-get upgrade -y
apt-get install -y git curl nginx certbot python3-certbot-nginx awscli

# Docker
curl -fsSL https://get.docker.com -o /root/get-docker.sh
sh /root/get-docker.sh
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# docker compose plugin
DOCKER_CONFIG=/root/.docker
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Link for ubuntu user
mkdir -p /home/ubuntu/.docker/cli-plugins
ln -s $DOCKER_CONFIG/cli-plugins/docker-compose /home/ubuntu/.docker/cli-plugins/docker-compose
chown -R ubuntu:ubuntu /home/ubuntu/.docker

# Clone repo
cd /home/ubuntu
git clone -b "${branch}" "${github_repo_url}" app 2>/dev/null || (cd app && git pull)
chown -R ubuntu:ubuntu /home/ubuntu/app

# Persistent data dirs
mkdir -p /home/ubuntu/app/data/mariadb /home/ubuntu/app/data/qdrant /home/ubuntu/app/data/redis \
         /home/ubuntu/app/data/grafana /home/ubuntu/app/data/embed-models /home/ubuntu/app/data/backups
chown -R ubuntu:ubuntu /home/ubuntu/app/data

# --- Host nginx reverse proxy (compose binds 127.0.0.1; proxy exposes 80/443)
cp -f /home/ubuntu/app/deployment/nginx/legal.conf /etc/nginx/sites-available/legal.conf
ln -sf /etc/nginx/sites-available/legal.conf /etc/nginx/sites-enabled/legal.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl enable nginx && systemctl restart nginx || echo "[cloud-init] nginx config not ready yet (compose not up) — will reload on first deploy"

# --- Nightly backup cron (deployment/scripts/backup.sh -> S3 via instance role)
chmod +x /home/ubuntu/app/deployment/scripts/backup.sh
( crontab -l -u ubuntu 2>/dev/null; echo "17 3 * * * /home/ubuntu/app/deployment/scripts/backup.sh >> /home/ubuntu/app/data/backups/backup.log 2>&1" ) | crontab -u ubuntu -

echo "Cloud-init done. First GitHub Actions deploy will: git pull -> assemble .env -> compose pull+up -> reload nginx." > /etc/motd