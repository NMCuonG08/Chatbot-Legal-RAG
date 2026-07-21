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
mkdir -p /home/ubuntu/app/data/mariadb /home/ubuntu/app/data/qdrant /home/ubuntu/app/data/redis
chown -R ubuntu:ubuntu /home/ubuntu/app/data

echo "Cloud-init done. SSH in, fill backend/.env, then: docker compose up -d --build" > /etc/motd