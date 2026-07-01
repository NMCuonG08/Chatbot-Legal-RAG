# AWS Single-Instance Production Deployment Plan 🚀

This document outlines the architecture, cost estimation, and step-by-step instructions for deploying the Vietnamese Legal Assistant on AWS. The deployment is optimized to be **highly cost-effective** (keeping databases and caching inside the main EC2 instance) while ensuring production-grade security, SSL, and monitoring.

---

## 🏗️ Architecture Overview

To minimize costs, we host all dockerized microservices on a single, medium-to-large EC2 instance. This eliminates the need for expensive managed services like AWS RDS, ElastiCache, or Application Load Balancer.

```
User ──► Route 53 ──► EC2 Instance (Elastic IP)
                        │
                        ├── Nginx Reverse Proxy (SSL via Let's Encrypt)
                        │     ├── Port 8501 ──► Streamlit UI
                        │     ├── Port 8000 ──► FastAPI Backend (Celery tasks)
                        │     └── Port 3000 ──► Grafana Dashboard
                        │
                        └── Docker Compose Tiers (EBS Persistent Volumes)
                              ├── Redis Broker (Celery tasks & checkpointers)
                              ├── Qdrant Vector DB (semantic chunks & memory)
                              ├── MariaDB/PostgreSQL (traces & history)
                              ├── Prometheus (metrics scraping)
                              └── Grafana (monitoring visualization)
```

### Key AWS Services Used:
1. **Route 53:** DNS routing and domain mapping.
2. **EC2 Instance:** Single computing host running Ubuntu 22.04 LTS.
3. **EBS Volume (gp3):** Persistent block storage attached to the EC2 instance to persist database data.
4. **Elastic IP:** A static IPv4 address for domain consistency.

---

## 💰 Cost Estimation (USD/Month)

| Service | Configuration | Estimated Cost (On-Demand) | Estimated Cost (Savings Plan / 1-Yr RI) |
|---------|---------------|---------------------------|----------------------------------------|
| **EC2** | `t3.large` (2 vCPUs, 8GB RAM)* | ~$60.00 | ~$37.00 |
| **EBS** | 30 GB gp3 Storage | ~$2.40 | ~$2.40 |
| **Route 53** | 1 Hosted Zone | $0.50 | $0.50 |
| **Traffic / IP**| Elastic IP + Data Transfer | ~$3.00 | ~$3.00 |
| **SSL Certs** | Let's Encrypt (Nginx) | **FREE** | **FREE** |
| **Total** | | **~$65.90 / month** | **~$42.90 / month** |

*\*Note: We recommend a `t3.large` instance because loading embedding models (BGE-M3) and processing legal text chunks requires stable memory. If you use external cloud APIs (Groq, OpenAI, Cohere) for all compute, a `t3.medium` (4GB RAM) can be used to lower costs further (approx. ~$25/month).*

---

## 🛠️ Step-by-Step Deployment Guide

### Step 1: Launch the EC2 Instance
1. Go to the AWS EC2 Console and click **Launch Instance**.
2. Select **Ubuntu Server 22.04 LTS** (64-bit x86).
3. Select Instance Type: `t3.large` (or `t3.medium` for a tighter budget).
4. Configure Key Pair for SSH access.
5. Under Network Settings, create a new Security Group:
   * **Port 22 (SSH):** Restricted to your personal IP.
   * **Port 80 (HTTP):** Open to Anywhere (0.0.0.0/0).
   * **Port 443 (HTTPS):** Open to Anywhere (0.0.0.0/0).
   * *Do NOT open ports 3000 (Grafana), 3306 (DB), 6333 (Qdrant), or 8000 (API) to the public. These will be securely routed through Nginx locally.*
6. Set Storage to **30 GB gp3** (EBS volume).
7. Allocate an **Elastic IP** in the EC2 Console and associate it with your instance.

### Step 2: Install Docker and Git on EC2
SSH into your instance:
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-elastic-ip
```
Update packages and install dependencies:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git curl nginx certbot python3-certbot-nginx

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker # Reload shell permissions without logging out
```

### Step 3: Clone Codebase and Configure Environment
```bash
git clone https://github.com/NMCuonG08/Chatbot-Legal-RAG.git
cd Chatbot-Legal-RAG

# Create folders for persistent data storage
mkdir -p data/mariadb data/qdrant data/redis data/pipeline_lake

# Copy env template and customize
cp backend/.env.example backend/.env
nano backend/.env
```
Ensure your database URLs in `backend/.env` point to the internal Docker service names:
```env
DATABASE_URL=mysql+pymysql://legal:legal@mariadb:3306/legal_db
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
```

### Step 4: Configure Nginx Reverse Proxy & SSL
Configure Nginx to route traffic to the container ports:
```bash
sudo nano /etc/nginx/sites-available/legal-assistant
```
Paste the configuration below (replace `yourdomain.com` with your actual domain):
```nginx
server {
    listen 80;
    server_name yourdomain.com api.yourdomain.com grafana.yourdomain.com;

    # Redirect all HTTP traffic to HTTPS once Certbot is configured
    location / {
        return 301 https://$host$request_uri;
    }
}

# Streamlit Frontend
server {
    listen 443 ssl;
    server_name yourdomain.com;

    # SSL Certs will be filled automatically by Certbot

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# FastAPI Backend API
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Grafana Monitoring Dashboard
server {
    listen 443 ssl;
    server_name grafana.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
Enable the site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/legal-assistant /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo systemctl restart nginx
```

#### Get Free SSL Certificates via Certbot:
```bash
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com -d grafana.yourdomain.com
```
Certbot will configure SSL parameters and handle automatic certificate renewals.

### Step 5: Start the Dockerized Application
Ensure Qdrant, MariaDB, Redis, and metrics scrapers write data to the persistent host volumes:
```bash
# Add volumes mapped to the EC2 filesystem in docker-compose.yml (already configured)
docker compose up -d --build
```

---

## 📈 Accessing Monitoring & Metrics
* **Frontend Application:** Access at `https://yourdomain.com`
* **FastAPI documentation:** Access at `https://api.yourdomain.com/docs`
* **Grafana:** Access at `https://grafana.yourdomain.com` (Credentials: Admin / `admin` - change password upon first login).
