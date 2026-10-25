#!/bin/bash
# EC2 user data (Ubuntu 24.04): installs Docker so the CI/CD pipeline can deploy the app.
# Paste into "Advanced details -> User data" when launching the instance.
set -eux
apt-get update -y
apt-get install -y docker.io curl
systemctl enable --now docker
usermod -aG docker ubuntu
