#!/bin/bash
# Manual deployment on the EC2 instance (alternative to the CI/CD deploy job).
#   ./deploy-manual.sh ghcr.io/<owner>/<repo>:latest
set -euo pipefail
IMAGE=${1:?usage: $0 <image>}
docker pull "$IMAGE"
docker rm -f digit-classifier 2>/dev/null || true
docker run -d --name digit-classifier --restart unless-stopped -p 80:5000 "$IMAGE"
sleep 3
curl -s http://localhost/health && echo
echo "App running at: http://$(curl -s https://checkip.amazonaws.com)/"
