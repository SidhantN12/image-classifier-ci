# Digit Image Classifier: DVC + GitHub Actions CI/CD + Docker + AWS

An end-to-end MLOps project. A handwritten-digit image classifier (SVM, **99.17 %** test
accuracy on 360 held-out images) is:

- **versioned with DVC**: 1,797 raw PNG images, the processed data and the trained model, stored in an **AWS S3** remote
- **tested and built by GitHub Actions**: the DVC pipeline is reproduced, 12 pytest tests run, and a Docker image is built, smoke-tested and pushed to GHCR
- **deployed with Docker on AWS EC2**: the pipeline SSHes into the instance and runs the new image on port 80

```
 git push ─► GitHub Actions ─┬─ test:   dvc pull/repro ─► pytest
                             ├─ build:  docker build ─► smoke test ─► push ghcr.io
                             └─ deploy: ssh EC2 ─► docker pull ─► docker run -p 80:5000
 DVC remote ◄──────────────── S3 bucket (data/raw, data/processed, model)
```

## Project structure

```
├── .github/workflows/ci-cd.yaml   # CI/CD pipeline (test → build → deploy)
├── dvc.yaml / dvc.lock            # DVC pipeline: prepare → train → evaluate
├── params.yaml                    # pipeline hyper-parameters
├── data/raw.dvc                   # DVC pointer to the 1,797-image dataset
├── metrics.json                   # evaluation metrics (DVC metric)
├── export_dataset.py              # writes the raw PNG dataset to data/raw/<label>/
├── prepare.py / train.py / evaluate.py   # DVC stages
├── model.py                       # preprocessing, model, prediction
├── app.py + templates/index.html  # Flask web app (draw/upload a digit)
├── test.py                        # 12 pytest tests
├── Dockerfile / .dockerignore
├── deploy/ec2-user-data.sh        # installs Docker on a new EC2 instance
├── deploy/deploy-manual.sh        # manual deploy on EC2
├── requirements.txt               # app runtime (in Docker image)
└── requirements-dev.txt           # + dvc[s3], pytest
```

## 1. Run locally

```bash
pip install -r requirements-dev.txt
python export_dataset.py      # or: dvc pull  (after S3 is configured)
dvc repro                     # prepare → train → evaluate
dvc metrics show
pytest test.py -v
python app.py                 # http://localhost:5000
```

## 2. AWS S3 as DVC remote

```bash
aws configure                                     # IAM user with S3 access
aws s3 mb s3://<your-bucket> --region us-east-1
dvc remote modify s3remote url s3://<your-bucket>/image-classifier
dvc push                                          # uploads images, processed data, model
aws s3 ls s3://<your-bucket>/image-classifier/ --recursive --summarize
git commit -am "Point DVC remote to S3" && git push
```

## 3. Docker

```bash
docker build -t digit-classifier .
docker run -d -p 8080:5000 digit-classifier       # http://localhost:8080
```

## 4. EC2 instance

1. EC2 → **Launch instance**: Ubuntu 24.04, `t2.micro`/`t3.micro` (free tier), create a key pair (`.pem`).
2. Security group inbound rules: **SSH 22** (your IP) and **HTTP 80** (0.0.0.0/0).
3. Advanced details → **User data**: paste `deploy/ec2-user-data.sh` (installs Docker).
4. Note the instance's **Public IPv4 address**.

## 5. Connect GitHub Actions to AWS

Repository → **Settings → Secrets and variables → Actions**:

| Type | Name | Value |
|------|------|-------|
| Variable | `EC2_HOST` | EC2 public IP (enables the deploy job) |
| Variable | `EC2_USER` | `ubuntu` |
| Secret | `EC2_SSH_KEY` | contents of the `.pem` key file |
| Secret | `AWS_ACCESS_KEY_ID` | IAM key (CI then uses `dvc pull` from S3) |
| Secret | `AWS_SECRET_ACCESS_KEY` | IAM secret |

Push to `main` (or Actions → CI/CD → **Run workflow**). The deploy job runs the new image
on EC2, then checks `http://<EC2_HOST>/health`.

## 6. Verify

Open `http://<EC2-public-IP>/`, draw a digit, and click **Predict drawing**. Or:

```bash
curl http://<EC2-public-IP>/health
curl -F image=@sample_images/digit_2.png http://<EC2-public-IP>/predict
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | `{"status": "ok"}` |
| POST | `/predict` | form field `image` → `{"label": 2, "confidence": 0.98}` |
