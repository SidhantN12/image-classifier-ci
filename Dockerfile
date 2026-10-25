# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

COPY requirements.txt .
# Optional: behind a TLS-intercepting proxy, pass its CA with
#   docker build --secret id=pip_ca,src=/path/to/ca.crt .
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir -r requirements.txt

# Application code + the DVC-tracked model (produced by `dvc pull` or `dvc repro`)
COPY model.py app.py ./
COPY templates/ templates/
COPY models/digit_classifier.joblib models/

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

RUN useradd --create-home appuser
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 app:app"]
