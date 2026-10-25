# Image Classification with a GitHub Actions CI Pipeline

A handwritten-digit **image classifier** (scikit-learn SVM on the 8×8 `digits` dataset,
99.2% test accuracy) served through a small **Flask** API, with a **GitHub Actions** CI
pipeline that trains the model and runs the test suite on every push to `main`/`master`.

## Project structure

```
image-classifier-ci/
├── .github/workflows/ci.yaml   # CI pipeline
├── model.py                    # preprocessing, training, prediction
├── train.py                    # trains + saves models/digit_classifier.joblib
├── app.py                      # Flask API: GET /health, POST /predict
├── test.py                     # 11 pytest tests (data, model, preprocessing, API)
├── sample_images/              # example digit PNGs for trying the API
├── screenshots/                # CI run screenshots
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                # -> Test accuracy: 0.9917
pytest test.py -v              # -> 11 passed
python app.py                  # API on http://localhost:5000
curl -F image=@sample_images/digit_2.png http://localhost:5000/predict
# {"confidence":0.9794,"label":2}
```

## What test.py checks

| Area | Tests |
|------|-------|
| Data | train/test split shapes, all 10 classes present |
| Model | test accuracy ≥ 0.95, model file is saved |
| Preprocessing | any image → 1×64 vector in 0–16, light backgrounds are inverted |
| Prediction | ≥ 45/50 real digit images classified correctly, output format |
| API | `/health` 200, `/predict` returns correct label, 400 on missing / invalid file |

## CI pipeline (`.github/workflows/ci.yaml`)

Triggers on **push** (and pull requests) to `main` or `master`. A `test` job runs on
`ubuntu-latest` for Python 3.11 and 3.12:

1. Check out code
2. Set up Python (with pip cache)
3. Install dependencies from `requirements.txt`
4. Train the model (`python train.py`)
5. Run tests (`pytest test.py -v`), with JUnit XML
6. Upload the test results as a build artifact

## Triggering the pipeline and taking screenshots

```bash
git remote add origin https://github.com/<user>/image-classifier-ci.git
git push -u origin main
```

Then on GitHub open **Actions → CI** and take screenshots of:
1. The workflow run list with a green ✔
2. The run summary (both matrix jobs passed, artifacts listed)
3. The expanded **Run tests** step showing `11 passed`
