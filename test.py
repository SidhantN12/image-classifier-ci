"""Tests for the image classification app. Run with: pytest test.py -v"""
import io

import numpy as np
import pytest
from PIL import Image

from model import digit_to_image, load_data, predict, preprocess, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    model, accuracy = train(model_path)
    return model, accuracy, model_path


@pytest.fixture(scope="module")
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


# --- data & model -----------------------------------------------------------

def test_dataset_split_shapes():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == X_test.shape[1] == 64
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert set(np.unique(y_train)) == set(range(10))


def test_model_accuracy_above_threshold(trained):
    _, accuracy, _ = trained
    assert accuracy >= 0.95, f"accuracy {accuracy:.4f} below 0.95"


def test_model_is_saved(trained):
    _, _, model_path = trained
    assert model_path.exists() and model_path.stat().st_size > 0


# --- preprocessing ------------------------------------------------------------

def test_preprocess_output_shape_and_range():
    image = Image.new("RGB", (100, 60), color=(30, 30, 30))
    features = preprocess(image)
    assert features.shape == (1, 64)
    assert features.min() >= 0 and features.max() <= 16


def test_preprocess_inverts_light_background():
    white = preprocess(Image.new("L", (32, 32), color=255))
    assert np.allclose(white, 0), "white background should become 0 after inversion"


# --- prediction -----------------------------------------------------------------

def test_predict_real_digit_images(trained):
    model, _, _ = trained
    _, X_test, _, y_test = load_data()
    correct = sum(
        predict(digit_to_image(x), model)["label"] == y for x, y in zip(X_test[:50], y_test[:50])
    )
    assert correct >= 45, f"only {correct}/50 upscaled images classified correctly"


def test_predict_output_format(trained):
    model, _, _ = trained
    _, X_test, _, _ = load_data()
    result = predict(digit_to_image(X_test[0]), model)
    assert set(result) == {"label", "confidence"}
    assert 0 <= result["label"] <= 9
    assert 0.0 <= result["confidence"] <= 1.0


# --- Flask API ------------------------------------------------------------------

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_predict_endpoint(client):
    _, X_test, _, y_test = load_data()
    resp = client.post(
        "/predict",
        data={"image": (png_bytes(digit_to_image(X_test[0])), "digit.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["label"] == int(y_test[0])


def test_predict_endpoint_missing_file(client):
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_endpoint_invalid_image(client):
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"not an image"), "bad.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
