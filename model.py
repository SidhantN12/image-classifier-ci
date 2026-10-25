"""Image preprocessing, training and prediction for the digit classifier."""
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

MODEL_PATH = Path(__file__).parent / "models" / "digit_classifier.joblib"
IMAGE_SIZE = (8, 8)


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    return train_test_split(
        digits.data, digits.target,
        test_size=test_size, random_state=random_state, stratify=digits.target,
    )


def build_model(C=10, gamma=0.001):
    """SVM with calibrated probabilities, so predictions come with a confidence score."""
    return CalibratedClassifierCV(SVC(C=C, gamma=gamma), ensemble=False)


def save_model(model, model_path=MODEL_PATH):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def train(model_path=MODEL_PATH):
    """Train on the bundled digits dataset, save the model, and return (model, test accuracy)."""
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    model.fit(X_train, y_train)
    save_model(model, model_path)
    return model, model.score(X_test, y_test)


def load_model(model_path=MODEL_PATH):
    """Load the saved model, training one first if it does not exist yet."""
    if not Path(model_path).exists():
        return train(model_path)[0]
    return joblib.load(model_path)


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert any image to the 1x64 feature vector the model expects.

    The training data is a light digit on a dark background with pixel values 0-16,
    so dark-on-light images (e.g. a pen drawing) are inverted first.
    """
    gray = np.asarray(image.convert("L").resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.float64)
    if gray.mean() > 127:
        gray = 255.0 - gray
    return (gray / 255.0 * 16.0).reshape(1, -1)


def predict(image: Image.Image, model=None) -> dict:
    model = model if model is not None else load_model()
    probs = model.predict_proba(preprocess(image))[0]
    label = int(model.classes_[np.argmax(probs)])
    return {"label": label, "confidence": round(float(probs.max()), 4)}


def digit_to_image(pixels, scale=16) -> Image.Image:
    """Turn a 64-value digits sample (0-16) into an upscaled PIL image."""
    arr = (np.asarray(pixels).reshape(IMAGE_SIZE) / 16.0 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((8 * scale, 8 * scale), Image.NEAREST)
