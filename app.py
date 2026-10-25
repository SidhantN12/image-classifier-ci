"""Flask web app for the handwritten digit image classifier.

    GET  /         -> web page: draw or upload a digit
    GET  /health   -> {"status": "ok"}
    POST /predict  -> multipart form field "image" (PNG/JPG) -> {"label": 7, "confidence": 0.99}
"""
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from model import load_model, predict

app = Flask(__name__)
classifier = load_model()
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")


@app.get("/")
def index():
    return render_template("index.html", model_version=MODEL_VERSION)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict_route():
    if "image" not in request.files:
        return jsonify(error="No file uploaded under form field 'image'"), 400
    try:
        image = Image.open(request.files["image"].stream)
    except UnidentifiedImageError:
        return jsonify(error="Uploaded file is not a valid image"), 400
    return jsonify(predict(image, classifier))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
