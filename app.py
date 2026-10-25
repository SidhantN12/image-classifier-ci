"""Flask API for the handwritten digit image classifier.

    GET  /health   -> {"status": "ok"}
    POST /predict  -> multipart form field "image" (PNG/JPG) -> {"label": 7, "confidence": 0.99}
"""
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from model import load_model, predict

app = Flask(__name__)
classifier = load_model()


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
    app.run(host="0.0.0.0", port=5000)
