"""Export the scikit-learn digits dataset as PNG images: data/raw/<label>/<index>.png.

This is the raw image dataset that DVC versions and stores in S3.
"""
from pathlib import Path

from sklearn.datasets import load_digits

from model import digit_to_image

RAW_DIR = Path("data/raw")

if __name__ == "__main__":
    digits = load_digits()
    for i, (pixels, label) in enumerate(zip(digits.data, digits.target)):
        out = RAW_DIR / str(label) / f"{i:04d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        digit_to_image(pixels, scale=4).save(out)
    print(f"Exported {len(digits.target)} images to {RAW_DIR}/")
