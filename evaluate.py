"""DVC stage 'evaluate': score the model on data/processed/test.npz and write metrics.json."""
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from model import load_model

data = np.load("data/processed/test.npz")
pred = load_model().predict(data["X"])
metrics = {
    "accuracy": round(accuracy_score(data["y"], pred), 4),
    "f1_macro": round(f1_score(data["y"], pred, average="macro"), 4),
    "test_images": int(len(data["y"])),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)
