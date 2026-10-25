"""DVC stage 'prepare': load raw PNG images, preprocess them and split into train/test."""
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from model import preprocess

params = yaml.safe_load(open("params.yaml"))["prepare"]

paths = sorted(Path("data/raw").glob("*/*.png"))
X = np.vstack([preprocess(Image.open(p)) for p in paths])
y = np.array([int(p.parent.name) for p in paths])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
)

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "train.npz", X=X_train, y=y_train)
np.savez_compressed(out / "test.npz", X=X_test, y=y_test)
print(f"{len(paths)} images -> train: {len(y_train)}, test: {len(y_test)}")
