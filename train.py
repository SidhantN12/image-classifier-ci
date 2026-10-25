"""DVC stage 'train': train the digit classifier on data/processed/train.npz."""
import numpy as np
import yaml

from model import MODEL_PATH, build_model, save_model

params = yaml.safe_load(open("params.yaml"))["train"]

if __name__ == "__main__":
    data = np.load("data/processed/train.npz")
    model = build_model(C=params["C"], gamma=params["gamma"])
    model.fit(data["X"], data["y"])
    save_model(model)
    print(f"Trained on {len(data['y'])} images, model saved to {MODEL_PATH}")
