# data_processing.py
# resizes images and saves them as numpy arrays, also does train/val split

import os
import numpy as np
import yaml
from PIL import Image

with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_dir = params["data"]["raw_dir"]
classes = params["data"]["classes"]
image_size = params["processing"]["image_size"]
processed_dir = params["processing"]["processed_dir"]
val_split = params["processing"]["validation_split"]


def load_images():
    X, y = [], []

    for label, cls in enumerate(classes):
        folder = os.path.join(raw_dir, cls)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size))
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass  # skip bad images

    return np.array(X), np.array(y)


def split_and_save(X, y):
    os.makedirs(processed_dir, exist_ok=True)

    # shuffle before splitting
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")


if __name__ == "__main__":
    X, y = load_images()
    split_and_save(X, y)
