# feature_selection.py
# flattens and normalizes image arrays so sklearn can use them

import os
import numpy as np
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

processed_dir = params["processing"]["processed_dir"]
normalize = params["features"]["normalize"]
flatten = params["features"]["flatten"]


def process_features(X):
    if flatten:
        # 64x64x3 image becomes a 12288 length vector
        X = X.reshape(len(X), -1)
    if normalize:
        # bring pixel values to 0-1 range
        X = X / 255.0
    return X


def run():
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    X_train = process_features(X_train)
    X_val = process_features(X_val)

    np.save(os.path.join(processed_dir, "X_train_feat.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_val_feat.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_train_feat.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_val_feat.npy"), y_val)

    print(f"Feature shape: {X_train.shape}")


if __name__ == "__main__":
    run()
