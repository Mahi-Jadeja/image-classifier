# model_training.py
# trains a random forest on the processed features and saves the model

import os
import numpy as np
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml") as f:
    params = yaml.safe_load(f)

processed_dir = params["processing"]["processed_dir"]
n_estimators = params["training"]["n_estimators"]
max_depth = params["training"]["max_depth"]
random_state = params["training"]["random_state"]


def train():
    X_train = np.load(os.path.join(processed_dir, "X_train_feat.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train_feat.npy"))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    print("Training...")
    model.fit(X_train, y_train)
    print("Done.")

    os.makedirs("data/model", exist_ok=True)
    with open("data/model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved.")


if __name__ == "__main__":
    train()
