# model_evaluation.py
# loads the saved model and checks how well it does on validation data

import os
import json
import numpy as np
import yaml
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open("params.yaml") as f:
    params = yaml.safe_load(f)

processed_dir = params["processing"]["processed_dir"]
metrics_file = params["evaluation"]["metrics_file"]
plots_dir = params["evaluation"]["plots_dir"]
classes = params["data"]["classes"]


def evaluate():
    X_val = np.load(os.path.join(processed_dir, "X_val_feat.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val_feat.npy"))

    with open("data/model/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds, target_names=classes))

    # save accuracy to json for dvc metrics
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump({"accuracy": round(acc, 4)}, f)

    # save confusion matrix
    os.makedirs(plots_dir, exist_ok=True)
    cm = confusion_matrix(y_val, preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    print("Plot saved.")


if __name__ == "__main__":
    evaluate()
