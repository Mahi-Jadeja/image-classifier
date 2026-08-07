# Image Classification Pipeline — Documentation

**Name:** YOUR_NAME_HERE  
**PRN:** YOUR_PRN_HERE  

## GitHub Repository
[https://github.com/YOUR_USERNAME/image_classifier](https://github.com/YOUR_USERNAME/image_classifier)

---

## What This Project Does
Classifies flower images (daisy, dandelion, roses) using a Random Forest model.
The pipeline is fully automated using DVC — one command runs all 5 stages.

---

## Tech Stack
- Python 3.x
- scikit-learn
- DVC
- Pillow
- NumPy, Matplotlib

---

## Pipeline Stages

| Stage | Script | What it does |
|---|---|---|
| data_collection | src/data_collection.py | Downloads flower images from Google |
| data_processing | src/data_processing.py | Resizes to 64x64, splits train/val |
| feature_selection | src/feature_selection.py | Flattens and normalizes arrays |
| model_training | src/model_training.py | Trains Random Forest |
| model_evaluation | src/model_evaluation.py | Accuracy score + confusion matrix |

---

## Parameters (params.yaml) — 10+ tunable values

| Parameter | Value | Why it matters |
|---|---|---|
| image_size | 64 | All images become 64x64 |
| validation_split | 0.2 | 20% data for testing |
| n_estimators | 100 | Number of trees in forest |
| max_depth | 10 | How deep each tree grows |
| normalize | true | Pixels scaled 0-1 |
| flatten | true | Image to 1D for sklearn |
| max_images_per_class | 100 | Limits dataset size |
| random_state | 42 | Reproducible results |
| learning_rate | 0.1 | Ready for gradient boosting |
| dropout_rate | 0.3 | Ready for neural network |
| batch_size | 32 | Configurable for future deep learning |
| epochs | 10 | Configurable for future deep learning |

---

## How to Run

```bash
pip install -r requirements.txt
git init && dvc init
dvc repro
dvc metrics show
```

---

## Output Files
- `data/model/model.pkl` — trained model
- `data/metrics.json` — accuracy score
- `data/plots/confusion_matrix.png` — visual results

---

## What is DVC?
DVC (Data Version Control) automates the ML pipeline.
Each stage has defined inputs and outputs.
If only params.yaml changes, DVC reruns only affected stages.
This saves time and keeps experiments reproducible.
