# run_experiments.py
# runs 20 dvc experiments with different hyperparameter combinations

import subprocess

# 20 experiment configurations — varying 4 key hyperparameters
experiments = [
    {"n_estimators": 50,  "max_depth": 5,  "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 50,  "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 50,  "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 5,  "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 2},
    {"n_estimators": 150, "max_depth": 5,  "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 150, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 150, "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 5,  "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 2},
    {"n_estimators": 75,  "max_depth": 8,  "min_samples_split": 3, "min_samples_leaf": 1},
    {"n_estimators": 75,  "max_depth": 12, "min_samples_split": 3, "min_samples_leaf": 2},
    {"n_estimators": 125, "max_depth": 8,  "min_samples_split": 3, "min_samples_leaf": 1},
    {"n_estimators": 125, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 175, "max_depth": 8,  "min_samples_split": 3, "min_samples_leaf": 1},
    {"n_estimators": 175, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 250, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 250, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 3},
]

for i, exp in enumerate(experiments):
    print(f"\n--- Running experiment {i+1}/20 ---")
    print(f"    Params: {exp}")

    cmd = [
        "dvc", "exp", "run",
        "--name", f"exp_{i+1}",
        "--set-param", f"training.n_estimators={exp['n_estimators']}",
        "--set-param", f"training.max_depth={exp['max_depth']}",
        "--set-param", f"training.min_samples_split={exp['min_samples_split']}",
        "--set-param", f"training.min_samples_leaf={exp['min_samples_leaf']}",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"    Experiment {i+1} done.")
    else:
        print(f"    Experiment {i+1} failed — check above output.")

print("\nAll 20 experiments complete.")