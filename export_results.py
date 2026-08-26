# export_results.py
# extracts experiment hyperparameters and accuracy metrics from dvc to csv

import csv
import json
import subprocess


def find_value(obj, key_name):
    # recursively searches for a key anywhere inside nested dicts/lists
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key_name and not isinstance(v, (dict, list)):
                return v
            if isinstance(v, (dict, list)):
                found = find_value(v, key_name)
                if found is not None:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value(item, key_name)
            if found is not None:
                return found
    return None


def collect_experiments(data):
    # finds all experiment objects in the dvc json tree
    experiments = []

    def walk(node, name_hint=""):
        if isinstance(node, dict):
            # check if this dictionary represents an experiment
            name = node.get("name") or node.get("data", {}).get("name") or name_hint
            if name and str(name).startswith("exp_"):
                experiments.append((name, node))
                return

            for k, v in node.items():
                walk(v, name_hint=k)
        elif isinstance(node, list):
            for item in node:
                walk(item, name_hint)

    walk(data)
    return experiments


# run dvc exp show as json
print("Fetching experiment results from DVC...")
result = subprocess.run(
    ["dvc", "exp", "show", "--json"],
    capture_output=True,
    text=True
)

try:
    data = json.loads(result.stdout)
except Exception as e:
    print(f"Error parsing DVC json output: {e}")
    exit(1)

exp_list = collect_experiments(data)
rows = []
seen = set()

for name, exp_data in exp_list:
    if name in seen:
        continue
    seen.add(name)

    # find each metric and parameter value recursively
    n_est = find_value(exp_data, "n_estimators")
    depth = find_value(exp_data, "max_depth")
    mss = find_value(exp_data, "min_samples_split")
    msl = find_value(exp_data, "min_samples_leaf")
    acc = find_value(exp_data, "accuracy")

    rows.append({
        "experiment": name,
        "n_estimators": n_est if n_est is not None else "",
        "max_depth": depth if depth is not None else "",
        "min_samples_split": mss if mss is not None else "",
        "min_samples_leaf": msl if msl is not None else "",
        "accuracy": round(float(acc), 4) if acc is not None else ""
    })

# sort by experiment number (exp_1 to exp_20)
def sort_key(row):
    try:
        return int(row["experiment"].replace("exp_", ""))
    except Exception:
        return 999

rows.sort(key=sort_key)

# write out to csv
csv_file = "experiment_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "experiment", "n_estimators", "max_depth",
        "min_samples_split", "min_samples_leaf", "accuracy"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} experiments to {csv_file}")