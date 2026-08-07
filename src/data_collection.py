# data_collection.py
# downloads the flower dataset and puts images into class folders

import os
import tarfile
import requests
import shutil
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

url = params["data"]["dataset_url"]
raw_dir = params["data"]["raw_dir"]
classes = params["data"]["classes"]
max_images = params["data"]["max_images_per_class"]


def download_dataset():
    os.makedirs(raw_dir, exist_ok=True)
    tar_path = os.path.join(raw_dir, "flowers.tgz")

    if not os.path.exists(tar_path):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print("Download done.")

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)
    print("Extraction done.")


def organize_classes():
    source = os.path.join(raw_dir, "flower_photos")

    for cls in classes:
        src_folder = os.path.join(source, cls)
        dst_folder = os.path.join(raw_dir, cls)
        os.makedirs(dst_folder, exist_ok=True)

        images = os.listdir(src_folder)[:max_images]
        for img in images:
            shutil.copy(
                os.path.join(src_folder, img),
                os.path.join(dst_folder, img)
            )

    print(f"Done — {len(classes)} classes, {max_images} images each.")


if __name__ == "__main__":
    download_dataset()
    organize_classes()
