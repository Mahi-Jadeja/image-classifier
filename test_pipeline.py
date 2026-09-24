# test_pipeline.py
#
# Unit tests for data_collection.py and data_processing.py.
#
# Assumes this file lives in the same directory as data_collection.py and
# data_processing.py (a typical repo layout). Both modules read a
# "params.yaml" file from the current working directory at import time, so
# each test runs in its own temp directory with a freshly written
# params.yaml, and (re)imports the modules against it.
#
# Run with:
#   pytest test_pipeline.py -v

import io
import os
import sys
import importlib
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import yaml
from PIL import Image

# Make sure the modules under test are importable regardless of where
# pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This warning comes from urllib3 (a `requests` dependency) noticing the
# local Python build uses LibreSSL instead of OpenSSL 1.1.1+ — an
# environment quirk, not something these tests can or should act on.
# Plain `warnings.filterwarnings()` calls get reset by pytest's per-test
# warning capture, so this has to go through pytest's own mechanism.
pytestmark = pytest.mark.filterwarnings(
    "ignore:urllib3 v2 only supports OpenSSL.*:Warning"
)


# ---------------------------------------------------------------------------
# Shared test configuration / helpers
# ---------------------------------------------------------------------------

TEST_PARAMS = {
    "data": {
        "dataset_url": "http://example.com/flowers.tgz",
        "raw_dir": "raw",
        "classes": ["rose", "tulip"],
        "max_images_per_class": 3,
    },
    "processing": {
        "image_size": 16,
        "processed_dir": "processed",
        "validation_split": 0.25,
    },
}


def make_image(path, size=(20, 20), color=(255, 0, 0)):
    """Write a small valid PNG/JPEG image to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_dir(tmp_path, monkeypatch):
    """Chdir into an isolated temp dir with a params.yaml written into it."""
    monkeypatch.chdir(tmp_path)
    with open("params.yaml", "w") as f:
        yaml.safe_dump(TEST_PARAMS, f)
    yield tmp_path


@pytest.fixture()
def data_collection(project_dir):
    """(Re)import data_collection against the fixture's params.yaml."""
    if "data_collection" in sys.modules:
        module = importlib.reload(sys.modules["data_collection"])
    else:
        import data_collection as module
    return module


@pytest.fixture()
def data_processing(project_dir):
    """(Re)import data_processing against the fixture's params.yaml."""
    if "data_processing" in sys.modules:
        module = importlib.reload(sys.modules["data_processing"])
    else:
        import data_processing as module
    return module


# ---------------------------------------------------------------------------
# data_collection.py
# ---------------------------------------------------------------------------

@patch("data_collection.tarfile.open")
@patch("data_collection.requests.get")
def test_download_dataset_downloads_and_extracts_when_missing(
    mock_get, mock_tarfile_open, data_collection
):
    dc = data_collection

    fake_response = MagicMock()
    fake_response.raw = io.BytesIO(b"fake-tar-bytes")
    mock_get.return_value = fake_response

    mock_tar = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar

    dc.download_dataset()

    mock_get.assert_called_once_with(dc.url, stream=True)

    tar_path = os.path.join(dc.raw_dir, "flowers.tgz")
    assert os.path.exists(tar_path)
    with open(tar_path, "rb") as f:
        assert f.read() == b"fake-tar-bytes"

    mock_tarfile_open.assert_called_once_with(tar_path, "r:gz")
    mock_tar.extractall.assert_called_once_with(dc.raw_dir)


@patch("data_collection.tarfile.open")
@patch("data_collection.requests.get")
def test_download_dataset_skips_download_if_tar_already_present(
    mock_get, mock_tarfile_open, data_collection
):
    dc = data_collection

    os.makedirs(dc.raw_dir, exist_ok=True)
    tar_path = os.path.join(dc.raw_dir, "flowers.tgz")
    with open(tar_path, "wb") as f:
        f.write(b"already-downloaded")

    mock_tar = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar

    dc.download_dataset()

    # Should not hit the network again...
    mock_get.assert_not_called()
    # ...but extraction always runs.
    mock_tarfile_open.assert_called_once_with(tar_path, "r:gz")
    mock_tar.extractall.assert_called_once_with(dc.raw_dir)

    # Existing tar file must be left untouched.
    with open(tar_path, "rb") as f:
        assert f.read() == b"already-downloaded"


def test_organize_classes_limits_images_per_class(data_collection):
    dc = data_collection

    source_root = os.path.join(dc.raw_dir, "flower_photos")
    for cls in dc.classes:
        cls_dir = os.path.join(source_root, cls)
        for i in range(5):  # more than max_images_per_class (3)
            make_image(os.path.join(cls_dir, f"{cls}_{i}.png"))

    dc.organize_classes()

    for cls in dc.classes:
        dst_dir = os.path.join(dc.raw_dir, cls)
        assert os.path.isdir(dst_dir)
        copied = os.listdir(dst_dir)
        assert len(copied) == dc.max_images


def test_organize_classes_copies_expected_filenames(data_collection):
    dc = data_collection

    source_root = os.path.join(dc.raw_dir, "flower_photos")
    cls = dc.classes[0]
    cls_dir = os.path.join(source_root, cls)
    make_image(os.path.join(cls_dir, "only_image.png"))

    # Make sure the other declared class also has a (empty-ish) source so
    # organize_classes() doesn't blow up on a missing folder.
    for other_cls in dc.classes[1:]:
        make_image(os.path.join(source_root, other_cls, "placeholder.png"))

    dc.organize_classes()

    dst_dir = os.path.join(dc.raw_dir, cls)
    assert os.path.exists(os.path.join(dst_dir, "only_image.png"))


# ---------------------------------------------------------------------------
# data_processing.py
# ---------------------------------------------------------------------------

def test_load_images_shapes_and_labels(data_processing):
    dp = data_processing

    images_per_class = 4
    for label, cls in enumerate(dp.classes):
        cls_dir = os.path.join(dp.raw_dir, cls)
        for i in range(images_per_class):
            # Deliberately non-square source images to confirm resizing.
            make_image(os.path.join(cls_dir, f"{cls}_{i}.png"), size=(50, 30))

    X, y = dp.load_images()

    total_images = images_per_class * len(dp.classes)
    assert X.shape == (total_images, dp.image_size, dp.image_size, 3)
    assert y.shape == (total_images,)
    assert set(y.tolist()) == set(range(len(dp.classes)))
    for label in range(len(dp.classes)):
        assert (y == label).sum() == images_per_class


def test_load_images_skips_unreadable_files(data_processing):
    dp = data_processing

    cls = dp.classes[0]
    cls_dir = os.path.join(dp.raw_dir, cls)
    make_image(os.path.join(cls_dir, "good.png"))

    with open(os.path.join(cls_dir, "corrupt.png"), "w") as f:
        f.write("this is not a real image")

    # load_images() iterates over every declared class, so each one needs
    # an existing (even if empty) folder or os.listdir() will raise.
    for other_cls in dp.classes[1:]:
        os.makedirs(os.path.join(dp.raw_dir, other_cls), exist_ok=True)

    X, y = dp.load_images()

    # The corrupt file should be silently skipped, not raise.
    assert len(X) == 1
    assert len(y) == 1


def test_split_and_save_ratio_and_pairing_preserved(data_processing):
    dp = data_processing

    n = 20
    # Encode each sample's index into the pixel value so we can verify
    # X/y pairing survives the shuffle-then-split.
    X = np.arange(n, dtype=np.uint8).reshape(n, 1, 1, 1)
    y = np.arange(n)

    dp.split_and_save(X, y)

    x_train = np.load(os.path.join(dp.processed_dir, "X_train.npy"))
    x_val = np.load(os.path.join(dp.processed_dir, "X_val.npy"))
    y_train = np.load(os.path.join(dp.processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(dp.processed_dir, "y_val.npy"))

    expected_val = int(n * dp.val_split)
    expected_train = n - expected_val

    assert len(x_train) == expected_train
    assert len(x_val) == expected_val
    assert len(y_train) == expected_train
    assert len(y_val) == expected_val

    # Every (X, y) pair must still match after shuffling, and every
    # original sample must appear exactly once across train + val.
    combined = {}
    for x_val_arr, label in zip(x_train.reshape(-1), y_train):
        combined[int(x_val_arr)] = int(label)
    for x_val_arr, label in zip(x_val.reshape(-1), y_val):
        combined[int(x_val_arr)] = int(label)

    assert combined == {i: i for i in range(n)}


def test_split_and_save_creates_processed_dir(data_processing):
    dp = data_processing
    assert not os.path.exists(dp.processed_dir)

    X = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    y = np.arange(8)
    dp.split_and_save(X, y)

    assert os.path.isdir(dp.processed_dir)
    for fname in ("X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"):
        assert os.path.exists(os.path.join(dp.processed_dir, fname))