"""Shared constants and helpers for the multi-dataset facial emotion model."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 42

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# code/ sits directly under project/, which sits directly under the repo root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"

# Windows + multiprocessing workers are unreliable here, so keep loaders in-process.
NUM_WORKERS = 0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def project_path(*parts: str) -> Path:
    return PROJECT_DIR.joinpath(*parts)


def dataset_path(name: str) -> Path:
    """Canonical on-disk location for each dataset, per datasets/manifest.json."""
    paths = {
        "fer2013": DATASETS_DIR / "FER-2013" / "extracted",
        "ferplus": DATASETS_DIR / "ferplus",
        "jaffe": DATASETS_DIR / "jaffe" / "extracted",
        "ravdess": DATASETS_DIR / "RAVDESS",
        "crema_d": DATASETS_DIR / "crema_d",
    }
    if name not in paths:
        raise KeyError(f"unknown dataset: {name}")
    return paths[name]


def read_state() -> dict:
    state_file = project_path("state.json")
    return json.loads(state_file.read_text(encoding="utf-8"))


def write_state(state: dict) -> None:
    project_path("state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_dirs() -> None:
    for sub in ("code", "data", "logs", "models", "results", "reports"):
        project_path(sub).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("classes:", CLASSES)
    print("device:", get_device())
