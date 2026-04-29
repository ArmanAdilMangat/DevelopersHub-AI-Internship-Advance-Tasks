# src/config.py
# Centralized configuration — single source of truth for all
# paths, hyperparameters, and constants across the project.

from pathlib import Path
import torch
import os

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR  = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR   = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 4

# ── Tokenization ───────────────────────────────────────────────
MAX_LENGTH = 128

# ── Training Hyperparameters ───────────────────────────────────
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01

# ── Label Mapping ──────────────────────────────────────────────
LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# ── Device Auto-detection ──────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── HuggingFace Cache → project data folder ───────────────────
os.environ["HF_DATASETS_CACHE"] = str(DATA_DIR)


# ── Sanity Check ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Root  : {PROJECT_ROOT}")
    print(f"Data  : {DATA_DIR}")
    print(f"Model : {MODEL_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Labels: {LABEL_MAP}")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"