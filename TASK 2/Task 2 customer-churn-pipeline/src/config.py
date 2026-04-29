# src/config.py
from pathlib import Path

# ─────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────
TASK_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────
# Data Paths
# ─────────────────────────────────────────
DATA_DIR = TASK_ROOT / "Data"
RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ─────────────────────────────────────────
# Model & Output Paths
# ─────────────────────────────────────────
MODEL_DIR = TASK_ROOT / "models"
PLOTS_DIR = TASK_ROOT / "models" / "plots"
PIPELINE_PATH = MODEL_DIR / "churn_pipeline.pkl"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve.png"

# ─────────────────────────────────────────
# Auto-create directories if missing
# ─────────────────────────────────────────
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)