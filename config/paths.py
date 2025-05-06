# config/paths.py

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Specific paths
TRAIN_DATA_PATH = DATA_DIR / "train_data.jsonl"
VAL_DATA_PATH = DATA_DIR / "val_data.jsonl"
MODEL_CACHE_DIR = MODEL_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories if they don't exist
for path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, MODEL_CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
