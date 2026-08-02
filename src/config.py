from pathlib import Path

# ============================
# Project Root
# ============================

ROOT_DIR = Path(__file__).resolve().parent.parent

# ============================
# Data
# ============================

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ============================
# Models
# ============================

MODELS_DIR = ROOT_DIR / "models"

MODEL_DIR = MODELS_DIR

TRAINED_MODELS_DIR = MODELS_DIR / "trained"

# ============================
# Outputs
# ============================


OUTPUTS_DIR = ROOT_DIR / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"

REPORTS_DIR = OUTPUTS_DIR / "reports"

DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"

# ============================
# Logs
# ============================

LOG_DIR = ROOT_DIR / "logs"

# ============================
# Create folders automatically
# ============================

for folder in [

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

   MODELS_DIR,
   TRAINED_MODELS_DIR,

    OUTPUTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    DASHBOARD_DIR,
    DASHBOARD_DIR,

    LOG_DIR

]:

    folder.mkdir(parents=True, exist_ok=True)