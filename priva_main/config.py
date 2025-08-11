from pathlib import Path

APP_NAME = "Project Sherlock"
APP_VERSION = "2.1"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"      # put .pth weights here
TEMP_DIR = ROOT_DIR / "temp"
REPORT_DIR = ROOT_DIR / "reports"     # kept for future use

for p in (DATA_DIR, MODELS_DIR, TEMP_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

SESSION_EXPIRY_DAYS = 7
PASSWORD_SALT = "deepfake_detection_secure_salt_2025"

MAX_FILE_SIZE_MB = 100
MAX_HISTORY_ITEMS = 100
