from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ARCHIVO_DATOS = RAW_DIR / "wdbc.data"
ARCHIVO_NOMBRES = EXTERNAL_DIR / "wdbc.names"

TARGET = "diagnosis"
SEMILLA = 42
TAM_PRUEBA = 0.2
FOLDS = 5

for carpeta in [EXTERNAL_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
  carpeta.mkdir(parents=True, exist_ok=True)