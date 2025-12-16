import os
from pathlib import Path

# Base paths
BASE_PATH = Path(__file__).parent.parent

# Dataset paths
DATASET_ROOT_PATH = str(BASE_PATH / "data")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "raw" / "PAKDD2010_Modeling_Data.txt")

DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "raw" / "PAKDD2010_Prediction_Data.txt")

DATASET_DESCRIPTION = str(
    Path(DATASET_ROOT_PATH) / "raw" / "PAKDD2010_VariablesList.XLS"
)

# Models path
MODELS_PATH = str(BASE_PATH / "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Model version management
# Puedes cambiar MODEL_VERSION vía variable de entorno o usar "production" por defecto
MODEL_VERSION = os.getenv("MODEL_VERSION", "production")

# Paths para modelo y preprocessor
# Soporta estructura: models/v1/, models/v2/, models/production/, etc.
MODEL_DIR = Path(MODELS_PATH) / MODEL_VERSION
PROCESSED_DIR = Path(BASE_PATH / "data" / "processed") / MODEL_VERSION

# Crear directorios si no existen
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Si no existe la versión específica, usar modelo directo (compatibilidad hacia atrás)
if not MODEL_DIR.exists() or not (MODEL_DIR / "model.joblib").exists():
    # Fallback a estructura simple: models/model.joblib
    MODEL_FILE = str(Path(MODELS_PATH) / "model.joblib")
else:
    MODEL_FILE = str(MODEL_DIR / "model.joblib")

if not PROCESSED_DIR.exists() or not (PROCESSED_DIR / "preprocessor.joblib").exists():
    # Fallback a estructura simple: data/processed/preprocessor.joblib
    PREPROCESSOR_FILE = str(Path(BASE_PATH / "data" / "processed") / "preprocessor.joblib")
else:
    PREPROCESSOR_FILE = str(PROCESSED_DIR / "preprocessor.joblib")

# Paths adicionales para gestión de modelos
MODELS_BASE_PATH = MODELS_PATH
PROCESSED_BASE_PATH = str(BASE_PATH / "data" / "processed")