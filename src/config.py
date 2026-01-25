"""
Configuration globale du projet de scoring de crédit.
Chemins, paramètres et constantes.
"""

from pathlib import Path

# Chemins racine
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Chemins données
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Chemins outputs
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Fichier de données principal
RAW_DATA_FILE = RAW_DATA_DIR / "credit_scoring_synth_130k_with_date_defaults.csv"

# Variable cible
TARGET_COLUMN = "TARGET"

# Colonne temporelle pour le split
DATE_COLUMN = "DATE_MONTH"

# Paramètres de modélisation
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paramètres de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
