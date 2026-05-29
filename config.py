import os
from pathlib import Path

# Setăm directorul de bază al proiectului (pentru a avea căi absolute flexibile)
BASE_DIR = Path(__file__).resolve().parent

# --- PARAMETRI FINANCIARI ---
TICKER = "SPY"
START_DATE = "1993-01-01"
END_DATE = "2026-01-01"

# --- PARAMETRI MODELE ---
LSTM_TIME_STEPS = 10  # 10 zile de tranzacționare (2 săptămâni)

# --- DIRECTOARE DE DATE ---
# Construim căile folosind operatorul / din modulul pathlib
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR = BASE_DIR / "data" / "clean"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SCALED_DATA_DIR = BASE_DIR / "data" / "scaled"
MODELS_DIR = BASE_DIR / "models"

# (Opțional) Crearea automată a folderelor dacă nu există, 
# ca să nu primești eroare la prima rulare a scriptului
for dir_path in [RAW_DATA_DIR, CLEAN_DATA_DIR, PROCESSED_DATA_DIR, SCALED_DATA_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)