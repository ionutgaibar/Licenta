# main.py
import config
from src.data_loader import run_loader_pipeline
from src.data_cleaner import run_cleaner_pipeline
from src.data_features import run_features_pipeline
from src.data_scaler import run_scaler_pipeline
from models.logreg import run_logreg_pipeline
from models.xgboost import run_xgboost_pipeline
from models.lstm import run_lstm_pipeline
from models.svm import run_svm_pipeline
from src.backtester import run_backtester



# 1. DESCARCĂ DATELE (Loader)
run_loader_pipeline(
    ticker=config.TICKER, 
    raw_dir=config.RAW_DATA_DIR, 
    start_date=config.START_DATE, 
    end_date=config.END_DATE
)

# 2. CURATA DATELE (Cleaner)
run_cleaner_pipeline(
    ticker=config.TICKER,
    raw_dir=config.RAW_DATA_DIR, 
    clean_dir=config.CLEAN_DATA_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 3. PROCESEAZA DATELE (Processor)
run_features_pipeline(
    ticker=config.TICKER,
    clean_dir=config.CLEAN_DATA_DIR, 
    processed_dir=config.PROCESSED_DATA_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)
# 4. SCALEAZA DATELE (Scaler)
run_scaler_pipeline(
    ticker=config.TICKER,
    processed_dir=config.PROCESSED_DATA_DIR,
    scaled_dir=config.SCALED_DATA_DIR,
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 4. LogReg
run_logreg_pipeline(
    ticker=config.TICKER,
    scaled_dir=config.SCALED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 5. XGBoost
run_xgboost_pipeline(
    ticker=config.TICKER,
    scaled_dir=config.SCALED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 6. LSTM
run_lstm_pipeline(
    ticker=config.TICKER,
    scaled_dir=config.SCALED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    time_steps=config.LSTM_TIME_STEPS
)

# 7. SVM
run_svm_pipeline(
    ticker=config.TICKER,
    scaled_dir=config.SCALED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

#8. Backtest
run_backtester(
    ticker=config.TICKER,
    clean_dir=config.CLEAN_DATA_DIR,
    scaled_dir=config.SCALED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)