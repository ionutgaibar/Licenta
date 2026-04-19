import config
from src.data_loader import run_loader_pipeline
from src.data_cleaner import run_cleaner_pipeline
from src.data_features import run_features_pipeline
from models.logreg import run_logreg_pipeline
from models.xgboost import run_xgboost_pipeline
from models.lstm import run_lstm_pipeline
from models.svm import run_svm_pipeline

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
    cleared_dir=config.CLEAN_DATA_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 3. PROCESEAZA DATELE (Processor)
run_features_pipeline(
        ticker=config.TICKER,
        input_dir=config.CLEAN_DATA_DIR, 
        output_dir=config.PROCESSED_DATA_DIR,
        start_date=config.START_DATE,
        end_date=config.END_DATE
)
# 4. LogReg
run_logreg_pipeline(
    ticker=config.TICKER,
    input_dir=config.PROCESSED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 5. XGBoost
run_xgboost_pipeline(
    ticker=config.TICKER,
    input_dir=config.PROCESSED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)

# 6. LSTM
run_lstm_pipeline(
    ticker=config.TICKER,
    input_dir=config.PROCESSED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    time_steps=10  # Se uită la ultimele 2 săptămâni (10 zile de tranzacționare)
)

#7. SVM
run_svm_pipeline(
    ticker=config.TICKER,
    input_dir=config.PROCESSED_DATA_DIR, 
    model_dir=config.MODELS_DIR,
    start_date=config.START_DATE,
    end_date=config.END_DATE
)