import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import glob
import math
import tensorflow as tf

def _load_data(features_file, cleaned_file):
    df_features = pd.read_csv(features_file)
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    df_cleaned = pd.read_csv(cleaned_file)
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    
    df_prices = df_cleaned[['Date', 'Close']]
    
    df = pd.merge(df_features, df_prices, on='Date', how='inner')
    return df

def _prepare_test_data(df):
    test_data = df[df['Date'].dt.year >= 2021].copy()
    
    test_data['Daily_Return'] = test_data['Close'].pct_change()
    test_data['Target_Next_Return'] = test_data['Daily_Return'].shift(-1)
    
    test_data = test_data.dropna(subset=['Target_Next_Return'])
    
    cols_to_drop = ['Date', 'Target_Direction', 'Target_Next_Return', 'Daily_Return', 'Close']
    features = test_data.drop(columns=cols_to_drop, errors='ignore')
    
    return test_data, features

def _prepare_lstm_data(features, time_steps):
    features_array = features.values
    X_lstm = []
    
    for i in range(len(features_array) - time_steps + 1):
        X_lstm.append(features_array[i : i + time_steps])
        
    return np.array(X_lstm)

def _generate_signals(model_path, features):
    if model_path.endswith(('.keras', '.h5')):
        model = tf.keras.models.load_model(model_path)
        time_steps = model.input_shape[1]
        
        X_lstm = _prepare_lstm_data(features, time_steps)
        y_pred = model.predict(X_lstm, verbose=0).ravel()
        raw_signals = (y_pred > 0.5).astype(int)
        
        signals = np.zeros(len(features))
        signals[time_steps - 1:] = raw_signals
        return signals
    else:
        model = joblib.load(model_path)
        return model.predict(features)
    
def _compute_metrics(equity_curve, returns, initial_capital):
    total_ret = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(252)
        if returns.std() != 0 else 0
    )
    
    rolling_max = equity_curve.cummax()
    max_dd = ((equity_curve - rolling_max) / rolling_max).min() * 100
    
    return round(total_ret, 2), round(sharpe, 2), round(max_dd, 2)

def _plot_results(test_data, models, initial_capital, ticker):
    num_models = len(models.keys())
    
    cols = 2
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    
    if num_models > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    culori = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    
    for idx, model_name in enumerate(models.keys()):
        ax = axes[idx]
        
        ax.plot(test_data['Date'], test_data['Equity_BuyHold'],
                label='Buy & Hold (Piața)', linestyle='--')
        
        if f'Equity_{model_name}' in test_data.columns:
            ax.plot(test_data['Date'],
                    test_data[f'Equity_{model_name}'],
                    label=f'Strategie: {model_name}')
            
        ax.set_title(f"{model_name} vs B&H")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for i in range(num_models, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

def find_models(base_models_dir):
    """
    Scanează recursiv folderul de modele și returnează un dicționar:
    {'Nume_Model': 'cale/catre/model.extensie'}
    """
    models_dict = {}
    
    # Căutăm toate fișierele .joblib (ML Clasic) și .keras (Deep Learning)
    # Recursiv (**) prin toate subfolderele
    patterns = [
        os.path.join(base_models_dir, "**", "*.joblib"),
        os.path.join(base_models_dir, "**", "*.keras"),
        os.path.join(base_models_dir, "**", "*.h5")
    ]
    
    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            # Extragem numele fișierului fără extensie (ex: 'xgboost_SPY')
            file_name = os.path.basename(file_path)
            model_name = os.path.splitext(file_name)[0]
            
            models_dict[model_name] = file_path
            
    return models_dict

def run_backtest(
    ticker: str,
    cleaned_dir: str,
    features_dir: str,
    model_dir: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0
    ):
    print(f"--- Inițiere Backtesting Multi-Model pentru {ticker} ---")

    models = find_models(model_dir)

    features_file = os.path.join(
        features_dir,
        f"{ticker}_{start_date}_to_{end_date}.csv"
    )

    cleaned_file = os.path.join(
        cleaned_dir,
        f"{ticker}_{start_date}_to_{end_date}.csv"
    )

    df = _load_data(features_file, cleaned_file)
    test_data, features = _prepare_test_data(df)

    test_data['Equity_BuyHold'] = initial_capital * (1 + test_data['Target_Next_Return']).cumprod()

    for model_name, model_path in models.items():
        signals = _generate_signals(model_path, features)

        test_data[f'Signal_{model_name}'] = signals
        test_data[f'Return_{model_name}'] = signals * test_data['Target_Next_Return']
        test_data[f'Equity_{model_name}'] = initial_capital * (1 + test_data[f'Return_{model_name}']).cumprod()

        total_ret, sharpe, max_dd = _compute_metrics(
            test_data[f'Equity_{model_name}'],
            test_data[f'Return_{model_name}'],
            initial_capital
        )

        results = {}
        results[model_name] = {
            'Profit (%)': total_ret,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_dd
        }

    _plot_results(test_data, models, initial_capital, ticker)