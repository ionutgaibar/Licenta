import os
import pandas as pd
import pandas_ta as ta  # Extensia care face magia tehnică
from sklearn.preprocessing import StandardScaler

def run_features_pipeline(ticker: str, input_dir: str, output_dir: str, start_date: str, end_date: str):
    """
    Preia datele curățate, calculează indicatorii tehnici (features) și le salvează.
    """
    print(f"--- Inițiere Pipeline de Feature Engineering pentru {ticker} ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorul {output_dir} a fost creat.")

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    input_file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_file_path):
        print(f"Eroare: Nu am găsit fișierul curățat '{file_name}' în '{input_dir}'.")
        return

    try:
        print(f"Calculez indicatorii pentru: {file_name}...")
        df = pd.read_csv(input_file_path)

        # Ne asigurăm că datele sunt sortate cronologic (esențial pentru indicatori!)
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # --- 1. Indicatori de bază (pentru calcule ulterioare) ---
        # 1. Randament Procentual Zilnic (Close vs Close ieri)
        # pct_change() returnează valori de tip 0.02 (pentru 2%), înmulțim cu 100 pentru vizibilitate
        df['Return_Pct'] = df['Close'].pct_change() * 100

        # 2. Indicatori de Trend
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)

        # 3. Volatilitate: Bollinger Bands pe 20 de perioade și 2 deviații standard
        df.ta.bbands(length=20, std=2, append=True)

        # 4. Volatilitate: Average True Range (ATR) pe 14 perioade
        df.ta.atr(length=14, append=True)

        # 5. Momentum: MACD (setări clasice 12, 26, 9)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # 6. Trend Strength: ADX pe 14 perioade
        df.ta.adx(length=14, append=True)

        # 7. Momentum: RSI pe 14 perioade
        df.ta.rsi(length=14, append=True)

        # 8. Volum: On-Balance Volume (Cumulativ)
        df.ta.obv(append=True)
        
        # --- 2. Feature Engineering Ortogonal (Rații) ---
        print("  -> Transform prețurile absolute în rații...")
        # Cât de departe este prețul față de medii?
        df['Dist_EMA_20'] = (df['Close'] / df['EMA_20']) - 1
        df['Dist_EMA_50'] = (df['Close'] / df['EMA_50']) - 1
        
        # Intersecția mediilor (Cât de departe e EMA 20 de EMA 50?)
        df['EMA_20_50_Ratio'] = (df['EMA_20'] / df['EMA_50']) - 1

        # OBV: Folosim ROC pe 5 zile pentru a vedea impulsul volumului
        df['OBV_ROC_5'] = df['OBV'].pct_change(periods=5) * 100
        
        # Volum relativ (Volumul de azi vs Media pe 14 zile)
        vol_ma_14 = df['Volume'].rolling(window=14).mean()
        df['Relative_Volume'] = df['Volume'] / vol_ma_14

        # --- 3. Definirea Target-ului (Direcția Mâine) ---
        # Ne interesează dacă prețul VA CREȘTE în ziua următoare.
        # .shift(-1) mută valoarea de mâine pe rândul de azi.
        # Astfel, modelul se uită la indicatorii de AZI pentru a prezice rezultatul de MÂINE.
        df['Target_Direction'] = (df['Return_Pct'].shift(-1) > 0).astype(int)

        # --- 4. Curățenia de Primăvară (Drop Absolute & Leakage) ---
        # Salvăm coloana Date pentru referință, dar aruncăm restul datelor absolute
        cols_to_drop = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'EMA_20', 'EMA_50', 
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV' # Păstrăm doar BBB și BBP din setul de benzi
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        # --- PAS CRITIC: Curățarea rândurilor invalide (NaNs) ---
        # De exemplu, EMA 50 are nevoie de 50 de zile pentru a calcula prima valoare.
        # Asta înseamnă că primele 49 de rânduri din dataset vor avea 'NaN' la coloana EMA_50.
        # Modelele de ML se vor bloca dacă văd NaNs.
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_rows - len(df)
        print(f"  -> Am șters primele {dropped_rows} rânduri (warm-up period) pentru indicatori.")

        # 6. STANDARDIZARE (Scaling)
        # Nu scalăm coloanele 'Date' și 'Target_Direction'
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target_Direction']]
        
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        print(f"  -> Scalare finalizată pentru {len(feature_cols)} indicatori.")

        # Salvarea fișierului final
        output_file_path = os.path.join(output_dir, file_name)
        df.to_csv(output_file_path, index=False) 
        
        print(f"  -> Succes! Dataset-ul cu {len(df.columns)} coloane a fost salvat în: {output_file_path}")

    except Exception as e:
        print(f"  -> Eroare la procesarea feature-urilor: {e}")