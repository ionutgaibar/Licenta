# data_features.py
import os  # Importă modulul 'os' pentru interacțiunea cu sistemul de operare (ex: lucrul cu directoare și căi de fișiere).
import pandas as pd  # Importă biblioteca 'pandas' cu aliasul 'pd', folosită pentru manipularea și analiza datelor tabelare.
import pandas_ta as ta # Importă 'pandas_ta' (Technical Analysis) cu aliasul 'ta' pentru calculul facil al indicatorilor financiari.
from sklearn.preprocessing import StandardScaler  # Importă clasa 'StandardScaler' din scikit-learn pentru a standardiza (scala) datele numerice.

def run_features_pipeline(ticker: str, clean_dir: str, processed_dir: str, start_date: str, end_date: str):  # Definește funcția principală cu parametrii necesari (simbol, directoare intrare/ieșire, date de început/sfârșit).
    """
    Preia datele curățate, calculează indicatorii tehnici (features) și le salvează.  # Descrie acțiunea funcției.
    """ 
    print(f"--- Inițiere Pipeline de Feature Engineering pentru {ticker} ---")  # Afișează în consolă începutul procesului de creare a feature-urilor pentru simbolul specificat.

    if not os.path.exists(processed_dir):  # Verifică dacă directorul de ieșire NU există deja.
        os.makedirs(processed_dir)  # Creează directorul de ieșire (și directoarele părinte, dacă lipsesc).
        print(f"Directorul {processed_dir} a fost creat.")  # Afișează un mesaj confirmând crearea directorului.

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Construiește numele fișierului folosind simbolul și intervalul de date.
    input_file_path = os.path.join(clean_dir, file_name)  # Construiește calea completă către fișierul de intrare combinând directorul cu numele fișierului.

    if not os.path.exists(input_file_path):  # Verifică dacă fișierul de intrare (datele curățate) lipsește.
        print(f"Eroare: Nu am găsit fișierul curățat '{file_name}' în '{clean_dir}'.")  # Afișează o eroare dacă fișierul nu este găsit.
        return  # Oprește execuția funcției deoarece nu avem date de procesat.

    try:  # Începe blocul de cod care gestionează posibilele erori de execuție.
        print(f"Calculez indicatorii pentru: {file_name}...")  # Afișează mesajul că începe procesarea indicatorilor pentru fișierul curent.
        df = pd.read_csv(input_file_path)  # Citește fișierul CSV de intrare și îl încarcă într-un DataFrame pandas numit 'df'.

        # Ne asigurăm că datele sunt sortate cronologic (esențial pentru indicatori!)
        df = df.sort_values(by='Date').reset_index(drop=True)  # Sortează rândurile după coloana 'Date' crescător și resetează indexul tabelului.

        # --- 1. Indicatori de bază (pentru calcule ulterioare) ---
        # 1. Randament Procentual Zilnic (Close vs Close ieri)
        # pct_change() returnează valori de tip 0.02 (pentru 2%), înmulțim cu 100 pentru vizibilitate
        df['Return_Pct'] = df['Close'].pct_change() * 100  # Creează coloana 'Return_Pct' calculând diferența procentuală a prețului de închidere față de ziua anterioară.

        # 2. Indicatori de Trend
        df.ta.ema(length=20, append=True)  # Calculează Exponential Moving Average (EMA) pe 20 de zile și adaugă direct coloana în DataFrame.
        df.ta.ema(length=50, append=True)  # Calculează Exponential Moving Average (EMA) pe 50 de zile și o adaugă în DataFrame.

        # 3. Volatilitate: Bollinger Bands pe 20 de perioade și 2 deviații standard
        df.ta.bbands(length=20, std=2, append=True)  # Calculează benzile Bollinger (inferioară, medie, superioară, etc.) pe 20 zile și 2 deviații standard.

        # 4. Volatilitate: Average True Range (ATR) pe 14 perioade
        df.ta.atr(length=14, append=True)  # Calculează indicatorul ATR pentru măsurarea volatilității pe 14 zile și îl adaugă.

        # 5. Momentum: MACD (setări clasice 12, 26, 9)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)  # Calculează Moving Average Convergence Divergence (MACD) cu parametrii standard.

        # 6. Trend Strength: ADX pe 14 perioade
        df.ta.adx(length=14, append=True)  # Calculează Average Directional Index (ADX) pentru puterea trendului pe 14 zile.

        # 7. Momentum: RSI pe 14 perioade
        df.ta.rsi(length=14, append=True)  # Calculează Relative Strength Index (RSI) pentru momentum pe 14 zile.

        # 8. Volum: On-Balance Volume (Cumulativ)
        df.ta.obv(append=True)  # Calculează On-Balance Volume (OBV) utilizând fluxul de volum.

        # --- 2. Feature Engineering Ortogonal (Rații) ---
        print("  -> Transform prețurile absolute în rații...")  # Afișează mesajul informativ despre calculul rațiilor.
        # Cât de departe este prețul față de medii?
        df['Dist_EMA_20'] = (df['Close'] / df['EMA_20']) - 1  # Calculează distanța procentuală a prețului de închidere față de media exponențială de 20 de zile.
        df['Dist_EMA_50'] = (df['Close'] / df['EMA_50']) - 1  # Calculează distanța procentuală a prețului de închidere față de media exponențială de 50 de zile.

        # Intersecția mediilor (Cât de departe e EMA 20 de EMA 50?)
        df['EMA_20_50_Ratio'] = (df['EMA_20'] / df['EMA_50']) - 1  # Creează o rație care exprimă raportul dintre EMA 20 și EMA 50 pentru a detecta încrucișările.

        # OBV: Folosim ROC pe 5 zile pentru a vedea impulsul volumului
        df['OBV_ROC_5'] = df['OBV'].pct_change(periods=5) * 100  # Calculează Rate of Change (ROC) pentru OBV pe o fereastră de 5 zile.

        # Volum relativ (Volumul de azi vs Media pe 14 zile)
        vol_ma_14 = df['Volume'].rolling(window=14).mean()  # Calculează media mobilă simplă a volumului pe ultimele 14 zile.
        df['Relative_Volume'] = df['Volume'] / vol_ma_14  # Creează un indicator de volum relativ împărțind volumul curent la media sa pe 14 zile.

        # --- 3. Definirea Target-ului (Direcția Mâine) ---
        # Ne interesează dacă prețul VA CREȘTE în ziua următoare.
        # .shift(-1) mută valoarea de mâine pe rândul de azi.
        # Astfel, modelul se uită la indicatorii de AZI pentru a prezice rezultatul de MÂINE.
        df['Target_Direction'] = (df['Return_Pct'].shift(-1) > 0).astype(int)  # Etichetează ziua curentă cu 1 dacă randamentul de mâine e pozitiv (creștere) și 0 în caz contrar.

        # --- 4. Curățenia de Primăvară (Drop Absolute & Leakage) ---
        # Salvăm coloana Date pentru referință, dar aruncăm restul datelor absolute
        cols_to_drop = [  # Inițializează o listă cu numele coloanelor care trebuie eliminate.
            'Open', 'High', 'Low', 'Close', 'Volume',  # Specifică datele brute de preț și volum care nu trebuie date direct modelului.
            'EMA_20', 'EMA_50',  # Specifică mediile absolute de eliminat, deoarece am creat deja rații bazate pe ele.
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV' # Păstrăm doar BBB și BBP din setul de benzi
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')  # Elimină efectiv coloanele din listă, doar dacă ele există în DataFrame.

        # --- PAS CRITIC: Curățarea rândurilor invalide (NaNs) ---
        # De exemplu, EMA 50 are nevoie de 50 de zile pentru a calcula prima valoare.
        # Asta înseamnă că primele 49 de rânduri din dataset vor avea 'NaN' la coloana EMA_50.
        # Modelele de ML se vor bloca dacă văd NaNs.
        initial_rows = len(df)  # Salvează numărul total de rânduri înainte de curățare.
        df = df.dropna().reset_index(drop=True)  # Elimină toate rândurile care conțin cel puțin o valoare lipsă (NaN) și resetează indexul.
        dropped_rows = initial_rows - len(df)  # Calculează numărul de rânduri eliminate.
        print(f"  -> Am șters primele {dropped_rows} rânduri (warm-up period) pentru indicatori.")  # Afișează numărul de rânduri eliminate din perioada de warm-up.

        # 6. STANDARDIZARE (Scaling)
        # Nu scalăm coloanele 'Date' și 'Target_Direction'
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target_Direction']]  # Creează o listă cu toate coloanele numerice ce trebuie standardizate, excluzând data și target-ul.
        scaler = StandardScaler()  # Instanțiază obiectul StandardScaler pentru a normaliza datele.
        
        df[feature_cols] = scaler.fit_transform(df[feature_cols])  # Aplică standardizarea pe coloanele selectate și suprascrie valorile vechi cu cele scalate.

        print(f"  -> Scalare finalizată pentru {len(feature_cols)} indicatori.")  # Confirmă în consolă numărul de coloane care au fost scalate.

        # Salvarea fișierului final
        output_file_path = os.path.join(processed_dir, file_name)  # Construiește calea completă pentru fișierul de ieșire.
        df.to_csv(output_file_path, index=False)  # Salvează DataFrame-ul final pe disc în format CSV, fără index.

        print(f"  -> Succes! Dataset-ul cu {len(df.columns)} coloane a fost salvat în: {output_file_path}")  # Afișează un mesaj de succes la finalizarea salvării.

    except Exception as e:  # Prinde orice excepție (eroare) care a apărut în blocul 'try'.
        print(f"  -> Eroare la procesarea feature-urilor: {e}")  # Afișează mesajul de eroare exact pentru diagnosticare.