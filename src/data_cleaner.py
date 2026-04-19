import os
import pandas as pd

def run_cleaner_pipeline(ticker: str, raw_dir: str, cleared_dir: str, start_date: str, end_date: str):
    """
    Caută un fișier specific în raw_dir pe baza parametrilor, îl curăță și îl salvează.
    
    Args:
        ticker (str): Simbolul bursier (ex: 'AAPL').
        raw_dir (str): Directorul sursă (raw data).
        cleared_dir (str): Directorul destinație (cleared data).
        start_date (str): Data de început format 'YYYY-MM-DD'.
        end_date (str): Data de sfârșit format 'YYYY-MM-DD'.
    """
    
    print(f"--- Inițiere Pipeline de Curățare pentru {ticker} ---")
    
    # 1. Asigurarea existenței directorului destinație
    if not os.path.exists(cleared_dir):
        os.makedirs(cleared_dir)
        print(f"Directorul {cleared_dir} a fost creat.")

    # 2. Reconstruirea numelui exact al fișierului
    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    raw_file_path = os.path.join(raw_dir, file_name)

    # 3. Verificăm proactiv dacă fișierul există înainte de a-l citi
    if not os.path.exists(raw_file_path):
        print(f"Eroare: Nu am găsit fișierul '{file_name}' în folderul '{raw_dir}'.")
        print("Te rog să te asiguri că run_loader_pipeline a fost rulat cu aceiași parametri.")
        return

    try:
        print(f"Procesez fișierul: {file_name}...")
        
        # Încărcăm fișierul CSV
        df = pd.read_csv(raw_file_path)

        # --- PASUL A: Ștergerea coloanelor nedorite ---
        cols_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # --- PASUL B: Curățarea coloanei 'Date' ---
        if 'Date' in df.columns:
            # Păstrăm doar formatul YYYY-MM-DD
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')

        # 4. Salvarea fișierului curățat
        cleared_file_path = os.path.join(cleared_dir, file_name)
        df.to_csv(cleared_file_path, index=False) 
        
        print(f"  -> Succes! Fișierul curățat a fost salvat în: {cleared_file_path}")

    except Exception as e:
        print(f"  -> Eroare la procesarea fișierului {file_name}: {e}")