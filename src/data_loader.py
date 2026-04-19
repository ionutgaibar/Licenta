import os
import yfinance as yf

def run_loader_pipeline(ticker: str, raw_dir: str, start_date: str, end_date: str):
    """
    Descarcă date istorice folosind yfinance și le salvează local.
    Sare peste descărcare dacă fișierul există deja.
    
    Args:
        ticker (str): Simbolul bursier (ex: 'AAPL').
        raw_dir (str): Calea către directorul unde se salvează datele.
        start_date (str): Data de început format 'YYYY-MM-DD'.
        end_date (str): Data de sfârșit format 'YYYY-MM-DD'.
    """
    
    print(f"--- Inițiere Pipeline pentru {ticker} ---")
    
    # 1. Asigurarea existenței directorului
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"Directorul {raw_dir} a fost creat.")

    # 2. Construire nume și cale fișier în avans
    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    file_path = os.path.join(raw_dir, file_name)

    # 3. VERIFICARE IDEMPOTENȚĂ: Există deja fișierul?
    if os.path.exists(file_path):
        print(f"  -> Fișierul '{file_name}' există deja în '{raw_dir}'.")
        print("  -> Trec peste descărcare pentru a economisi resurse (Skip).")
        return # Ieșim din funcție, nu mai mergem mai departe

    try:
        print(f"  -> Descarc datele pentru {ticker} de la API...")
        # 4. Inițializare obiect Ticker și descărcare
        asset = yf.Ticker(ticker)
        df = asset.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            print(f"  -> Atenție: Nu s-au găsit date pentru {ticker} în perioada selectată.")
            return
        
        # 5. Salvare date
        df.to_csv(file_path)
        print(f"  -> Succes! Datele au fost descărcate și salvate în: {file_path}")
        print(f"  -> Total rânduri noi salvate: {len(df)}")

    except Exception as e:
        print(f"  -> Eroare în timpul descărcării de la yfinance: {e}")