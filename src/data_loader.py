import os  # Importă modulul 'os' folosit pentru a interacționa cu sistemul de operare (pentru a verifica și crea directoare/căi).
import yfinance as yf  # Importă biblioteca 'yfinance' și îi atribuie aliasul 'yf', folosită pentru a extrage date de la Yahoo Finance.

def run_loader_pipeline(ticker: str, raw_dir: str, start_date: str, end_date: str):  # Definește funcția cu parametri tipizați (simbolul acțiunii, directorul destinație și intervalul de timp).
    """
    Descarcă date istorice folosind yfinance și le salvează local.
    Sare peste descărcare dacă fișierul există deja.

    Args:
        ticker (str): Simbolul bursier (ex: 'AAPL').
        raw_dir (str): Calea către directorul unde se salvează datele.
        start_date (str): Data de început format 'YYYY-MM-DD'.
        end_date (str): Data de sfârșit format 'YYYY-MM-DD'.
    """

    print(f"--- Inițiere Pipeline pentru {ticker} ---")  # Afișează în consolă un mesaj formatat care anunță începutul procesului pentru simbolul cerut.

    # 1. Asigurarea existenței directorului
    if not os.path.exists(raw_dir):  # Evaluează o condiție: verifică dacă pe disc NU există deja calea specificată în variabila 'raw_dir'.
        os.makedirs(raw_dir)  # Dacă directorul lipsește, îl creează fizic (inclusiv eventualele subdirectoare necesare din structură).
        print(f"Directorul {raw_dir} a fost creat.")  # Afișează un mesaj de confirmare a creării noului director.

    # 2. Construire nume și cale fișier în avans
    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Formează dinamic numele fișierului, concatenând simbolul și datele de început/sfârșit, plus extensia .csv.
    file_path = os.path.join(raw_dir, file_name)  # Combină directorul destinație cu numele fișierului într-o cale completă, corectă pentru orice sistem de operare.

    # 3. VERIFICARE IDEMPOTENȚĂ: Există deja fișierul?
    if os.path.exists(file_path):  # Verifică dacă fișierul pe care vrem să-l creăm a fost deja salvat într-o execuție anterioară.
        print(f"  -> Fișierul '{file_name}' există deja în '{raw_dir}'.")  # Informează utilizatorul că fișierul a fost găsit.
        print("  -> Trec peste descărcare pentru a economisi resurse (Skip).")  # Explică de ce nu se va mai apela API-ul yfinance.
        return # Ieșim din funcție, nu mai mergem mai departe  

    try:  # Începe un bloc de protecție pentru gestionarea erorilor; codul din acest bloc va fi monitorizat pentru posibile excepții (ex: lipsă internet).
        print(f"  -> Descarc datele pentru {ticker} de la API...")  # Afișează un mesaj care indică începerea cererii către serverul Yahoo Finance.
        # 4. Inițializare obiect Ticker și descărcare  
        asset = yf.Ticker(ticker)  # Creează un obiect special Ticker din biblioteca yfinance, bazat pe simbolul cerut, care ne oferă acces la metode de descărcare.
        df = asset.history(start=start_date, end=end_date, auto_adjust=True)  # Apelează metoda 'history' pe obiectul generat anterior pentru a prelua istoricul ca DataFrame pandas, ajustând automat prețurile.

        if df.empty:  # Verifică dacă tabelul (DataFrame) descărcat este complet gol (ceea ce ar însemna că nu s-au găsit cotații în acel interval).
            print(f"  -> Atenție: Nu s-au găsit date pentru {ticker} în perioada selectată.")  # Afișează un avertisment explicit despre lipsa datelor.
            return  # Întrerupe funcția deoarece nu avem ce salva mai departe.

        # 5. Salvare date
        df.to_csv(file_path)  # Salvează tabelul pandas descărcat pe hard disk, la calea stabilită anterior, în format CSV.
        print(f"  -> Succes! Datele au fost descărcate și salvate în: {file_path}")  # Confirmă utilizatorului că salvarea s-a produs cu succes.
        print(f"  -> Total rânduri noi salvate: {len(df)}")  # Calculează numărul total de rânduri (folosind lungimea DataFrame-ului) și îl afișează pentru validare vizuală.

    except Exception as e:  # Dacă oricare dintre liniile din blocul 'try' generează o eroare de execuție, este capturată aici sub variabila 'e'.
        print(f"  -> Eroare în timpul descărcării de la yfinance: {e}")  # Afișează un mesaj care arată că descărcarea a eșuat și tipărește textul exact al erorii pentru depanare.