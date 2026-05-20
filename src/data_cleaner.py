import os  # Importă modulul 'os' pentru interacțiunea cu sistemul de operare (cum ar fi lucrul cu directoare și fișiere).
import pandas as pd  # Importă biblioteca 'pandas' sub aliasul 'pd', folosită pentru procesarea și manipularea datelor.

def run_cleaner_pipeline(ticker: str, raw_dir: str, cleared_dir: str, start_date: str, end_date: str):  # Definește funcția de curățare, specificând tipurile parametrilor așteptați (simbol, directoare, date).
    """
    Caută un fișier specific în raw_dir pe baza parametrilor, îl curăță și îl salvează.

    Args:
        ticker (str): Simbolul bursier (ex: 'AAPL').
        raw_dir (str): Directorul sursă (raw data).
        cleared_dir (str): Directorul destinație (cleared data).
        start_date (str): Data de început format 'YYYY-MM-DD'.
        end_date (str): Data de sfârșit format 'YYYY-MM-DD'.
    """

    print(f"--- Inițiere Pipeline de Curățare pentru {ticker} ---")  # Afișează în consolă un mesaj care semnalizează începerea procesului pentru simbolul curent.

    # 1. Asigurarea existenței directorului destinație
    if not os.path.exists(cleared_dir):  # Verifică dacă directorul unde trebuie salvate fișierele curățate NU există.
        os.makedirs(cleared_dir)  # Dacă nu există, creează directorul destinație (inclusiv pe cele părinte, dacă e necesar).
        print(f"Directorul {cleared_dir} a fost creat.")  # Informează utilizatorul că directorul a fost creat cu succes.

    # 2. Reconstruirea numelui exact al fișierului
    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Generează numele fișierului folosind același format (simbol + date) ca în pipeline-ul de descărcare.
    raw_file_path = os.path.join(raw_dir, file_name)  # Creează calea completă către fișierul brut, unind directorul sursă cu numele fișierului.

    # 3. Verificăm proactiv dacă fișierul există înainte de a-l citi
    if not os.path.exists(raw_file_path):  # Verifică dacă fișierul pe care încercăm să-l curățăm lipsește fizic de pe disc.
        print(f"Eroare: Nu am găsit fișierul '{file_name}' în folderul '{raw_dir}'.")  # Afișează un mesaj de eroare dacă fișierul brut nu este găsit.
        return  # Oprește execuția funcției prematur, deoarece nu are ce date să proceseze.

    try:  # Începe un bloc 'try' pentru a prinde și gestiona eventualele erori (de citire/scriere/procesare).
        print(f"Procesez fișierul: {file_name}...")  # Anunță în consolă că începe efectiv citirea și procesarea fișierului.

        # Încărcăm fișierul CSV
        df = pd.read_csv(raw_file_path)  # Folosește pandas pentru a citi datele din fișierul CSV și le stochează într-un DataFrame numit 'df'.

        # --- PASUL A: Ștergerea coloanelor nedorite ---
        cols_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']  # Definește o listă cu numele coloanelor care nu sunt relevante pentru analiză și trebuie șterse.
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')  # Elimină coloanele specificate din DataFrame, asigurându-se cu 'if col in df.columns' că șterge doar ce există, evitând erorile.

        # --- PASUL B: Curățarea coloanei 'Date'
        if 'Date' in df.columns:  # Verifică preventiv dacă DataFrame-ul conține coloana 'Date' înainte de a încerca să o modifice.
            # Păstrăm doar formatul YYYY-MM-DD
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')  # Convertește coloana 'Date' într-un obiect datetime (formatând-o UTC) și extrage doar șirul de caractere an-lună-zi.

        # 4. Salvarea fișierului curățat
        cleared_file_path = os.path.join(cleared_dir, file_name)  # Formează calea completă unde va fi salvat noul fișier curățat (director destinație + nume fișier).
        df.to_csv(cleared_file_path, index=False)  # Salvează DataFrame-ul procesat înapoi pe disc ca CSV, fără a adăuga coloana cu indexul numeric rândurilor.

        print(f"  -> Succes! Fișierul curățat a fost salvat în: {cleared_file_path}")  # Afișează un mesaj de confirmare după ce fișierul a fost scris cu succes.

    except Exception as e:  # Dacă apare orice eroare în blocul 'try' (ex: fișier corupt, permisiuni), este capturată aici în variabila 'e'.
        print(f"  -> Eroare la procesarea fișierului {file_name}: {e}")  # Afișează mesajul de eroare specific pentru a ajuta la depanare.