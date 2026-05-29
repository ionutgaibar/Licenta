# Importăm 'pandas' (pd) pentru a citi, manipula și salva dataset-ul tabular.
import pandas as pd 
# Importăm 'joblib' pentru a salva fizic (serializa) pe disc "formula" scaler-ului, ca să o putem folosi mai târziu în producție.
import joblib 
# Importăm 'logging' pentru a înregistra pașii execuției elegant, fără să aglomerăm consola cu print-uri.
import logging 
# Importăm 'Path' din 'pathlib' pentru a construi căi către fișiere și foldere care să funcționeze perfect pe orice sistem de operare (Windows, Mac, Linux).
from pathlib import Path 
# Importăm 'StandardScaler' din scikit-learn, care va transforma toate valorile indicatorilor tehnici astfel încât să aibă media 0 și deviația standard 1.
from sklearn.preprocessing import StandardScaler 

# Setăm configurația globală de logare: afișăm doar nivelul INFO și mai sus, formatând mesajul curat.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Definim funcția principală a modulului, specificând clar ce tip de date așteaptă fiecare parametru.
def run_scaler_pipeline(ticker: str, processed_dir: str, scaled_dir: str, model_dir: str, start_date: str, end_date: str):
    # Mesaj de start pentru a ști vizual ce activ financiar (ticker) procesăm în acest moment.
    logging.info(f"--- Inițiere Pipeline de Scalare pentru {ticker} ---")

    # Convertim string-urile primite ca parametri în obiecte de tip Path, ceea ce ne permite să operăm cu ele mult mai ușor.
    processed_path = Path(processed_dir)
    scaled_path = Path(scaled_dir)
    model_path = Path(model_dir)
    
    # Ne asigurăm că folderul unde vom salva datele scalate există. Dacă nu, îl creăm (exist_ok=True previne eroarea dacă el există deja).
    scaled_path.mkdir(parents=True, exist_ok=True)
    # Facem același lucru și pentru folderul de modele, deoarece acolo vom salva fișierul .joblib al scaler-ului.
    model_path.mkdir(parents=True, exist_ok=True)

    # Reconstruim numele exact al fișierului CSV bazat pe convenția proiectului (simbol + perioadă).
    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    
    # Construim rutele complete folosind operatorul `/` (specific pathlib), care lipește folderul de numele fișierului.
    input_file_path = processed_path / file_name
    output_file_path = scaled_path / file_name
    # Numele sub care vom salva "matematica" scaler-ului (media și deviația învățate).
    scaler_file_path = model_path / f"scaler_{ticker}.joblib"

    # Verificare defensivă: dacă fișierul cu features generat anterior nu există la locație, nu are sens să continuăm.
    if not input_file_path.exists():
        logging.error(f"Eroare: Nu am găsit fișierul procesat '{file_name}'.")
        return # Oprim execuția funcției aici.

    # Începem blocul try-except pentru a prinde și raporta civilizat orice eroare neașteptată (ex: date corupte, memorie plină).
    try:
        logging.info("Încarc datele procesate...")
        # Citim dataset-ul brut (cu indicatorii tehnici nescalați) din folderul 'processed'.
        df = pd.read_csv(input_file_path)
        # Parsează coloana 'Date' în format nativ de timp (datetime) pentru a putea filtra matematic pe ani.
        df['Date'] = pd.to_datetime(df['Date'])

        # 1. Creăm o listă cu toate coloanele care conțin indicatori. 
        # Excludem intenționat 'Date' (nu o putem scala) și 'Target_Direction' (este clasa 0 sau 1, nu o stricăm).
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target_Direction']]

        # 2. DEFINIREA GRANIȚEI DE TIMP (Prevenirea Data Leakage)
        # Creăm o "mască" booleană (o serie de True/False) care este True doar pentru rândurile de până în anul 2017 inclusiv.
        # Acesta este exact setul pe care modelele tale de ML se vor antrena mai târziu.
        train_mask = df['Date'].dt.year <= 2017

        # 3. SCALAREA PROPRIU-ZISĂ
        # Instanțiem clasa scaler-ului.
        scaler = StandardScaler()
        
        logging.info("Calculez mediile și deviațiile standard (Fit pe Train)...")
        # PASUL CRITIC: Apelăm .fit() DOAR pe rândurile care respectă train_mask. 
        # Astfel, scaler-ul "învață" media pieței doar din perioada istorică (fără să vadă viitorul din 2020 sau 2021).
        scaler.fit(df.loc[train_mask, feature_cols])

        logging.info("Aplic scalarea pe întregul dataset (Transform)...")
        # Acum că scaler-ul știe mediile din trecut, aplicăm formula (.transform) pe ABSOLUT TOATE rândurile (Train + Test).
        # Suprascriem valorile brute din dataframe cu noile valori scalate.
        df[feature_cols] = scaler.transform(df[feature_cols])

        # 4. SALVAREA PENTRU PRODUCȚIE
        # Luăm obiectul 'scaler' (care conține mediile memorate) și îl dump-uim (salvăm) pe disc în folderul de modele.
        joblib.dump(scaler, scaler_file_path)
        logging.info(f"Scaler salvat cu succes în: {scaler_file_path}")

        # 5. SALVAREA NOULUI DATASET
        # Convertim coloana de date înapoi într-un text formatat frumos ('An-Lună-Zi') pentru ca CSV-ul să fie curat.
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d') 
        # Salvăm tabelul complet, acum conținând indicatori scalati corect, în noul folder 'scaled'. (index=False ignoră indexul numeric intern al pandas).
        df.to_csv(output_file_path, index=False)
        logging.info(f"Dataset scalat salvat cu succes în: {output_file_path}")

    # Dacă a picat ceva în blocul 'try', capturăm eroarea sub numele 'e'.
    except Exception as e:
        # O afișăm în consolă folosind logging, ca să știm exact ce trebuie să depanăm.
        logging.error(f"Eroare la scalarea datelor: {e}")