# logreg.py
import os  # Importă modulul 'os' pentru interacțiunea cu sistemul de operare (creare de directoare, manipulare căi de fișiere).
import pandas as pd  # Importă biblioteca 'pandas' cu aliasul 'pd', folosită pentru încărcarea, manipularea și analiza datelor tabelare.
import joblib  # Importă 'joblib', un utilitar excelent pentru serializarea (salvarea) și deserializarea obiectelor Python grele, cum ar fi modelele ML.
from sklearn.linear_model import LogisticRegression  # Importă algoritmul de Regresie Logistică din scikit-learn, folosit aici ca model de bază (baseline) pentru clasificare.
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Importă metricile necesare pentru evaluarea detaliată a performanței modelului.
                             f1_score, roc_auc_score, confusion_matrix)

def run_logreg_pipeline(ticker: str, scaled_dir: str, model_dir: str, start_date: str, end_date: str):  # Definește funcția pipeline cu parametrii de configurare necesari, specificând tipurile de date.
    """
    Antrenează un model de Regresie Logistică folosind o împărțire cronologică a datelor.  # Explică scopul principal: antrenarea unui clasificator pe o serie de timp.
    """
    print(f"--- Inițiere Antrenare Baseline (Logistic Regression) pentru {ticker} ---")  # Afișează un mesaj în consolă pentru a semnala începutul procesului de antrenare pentru simbolul bursier curent.

    if not os.path.exists(model_dir):  # Verifică dacă directorul destinat salvării modelului antrenat lipsește.
        os.makedirs(model_dir)  # Creează directorul pentru model (și orice directoare părinte necesare) dacă acesta nu a fost găsit.

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Construiește dinamic numele fișierului de intrare, menținând convenția de denumire stabilită în pașii anteriori.
    input_file_path = os.path.join(scaled_dir, file_name)  # Formează calea absolută sau relativă completă către fișierul CSV cu feature-urile deja procesate.

    if not os.path.exists(input_file_path):  # Validează existența fișierului de intrare generat de pipeline-ul anterior (feature engineering).
        print(f"Eroare: Nu am găsit fișierul cu features '{file_name}'.")  # Informează utilizatorul că procesul nu poate continua din lipsa datelor.
        return  # Abandonează execuția funcției, prevenind erorile la încercarea de citire a unui fișier inexistent.

    try:  # Lansează un bloc 'try' pentru a intercepta și gestiona grațios eventualele excepții din timpul încărcării, antrenării sau evaluării.
        # 1. Încărcarea datelor
        df = pd.read_csv(input_file_path)  # Încarcă întregul set de date cu feature-uri și target din fișierul CSV într-un DataFrame pandas.

        # Ne asigurăm că data este un obiect datetime pentru a putea filtra pe ani
        df['Date'] = pd.to_datetime(df['Date'])  # Convertește coloana text 'Date' într-un format temporal (datetime) nativ pandas, esențial pentru tăieturile cronologice.

        # 2. Împărțirea Cronologică (Time-Series Split)
        print("  -> Execut împărțirea cronologică a datelor...")  # Anunță începerea procesului de divizare a datelor (train/val/test).
        train_df = df[df['Date'].dt.year <= 2017]  # Filtrează rândurile creând setul de Antrenament (Train) folosind exclusiv datele de până în anul 2017 inclusiv.
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]  # Izolează rândurile pentru setul de Validare (Val) pentru anii cuprinși între 2018 și 2020.
        test_df = df[df['Date'].dt.year >= 2021]  # Formează setul de Testare (Test), strict cu date noi, nemaivăzute, începând cu 2021.

        # 3. Separarea Features (X) de Target (y)
        # Aruncăm 'Date' (nu e feature) și 'Target_Direction' (e rezultatul)
        features_to_drop = ['Date', 'Target_Direction']  # Centralizează într-o listă numele coloanelor care nu trebuie date algoritmului ca informație de învățare.

        X_train = train_df.drop(columns=features_to_drop)  # Elimină 'Date' și Target-ul din setul de antrenament pentru a păstra doar indicatorii puri (Matricea X).
        y_train = train_df['Target_Direction']  # Extrage doar coloana cu etichete (0 sau 1) aferentă setului de antrenament (Vectorul y).

        X_val = val_df.drop(columns=features_to_drop)  # Repetă procesul de izolare a feature-urilor pentru setul de validare.
        y_val = val_df['Target_Direction']  # Repetă procesul de izolare a target-ului pentru setul de validare.

        X_test = test_df.drop(columns=features_to_drop)  # Izolează matricea de feature-uri X pentru setul final de testare.
        y_test = test_df['Target_Direction']  # Extrage etichetele reale y pentru setul de testare, cu care vom compara predicțiile.

        print(f"    Train Set: {len(X_train)} zile (Până în 2017)")  # Printează lungimea (numărul de rânduri) a setului de antrenament pentru control vizual.
        print(f"    Val Set:   {len(X_val)} zile (2018 - 2020)")  # Printează numărul de zile reținute pentru validare.
        print(f"    Test Set:  {len(X_test)} zile (2021 - Prezent)")  # Printează numărul de zile păstrate strict pentru testarea finală out-of-sample.

        # 4. Inițializarea și Antrenarea Modelului
        # class_weight='balanced' este critic! Previne modelul să parieze mereu pe "Creștere"
        # doar pentru că piața are un istoric pozitiv natural.
        # C=0.1 aplică o ușoară regularizare (penalizează complexitatea)
        model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)  # Instanțiază Regresia Logistică cu ponderi echilibrate, regularizare 0.1, stare aleatoare fixă pentru reproductibilitate și max 1000 iterații pentru convergență.

        print("  -> Antrenez modelul pe Train Set...")  # Anunță că algoritmul a început procesul efectiv de învățare matematică.
        model.fit(X_train, y_train)  # Apelează metoda 'fit', prin care modelul de regresie logistică își ajustează coeficienții interni pe baza feature-urilor (X) pentru a prezice corect target-ul (y).

        # 5. Evaluarea pe Test Set (Simularea vieții reale)
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")  # Semnalizează că modelul a fost antrenat și începe testarea pe date necunoscute anterior.

        # model.predict() returnează 0 sau 1 (folosind pragul standard de 50%)
        y_pred = model.predict(X_test)  # Cere modelului să facă predicții clare, binare (0 sau 1) pentru datele din setul de test.

        # model.predict_proba() returnează probabilitatea brută (ex: 62% șanse să crească)
        # Luăm a doua coloană [:, 1] care reprezintă probabilitatea clasei 1 (Creștere)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Returnează probabilitățile continue ale predicției, selectând exclusiv coloana 1 (șansa procentuală ca direcția să fie '1' - Creștere).

        # 6. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)  # Calculează proporția totală a predicțiilor corecte (câte direcții a ghicit exact din total).
        prec = precision_score(y_test, y_pred)  # Calculează precizia: cât la sută din zilele prezise ca fiind de "Creștere" au fost, de fapt, creșteri reale.
        rec = recall_score(y_test, y_pred)  # Calculează sensibilitatea (recall): cât la sută din totalul zilelor reale de "Creștere" au fost identificate de model.
        f1 = f1_score(y_test, y_pred)  # Calculează Scorul F1, care reprezintă media armonică (echilibrul) dintre Precizie și Recall.
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Calculează Aria de sub Curba ROC, măsurând abilitatea agregată a modelului de a face distincția clară între clasa 0 și clasa 1, indiferent de prag.
        cm = confusion_matrix(y_test, y_pred)  # Generează un tabel de 2x2 (Matricea de Confuzie) cu distribuția exactă a predicțiilor corecte și greșite pentru fiecare clasă.

        print("\n=== REZULTATE TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f} (Capacitatea de separare a claselor)")  # Afișează scorul ROC-AUC formatat cu 4 zecimale.
        print(f"Precizie:  {prec:.4f} (Din câte a zis 'Cumpără', atâtea au fost corecte)")  # Afișează valoarea preciziei modelului.
        print(f"Recall:    {rec:.4f} (A captat X% din totalul zilelor crescătoare)")  # Afișează valoarea recall-ului.
        print(f"Acuratețe: {acc:.4f}")  # Afișează acuratețea globală.
        print(f"F1-Score:  {f1:.4f}")  # Afișează echilibrul F1 între precizie și recall.
        print("\nMatrice de Confuzie:")  # Inițiază desenarea în consolă a matricei de confuzie.
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")  # Formatează primul rând: Adevărat Negative (a prezis scădere, a scăzut) și Fals Pozitive (a prezis creștere, dar a scăzut).
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")  # Formatează al doilea rând: Fals Negative (a prezis scădere, dar a crescut) și Adevărat Pozitive (a prezis creștere, a crescut).
        print("==================================\n")

        # 7. Salvarea modelului pentru a fi folosit în producție (ex: pentru a prezice ziua de mâine)
        model_file_path = os.path.join(model_dir, f"log_reg_{ticker}.joblib")  # Construiește numele și locația finală pentru fișierul fizic al modelului (.joblib).
        joblib.dump(model, model_file_path)  # Scrie ('dumpează') modelul ML exact în starea lui de după antrenare direct pe unitatea de stocare, pentru reutilizări viitoare.
        print(f"  -> Model salvat cu succes în: {model_file_path}")  # Confirmă operatorului finalizarea tuturor pașilor și salvarea reușită.

    except Exception as e:  # Captează sub aliasul 'e' orice posibilă eroare apărută oriunde în structura blocului 'try' de mai sus.
        print(f"  -> Eroare la antrenarea modelului: {e}")  # Returnează în consolă o descriere exactă a erorii tehnice, evitând oprirea subită (crash) a întregului script superior.