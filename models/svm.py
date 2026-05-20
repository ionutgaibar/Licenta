import os  # Importă modulul 'os' pentru a interacționa cu sistemul de operare (pentru a lucra cu directoare și căi).
import pandas as pd  # Importă biblioteca 'pandas' cu aliasul 'pd', necesară pentru încărcarea și prelucrarea datelor în format tabelar.
import joblib  # Importă 'joblib', utilizat pentru a serializa (salva) și deserializa modelul antrenat pe disc.
from sklearn.svm import SVC  # Importă clasa 'SVC' (Support Vector Classification) din scikit-learn, algoritmul principal folosit aici.
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Importă funcțiile pentru a calcula performanța modelului.
                             f1_score, roc_auc_score, confusion_matrix)

def run_svm_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str):  # Definește funcția pipeline cu parametrii necesari (simbol, directoare, date).
    print(f"--- Inițiere Antrenare SVM pentru {ticker} ---")  # Afișează un mesaj în consolă indicând începerea antrenării modelului SVM.

    if not os.path.exists(model_dir):  # Verifică dacă directorul pentru salvarea modelelor nu există.
        os.makedirs(model_dir)  # Creează directorul respectiv dacă acesta lipsește.

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Construiește numele fișierului de intrare pe baza simbolului și a perioadei.
    input_file_path = os.path.join(input_dir, file_name)  # Combină directorul de intrare cu numele fișierului pentru a obține calea completă.

    if not os.path.exists(input_file_path):  # Verifică dacă fișierul de date lipsește de pe disc.
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")  # Afișează un mesaj de eroare dacă fișierul nu a fost găsit.
        return  # Oprește execuția funcției deoarece nu avem date pentru antrenare.

    try:  # Începe un bloc de tratare a excepțiilor pentru a capta eventualele erori din timpul rulării.
        # 1. Încărcare și împărțire cronologică (Păstrăm același standard)
        df = pd.read_csv(input_file_path)  # Citește datele din fișierul CSV și le stochează într-un DataFrame numit 'df'.
        df['Date'] = pd.to_datetime(df['Date'])  # Convertește coloana 'Date' din format text într-un format nativ datetime.

        # Folosim doar Train pentru a respecta izolarea de timp din trecut
        # (SVM nu folosește Validation Set nativ pentru Early Stopping ca XGBoost)
        train_df = df[df['Date'].dt.year <= 2017]  # Creează setul de antrenament selectând doar rândurile cu anul 2017 sau mai vechi.
        test_df = df[df['Date'].dt.year >= 2021]  # Creează setul de testare selectând doar rândurile începând din anul 2021.

        features_to_drop = ['Date', 'Target_Direction']  # Definește lista cu coloanele care nu trebuie folosite ca intrări pentru model (data și ținta).

        X_train = train_df.drop(columns=features_to_drop)  # Elimină coloanele non-predictive din setul de antrenament pentru a păstra doar caracteristicile (X).
        y_train = train_df['Target_Direction']  # Extrage doar coloana țintă din setul de antrenament (y).

        X_test = test_df.drop(columns=features_to_drop)  # Păstrează doar caracteristicile pentru setul de testare.
        y_test = test_df['Target_Direction']  # Extrage valorile țintă reale pentru setul de testare.

        print(f"  -> Train Set: {len(X_train)} zile | Test Set: {len(X_test)} zile")  # Afișează numărul de înregistrări din seturile de antrenament și testare.

        # 2. Configurarea Modelului SVM
        print("  -> Construiesc hiperplanul (Antrenare SVM)... (poate dura câteva secunde)")  # Informează utilizatorul că urmează antrenarea propriu-zisă.
        model = SVC(  # Instanțiază modelul Support Vector Classifier.
            kernel='rbf',               # Funcția care permite granițe non-liniare (curbate)  # Specifică utilizarea nucleului Radial Basis Function pentru a capta relații complexe.
            C=1.0,                      # Gradul de penalizare a greșelilor (1.0 e standard)  # Setează parametrul de regularizare; controlează compromisul dintre o margine netedă și clasificarea corectă.
            gamma='scale',              # Cum se calculează raza "insulelor" de date  # Definește parametrul gamma ca fiind calculat automat pe baza numărului de features.
            class_weight='balanced',    # Critic pentru bias-ul pozitiv al pieței!  # Ajustează greutățile invers proporțional cu frecvența claselor pentru a trata dezechilibrul datelor.
            probability=True,           # FORȚĂM SVM să emită probabilități (necesar pentru ROC-AUC)  # Activează calculul probabilităților subiacente, necesare mai târziu pentru metrica ROC-AUC.
            random_state=42  # Setează un seed pentru generatorul de numere aleatoare, garantând reproductibilitatea rezultatelor.
        )

        # 3. Antrenarea modelului
        model.fit(X_train, y_train)  # Ajustează modelul SVM pe datele de antrenament (învață relațiile dintre X și y).

        # 4. Evaluarea pe Test Set
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")  # Semnalează începerea procesului de predicție pe datele nevăzute.
        y_pred = model.predict(X_test)  # Generază predicțiile binare (0 sau 1) pentru setul de testare.

        # Extragem probabilitățile de creștere
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Extrage probabilitățile continue specifice clasei 1 (creștere) pentru a le folosi la calculul ROC-AUC.

        # 5. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)  # Calculează acuratețea generală a modelului (procentul de predicții corecte).
        prec = precision_score(y_test, y_pred)  # Calculează precizia (cât la sută din predicțiile de creștere au fost reale).
        rec = recall_score(y_test, y_pred)  # Calculează sensibilitatea/recall (câte creșteri reale au fost identificate de model).
        f1 = f1_score(y_test, y_pred)  # Calculează scorul F1, adică media armonică între precizie și recall.
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Calculează scorul ROC-AUC folosind probabilitățile extrase anterior.
        cm = confusion_matrix(y_test, y_pred)  # Generează matricea de confuzie pentru a vedea distribuția predicțiilor corecte și greșite.

        print("\n=== REZULTATE SVM pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")  # Tipărește valoarea ROC-AUC cu 4 zecimale.
        print(f"Precizie:  {prec:.4f}")  # Tipărește valoarea preciziei cu 4 zecimale.
        print(f"Recall:    {rec:.4f}")  # Tipărește valoarea recall-ului cu 4 zecimale.
        print(f"Acuratețe: {acc:.4f}")  # Tipărește acuratețea cu 4 zecimale.
        print(f"F1-Score:  {f1:.4f}")  # Tipărește scorul F1 cu 4 zecimale.
        print("\nMatrice de Confuzie:")  # Inițiază printarea matricei de confuzie.
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")  # Afișează rândul 1 din matrice: Adevărat Negative și Fals Pozitive.
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")  # Afișează rândul 2 din matrice: Fals Negative și Adevărat Pozitive.
        print("========================================\n")

        # 6. Salvarea modelului
        model_file_path = os.path.join(model_dir, f"svm_{ticker}.joblib")  # Formează calea finală a fișierului unde va fi salvat modelul SVM.
        joblib.dump(model, model_file_path)  # Salvează modelul antrenat pe disc folosind biblioteca joblib.
        print(f"  -> Model SVM salvat cu succes în: {model_file_path}")  # Afișează mesajul de confirmare a salvării reușite.

    except Exception as e:  # Captează eventualele erori survenite în timpul execuției blocului 'try'.
        print(f"  -> Eroare la antrenarea SVM: {e}")  # Afișează mesajul exact al erorii pentru diagnosticare.