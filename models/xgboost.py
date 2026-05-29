# xgboost.py
import os  # Importă modulul 'os' pentru operațiuni la nivel de sistem (ex: gestionarea căilor și directoarelor).
from matplotlib import ticker
import pandas as pd  # Importă biblioteca 'pandas' cu aliasul 'pd', esențială pentru încărcarea și prelucrarea datelor tabelare.
import joblib  # Importă 'joblib', un utilitar folosit pentru a salva și încărca modele Python complexe (cum ar fi cele de ML) direct pe disc.
import matplotlib.pyplot as plt  # Importă 'matplotlib.pyplot' sub aliasul 'plt', o bibliotecă standard pentru vizualizarea datelor și a rezultatelor.
import xgboost as xgb  # Importă biblioteca 'xgboost' sub aliasul 'xgb', un algoritm extrem de performant bazat pe arbori de decizie și gradient boosting.
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Importă din scikit-learn funcțiile matematice pentru evaluarea calității modelului.
                             f1_score, roc_auc_score, confusion_matrix)

def run_xgboost_pipeline(ticker: str, scaled_dir: str, model_dir: str, start_date: str, end_date: str):  # Definește funcția care rulează întregul proces, cerând simbolul, căile directoarelor și intervalul de date ca parametri.
    print(f"--- Inițiere Antrenare XGBoost pentru {ticker} ---")  # Afișează în consolă un mesaj de pornire a procesului pentru simbolul bursier curent.

    if not os.path.exists(model_dir):  # Verifică dacă folderul destinat salvării modelului final nu există încă.
        os.makedirs(model_dir)  # Creează automat structura de directoare necesară pentru salvarea modelului.

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Construiește dinamic numele fișierului ce conține dataset-ul (bazându-se pe setările de generare din pașii precedenți).
    input_file_path = os.path.join(scaled_dir, file_name)  # Creează calea completă către fișierul cu date prelucrate (features) concatenând folderul cu numele fișierului.

    if not os.path.exists(input_file_path):  # Validează prezența fizică a fișierului de intrare pe hard disk.
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")  # Trimite o eroare în consolă dacă dataset-ul lipsește.
        return  # Oprește imediat execuția funcției, deoarece XGBoost nu poate fi antrenat fără date.

    try:  # Lansează un bloc 'try' pentru a proteja execuția codului; orice eroare de aici în jos va fi interceptată la final.
        # 1. Încărcarea și formatarea datelor
        df = pd.read_csv(input_file_path)  # Citește conținutul CSV-ului și îl memorează într-un DataFrame numit 'df'.
        df['Date'] = pd.to_datetime(df['Date'])  # Transformă coloana 'Date' (care este doar text inițial) într-un format temporal nativ pentru a putea face operațiuni cronologice (ex: extragerea anului).

        # 2. Împărțirea Cronologică (Time-Series Split)
        train_df = df[df['Date'].dt.year <= 2017]  # Separă rândurile pentru setul de Antrenament, păstrând doar datele de până la sfârșitul anului 2017 inclusiv.
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]  # Izolează setul de Validare (folosit pentru evaluări intermediare) restricționându-l între anii 2018 și 2020.
        test_df = df[df['Date'].dt.year >= 2021]  # Păstrează setul de Testare izolat absolut, luând doar datele din anul 2021 până în prezent, simulând un scenariu real viitor.

        features_to_drop = ['Date', 'Target_Direction']  # Face o listă cu acele coloane care nu reprezintă un "semnal" (data calendaristică) sau care reprezintă însăși "rezolvarea" (target-ul).

        X_train = train_df.drop(columns=features_to_drop)  # Elimină coloanele non-predictive din tabelul de antrenament, lăsând doar indicatorii financiari puri (Matricea X).
        y_train = train_df['Target_Direction']  # Selectează exclusiv coloana cu direcția pieței (0 sau 1) pentru a o folosi drept ghidaj în antrenament (Vectorul y).

        X_val = val_df.drop(columns=features_to_drop)  # Repetă izolarea feature-urilor pentru setul de validare.
        y_val = val_df['Target_Direction']  # Repetă izolarea răspunsurilor reale pentru setul de validare.

        X_test = test_df.drop(columns=features_to_drop)  # Izolează mediul de intrare (features) pentru setul de test final.
        y_test = test_df['Target_Direction']  # Izolează target-ul real pentru a-l compara mai târziu cu predicțiile modelului pe setul de test.

        print(f"  -> Train Set: {len(X_train)} zile | Val Set: {len(X_val)} zile | Test Set: {len(X_test)} zile")  # Printează o informare cu distribuția numărului de înregistrări din fiecare set de date.

        # 3. Calcularea 'scale_pos_weight' (Echivalentul class_weight='balanced')
        # Formula: Numarul de exemple negative / Numarul de exemple pozitive
        negatives = len(y_train[y_train == 0])  # Numără câte rânduri din setul de antrenament sunt etichetate cu 0 (adică zile în care prețul a scăzut).
        positives = len(y_train[y_train == 1])  # Numără câte rânduri din setul de antrenament au eticheta 1 (zile cu creștere a prețului).
        scale_ratio = negatives / positives  # Calculează raportul exact dintre scăderi și creșteri pentru a informa modelul despre dezechilibrul natural al pieței.

        # 4. Configurarea modelului XGBoost
        # Setările de bază pentru date financiare zgomotoase
        model = xgb.XGBClassifier(  # Inițializează instanța clasificatorului XGBoost.
            n_estimators=1000,          # Numărul maxim de arbori (dar ne vom opri mai devreme)  # Definește limita maximă de "cicluri" (arbori de decizie) la 1000, acționând ca un plafon de siguranță.
            learning_rate=0.05,         # Pași mici și precauți  # Setează "viteza" de învățare scăzută pentru a împiedica modelul să tragă concluzii bruște, adaptându-se fin la date.
            max_depth=4,                # Arbori scunzi (3-5 max în finanțe pt a evita overfitting)  # Restricționează complexitatea fiecărui arbore la 4 niveluri de adâncime, prevenind memorarea excesivă a zgomotului din piață.
            scale_pos_weight=scale_ratio, # Balansarea claselor  # Introduce parametrul calculat mai sus, forțând modelul să penalizeze mai mult greșelile pe clasa minoritară.
            eval_metric='auc',          # Urmărim ROC-AUC în timpul antrenamentului  # Setează capacitatea de separare (Area Under Curve) drept criteriul oficial după care evaluează performanța la fiecare pas.
            early_stopping_rounds=50,   # Dacă pe setul de VALIDARE scorul nu crește timp de 50 de arbori, oprește-te!  # Activează mecanismul defensiv: dacă modelul adaugă 50 de arbori la rând și performanța pe datele de validare nu se mai îmbunătățește, oprește antrenamentul.
            random_state=42  # Blochează seed-ul generatorului de numere aleatoare la 42 pentru ca rezultatele antrenamentului să fie mereu la fel pe aceleași date.
        ) 

        # 5. Antrenarea modelului (Folosind setul de validare ca "arbitru")
        print("  -> Antrenez modelul (cu Early Stopping)...")  # Semnalează utilizatorului începutul procesului intensiv de calcul (fitting).
        model.fit(  # Execută comanda de pornire a învățării algoritmului XGBoost.
            X_train, y_train,  # Îi oferă modelului variabilele independente (X) și răspunsurile corecte (y) aferente perioadei de învățare.
            eval_set=[(X_val, y_val)],  # Oferă setul de validare separat, pe care modelul îl va privi discret la fiecare pas pentru a determina dacă se generalizează corect sau doar memorează (overfit).
            verbose=False # Pune True dacă vrei să vezi cum evoluează scorul la fiecare arbore  # Oprește log-urile detaliate care ar "inunda" consola cu scorul pentru fiecare dintre cei până la 1000 de arbori.
        )

        print(f"  -> Antrenament oprit la arborele nr: {model.best_iteration}")  # Afișează numărul iterației la care Early Stopping-ul a oprit procesul (momentul în care s-a atins vârful de performanță pe setul de validare).

        # 6. Evaluarea pe Test Set
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")  # Anunță că urmează verificarea performanței pe un mediu controlat necunoscut.
        y_pred = model.predict(X_test)  # Returnează clasa finală (0 - Scade, 1 - Crește) stabilită de XGBoost pentru tot setul de date 2021+.
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Preia nu doar decizia binară, ci și certitudinea procentuală a modelului referitoare la probabilitatea ca piața să crească (coloana de index 1).

        # 7. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)  # Măsoară rata totală de răspunsuri corecte din totalul deciziilor luate.
        prec = precision_score(y_test, y_pred)  # Evaluează precizia ("Cât de des a avut dreptate CÂND a zis că piața va crește?").
        rec = recall_score(y_test, y_pred)  # Evaluează sensibilitatea/recall ("Din toate zilele în care piața a crescut REA, câte au fost depistate de model?").
        f1 = f1_score(y_test, y_pred)  # Calculează echilibrul (media armonică) dintre metrica Precision și metrica Recall.
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Măsoară abilitatea modelului de a distribui probabilitățile, distingând între zgomot pozitiv și zgomot negativ independent de orice prag setat.
        cm = confusion_matrix(y_test, y_pred)  # Creează un grid (matrice) pentru a vedea concret câte predicții din fiecare tip de eroare sau succes au avut loc.

        print("\n=== REZULTATE XGBOOST pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")  # Tipărește scorul de separare (AUC) retezat la 4 zecimale.
        print(f"Precizie:  {prec:.4f}")  # Tipărește rata de Precizie retezată la 4 zecimale.
        print(f"Recall:    {rec:.4f}")  # Tipărește scorul Recall retezat la 4 zecimale.
        print(f"Acuratețe: {acc:.4f}")  # Tipărește Acuratețea globală.
        print(f"F1-Score:  {f1:.4f}")  # Tipărește F1-Score-ul pentru o imagine de ansamblu echilibrată.
        print("\nMatrice de Confuzie:")  # Inițiază desenul consolei care prezintă repartiția erorilor.
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")  # Formatează primul rând, prezentând Adevărat Negative (Scăderi identificate corect) vs Fals Pozitive (Creșteri false prezise).
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")  # Formatează al doilea rând: Fals Negative (Creșteri ratate) vs Adevărat Pozitive (Creșteri confirmate și prinse).
        print("=============================================\n")

        # Desenează graficul
        xgb.plot_importance(model, max_num_features=10, importance_type='weight')
        plt.title(f"Importanța Indicatorilor Tehnici pentru {ticker}")
        plt.show()

        # 8. Salvarea modelului
        model_file_path = os.path.join(model_dir, f"xgboost_{ticker}.joblib")  # Generează o denumire unică de fișier pentru a exporta instanța antrenată în disc.
        joblib.dump(model, model_file_path)  # Folosește joblib pentru a stoca algoritmul XGBoost cu toate regulile și arborii pe care i-a învățat pe calea construită.
        print(f"  -> Model XGBoost salvat cu succes în: {model_file_path}")  # Oferă printul de confirmare a finalizării cu succes a întregului pipeline.

    except Exception as e:  # Variabila de siguranță "catch-all" care deviază cursul dacă pică vreun pas (lipsă date, format corupt de dataframe, lipsă ram etc.).
        print(f"  -> Eroare la antrenarea XGBoost: {e}")  # Extrage stringul propriu-zis de eroare și îl loghează curat fără să termine brusc procesul root.