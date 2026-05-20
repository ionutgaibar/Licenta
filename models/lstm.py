import os  # Importă modulul 'os' pentru a interacționa cu sistemul de operare (verificare și creare de directoare).
import numpy as np  # Importă 'numpy' cu aliasul 'np', biblioteca standard pentru calcule numerice și manipularea array-urilor multidimensionale.
import pandas as pd  # Importă 'pandas' cu aliasul 'pd', esențială pentru încărcarea, procesarea și analiza datelor tabelare (DataFrames).
import tensorflow as tf  # Importă biblioteca 'tensorflow' sub aliasul 'tf', framework-ul principal creat de Google pentru Deep Learning.
from tensorflow.keras.models import Sequential  # Importă 'Sequential' din Keras, o clasă care ne permite să construim rețeaua neurală adăugând straturi unul după altul.
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Importă straturile de bază: LSTM (pentru procesarea secvențelor de timp), Dense (strat clasic conectat complet) și Dropout.
from tensorflow.keras.callbacks import EarlyStopping  # Importă 'EarlyStopping', o funcție care monitorizează antrenamentul și îl oprește dacă modelul începe să memoreze datele (overfitting).
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Importă metricile de bază din scikit-learn pentru a evalua performanța clasificatorului.
                             f1_score, roc_auc_score, confusion_matrix)

def create_sequences(X, y, time_steps):  # Definește o funcție ajutătoare care primește feature-urile, target-ul și dimensiunea ferestrei de timp.
    """
    Transformă datele tabulare (2D) în secvențe 3D pentru LSTM.  # Explică scopul critic al funcției: modelarea secvențială necesită 3 dimensiuni (pachete, pași de timp, feature-uri).
    """
    Xs, ys = [], []  # Inițializează două liste goale: 'Xs' pentru a stoca calupurile de date (ferestrele de timp) și 'ys' pentru etichetele corespunzătoare de a doua zi.
    # Glisăm o fereastră de mărimea 'time_steps' peste date
    for i in range(len(X) - time_steps):  # Parcurge setul de date cu o buclă, oprindu-se suficient de devreme pentru a putea extrage o ultimă fereastră completă.
        Xs.append(X.iloc[i:(i + time_steps)].values)  # Decupează 'time_steps' rânduri din dataset (un calup istoric) și adaugă valorile (ca matrice) în lista Xs.
        ys.append(y.iloc[i + time_steps])  # Extrage răspunsul corect (creștere/scădere) aferent zilei imediat următoare calupului de mai sus și îl salvează în ys.
    return np.array(Xs), np.array(ys)  # Transformă listele standard din Python în array-uri NumPy extrem de optimizate, pregătite pentru TensorFlow, și le returnează.

def run_lstm_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str, time_steps: int = 10):  # Definește funcția principală, setând implicit 'time_steps' la 10 zile (memoria LSTM-ului).
    print(f"--- Inițiere Antrenare LSTM pentru {ticker} ---")  # Printează în consolă mesajul care marchează începutul procesului pentru simbolul curent.

    if not os.path.exists(model_dir):  # Verifică dacă folderul destinat salvării modelului Keras formatat .keras NU există.
        os.makedirs(model_dir)  # Creează structura de foldere necesară pe disc.

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"  # Construiește dinamic numele fișierului de unde vor fi citite feature-urile.
    input_file_path = os.path.join(input_dir, file_name)  # Creează calea completă, îmbinând numele folderului sursă cu fișierul.

    if not os.path.exists(input_file_path):  # Validează prezența datelor de intrare pe unitatea de stocare.
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")  # Informează utilizatorul despre lipsa datelor.
        return  # Oprește imediat rularea funcției dacă fișierul este de negăsit.

    try:  # Lansează un bloc 'try' general pentru a proteja execuția codului (va prinde eventualele erori TensorFlow sau Pandas).
        # 1. Încărcare și împărțire cronologică (ca înainte)
        df = pd.read_csv(input_file_path)  # Citește datele tabelare curățate și standardizate din fișierul CSV.
        df['Date'] = pd.to_datetime(df['Date'])  # Parsează coloana 'Date' din format String într-un obiect special Datetime pentru a permite filtrarea logică pe ani.

        train_df = df[df['Date'].dt.year <= 2017]  # Restricționează datele de antrenament strict la perioada de până la finele anului 2017.
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]  # Alocă anii 2018-2020 exclusiv setului de validare, folosit pentru Early Stopping.
        test_df = df[df['Date'].dt.year >= 2021]  # Păstrează anii 2021+ izolați pentru testarea finală out-of-sample.

        features_to_drop = ['Date', 'Target_Direction']  # Definește lista cu coloanele pe care rețeaua neurală nu trebuie să le vadă ca informație de input.

        # 2. Crearea Secvențelor 3D (Aici se întâmplă magia)
        print(f"  -> Transform datele în secvențe de {time_steps} zile...")  # Anunță începerea procesării ferestrelor glisante.
        X_train, y_train = create_sequences(train_df.drop(columns=features_to_drop), train_df['Target_Direction'], time_steps)  # Transformă datele de Train în matrice 3D și extrage vectorul y corespondent.
        X_val, y_val = create_sequences(val_df.drop(columns=features_to_drop), val_df['Target_Direction'], time_steps)  # Transformă datele de Val în secvențe 3D.
        X_test, y_test = create_sequences(test_df.drop(columns=features_to_drop), test_df['Target_Direction'], time_steps)  # Transformă datele de Test în secvențe 3D.

        print(f"  -> Forma X_train 3D: {X_train.shape} (Pachete, Zile, Indicatori)")  # Printează structura dimensională (shape) pentru a verifica vizual tensorul (ex: [1000, 10, 15]).

        # 3. Calcularea greutăților pentru clase (Balansare)
        negatives = np.sum(y_train == 0)  # Numără efectiv câte calupuri de antrenament duc la o scădere a prețului (clasa 0).
        positives = np.sum(y_train == 1)  # Numără câte calupuri de antrenament au o țintă pozitivă (creștere, clasa 1).
        total = len(y_train)  # Calculează numărul total de calupuri disponibile în antrenament.

        # Formula Keras pentru class_weight
        weight_for_0 = (1 / negatives) * (total / 2.0)  # Calculează matematic greutatea compensatorie pentru zilele de scădere, acordându-le importanță sporită dacă sunt rare.
        weight_for_1 = (1 / positives) * (total / 2.0)  # Calculează greutatea compensatorie pentru zilele de creștere.
        class_weight = {0: weight_for_0, 1: weight_for_1}  # Leagă greutățile calculate într-un dicționar exact în formatul cerut de Keras.

        # 4. Arhitectura Rețelei Neurale LSTM
        print("  -> Construiesc arhitectura rețelei...")  # Anunță inițializarea modelului secvențial.
        model = Sequential([  # Instanțiază un model Keras tip Sequential, care execută feed-forward-ul în ordinea straturilor de mai jos.
            # Stratul LSTM: Citește secvența temporală
            LSTM(units=50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),  # Primul strat (LSTM): definește 50 de unități de procesare, funcția de activare nativă 'tanh', și primește strict dimensiunile (10 zile, N indicatori).

            # Dropout: Închide aleatoriu 30% din neuroni pentru a preveni memorarea (overfitting)
            Dropout(0.3),  # Stratul de Dropout va "dezactiva" aleatoriu 30% din fluxul de date dintre LSTM și ieșire la fiecare pas, forțând rețeaua să generalizeze mai bine.

            # Stratul de ieșire: Funcția 'sigmoid' ne dă o probabilitate între 0 și 1
            Dense(units=1, activation='sigmoid')  # Stratul final (Dense): condensează informația într-un singur neuron, iar funcția 'sigmoid' presează valoarea rezultată între 0.0% și 100.0%.
        ])

        # Compilarea modelului (Setarea optimizatorului și a funcției de pierdere)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),   # Atașează optimizatorul 'Adam' (care ajustează greutățile în timpul învățării), setându-i o rată de învățare fină.
                      loss='binary_crossentropy',   # Setează funcția de "pedeapsă" (loss); 'binary_crossentropy' este matematic optimă pentru problemele cu două clase (0 și 1).
                      metrics=['AUC'])  # Instruiește modelul să monitorizeze metrica Area Under Curve (ROC-AUC) ca principal indicator de performanță în timpul epocilor.

        # 5. Antrenamentul (cu oprire inteligentă)
        # Dacă modelul nu își îmbunătățește AUC-ul pe setul de validare timp de 15 epoci, se oprește.
        early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)  # Instanțiază monitorul care urmărește 'val_auc' (să fie cât mai 'max'); dacă stagnează 15 epoci ('patience'), oprește procesul și recuperează cele mai bune greutăți ('restore_best_weights').

        print("  -> Începe antrenamentul (Training)...")  # Semnalează că începe procesul intens de calcul pe plăci grafice sau procesor.
        model.fit(  # Execută procesul de propagare și retropropagare (Backpropagation) pe datele noastre.
            X_train, y_train,  # Îi oferă tensorul 3D de input și vectorul cu răspunsuri corecte pentru învățare.
            epochs=100,             # Maxim 100 de runde  # Limitează numărul maxim de treceri complete prin setul de date la 100 de epoci.
            batch_size=32,          # Ia câte 32 de secvențe odată  # Procesează datele în pachete mici de 32 de rânduri simultan pentru a eficientiza consumul de memorie (RAM/VRAM) și a stabiliza gradienții.
            validation_data=(X_val, y_val),  # Pasează setul independent de validare pentru a testa modelul la finalul fiecărei epoci.
            class_weight=class_weight,  # Aplică dicționarul de greutăți pentru a forța modelul să acorde atenție sporită cazurilor minoritare.
            callbacks=[early_stopping],  # Injectează funcția definită anterior care poate întrerupe antrenamentul prematur.
            verbose=1               # Va afișa o bară de progres  # Setează modul de afișare la 1, ceea ce va printa o bară de progres animată în consolă pentru fiecare epocă.
        )

        # 6. Evaluarea pe Setul de Test
        print("\n  -> Evaluez modelul pe Test Set (Out-of-Sample)...")  # Anunță tranziția spre verificarea modelului pe date din 2021+.
        y_pred_proba = model.predict(X_test).ravel() # .ravel() transformă matricea 2D într-un vector 1D  # Cere modelului predicții brute, iar '.ravel()' aplatizează rezultatul (care ar ieși sub formă de listă de liste) într-un array simplu.

        # Transformăm probabilitățile în decizii (0 sau 1) folosind pragul de 50%
        y_pred = (y_pred_proba > 0.5).astype(int)  # Transformă orice probabilitate strict mai mare de 50% în decizia '1' (Creștere) și restul în '0', salvându-le ca întregi.

        # 7. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)  # Calculează proporția totală a predicțiilor corecte din setul de test.
        prec = precision_score(y_test, y_pred)  # Calculează acuratețea modelului atunci când acesta susține explicit că prețul va crește (rata de "false alarms").
        rec = recall_score(y_test, y_pred)  # Evaluează capacitatea modelului de a intercepta toate creșterile reale din piață.
        f1 = f1_score(y_test, y_pred)  # Calculează scorul F1 (media armonică între precizie și recall), relevant când clasele sunt dezechilibrate.
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Evaluează puterea de separare a modelului calculând Aria de sub Curba ROC bazată pe probabilitățile continue.
        cm = confusion_matrix(y_test, y_pred)  # Calculează grila de distribuție exactă a succeselor și erorilor de predicție.

        print("\n=== REZULTATE LSTM pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")  # Tipărește ROC-AUC limitat la 4 zecimale.
        print(f"Precizie:  {prec:.4f}")  # Tipărește valoarea Preciziei cu 4 zecimale.
        print(f"Recall:    {rec:.4f}")  # Tipărește valoarea metricii Recall.
        print(f"Acuratețe: {acc:.4f}")  # Tipărește Acuratețea globală obținută pe test.
        print(f"F1-Score:  {f1:.4f}")  # Tipărește Scorul F1 global.
        print("\nMatrice de Confuzie:")  # Inițiază desenul consolei care prezintă repartiția erorilor pe cadran.
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")  # Printează numărul de True Negatives (scăderi corecte) și False Positives (a zis că scade, dar a crescut).
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")  # Printează numărul de False Negatives (a ratat creșterea) și True Positives (a depistat corect creșterea).
        print("==========================================\n")

        # 8. Salvarea modelului (Atenție la extensia specifică Keras)
        model_file_path = os.path.join(model_dir, f"lstm_{ticker}.keras")  # Generează o cale absolută și un nume de fișier cu terminația modernă '.keras' pentru a stoca rețeaua.
        model.save(model_file_path)  # Folosește metoda nativă Keras 'save()' pentru a exporta arhitectura, greutățile și optimizatorul direct pe disk.
        print(f"  -> Model LSTM salvat cu succes în: {model_file_path}")

    except Exception as e:  # Variabilă de siguranță generală care stochează eroarea tehnică dacă unul dintre pașii anteriori cedează (ex: memorie GPU insuficientă).
        print(f"  -> Eroare la antrenarea LSTM: {e}")  # Afișează printul tehnic al erorii, permițând ca restul programului să ruleze în continuare dacă pipeline-ul este într-o buclă de ticker-e.