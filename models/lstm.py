import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

def create_sequences(X, y, time_steps):
    """
    Transformă datele tabulare (2D) în secvențe 3D pentru LSTM.
    """
    Xs, ys = [], []
    # Glisăm o fereastră de mărimea 'time_steps' peste date
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def run_lstm_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str, time_steps: int = 10):
    print(f"--- Inițiere Antrenare LSTM pentru {ticker} ---")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    input_file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_file_path):
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")
        return

    try:
        # 1. Încărcare și împărțire cronologică (ca înainte)
        df = pd.read_csv(input_file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        train_df = df[df['Date'].dt.year <= 2017]
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]
        test_df = df[df['Date'].dt.year >= 2021]

        features_to_drop = ['Date', 'Target_Direction']
        
        # 2. Crearea Secvențelor 3D (Aici se întâmplă magia)
        print(f"  -> Transform datele în secvențe de {time_steps} zile...")
        X_train, y_train = create_sequences(train_df.drop(columns=features_to_drop), train_df['Target_Direction'], time_steps)
        X_val, y_val = create_sequences(val_df.drop(columns=features_to_drop), val_df['Target_Direction'], time_steps)
        X_test, y_test = create_sequences(test_df.drop(columns=features_to_drop), test_df['Target_Direction'], time_steps)

        print(f"  -> Forma X_train 3D: {X_train.shape} (Pachete, Zile, Indicatori)")

        # 3. Calcularea greutăților pentru clase (Balansare)
        negatives = np.sum(y_train == 0)
        positives = np.sum(y_train == 1)
        total = len(y_train)
        
        # Formula Keras pentru class_weight
        weight_for_0 = (1 / negatives) * (total / 2.0)
        weight_for_1 = (1 / positives) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        # 4. Arhitectura Rețelei Neurale LSTM
        print("  -> Construiesc arhitectura rețelei...")
        model = Sequential([
            # Stratul LSTM: Citește secvența temporală
            LSTM(units=50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
            
            # Dropout: Închide aleatoriu 30% din neuroni pentru a preveni memorarea (overfitting)
            Dropout(0.3),
            
            # Stratul de ieșire: Funcția 'sigmoid' ne dă o probabilitate între 0 și 1
            Dense(units=1, activation='sigmoid')
        ])

        # Compilarea modelului (Setarea optimizatorului și a funcției de pierdere)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['AUC'])

        # 5. Antrenamentul (cu oprire inteligentă)
        # Dacă modelul nu își îmbunătățește AUC-ul pe setul de validare timp de 15 epoci, se oprește.
        early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)

        print("  -> Începe antrenamentul (Training)...")
        model.fit(
            X_train, y_train,
            epochs=100,             # Maxim 100 de runde
            batch_size=32,          # Ia câte 32 de secvențe odată
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=1               # Va afișa o bară de progres
        )

        # 6. Evaluarea pe Setul de Test
        print("\n  -> Evaluez modelul pe Test Set (Out-of-Sample)...")
        y_pred_proba = model.predict(X_test).ravel() # .ravel() transformă matricea 2D într-un vector 1D
        
        # Transformăm probabilitățile în decizii (0 sau 1) folosind pragul de 50%
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 7. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== REZULTATE LSTM pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Precizie:  {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Acuratețe: {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nMatrice de Confuzie:")
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")
        print("==========================================\n")

        # 8. Salvarea modelului (Atenție la extensia specifică Keras)
        model_file_path = os.path.join(model_dir, f"lstm_{ticker}.keras")
        model.save(model_file_path)
        print(f"  -> Model LSTM salvat cu succes în: {model_file_path}")

    except Exception as e:
        print(f"  -> Eroare la antrenarea LSTM: {e}")