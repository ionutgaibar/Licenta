import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

def run_xgboost_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str):
    print(f"--- Inițiere Antrenare XGBoost pentru {ticker} ---")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    input_file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_file_path):
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")
        return

    try:
        # 1. Încărcarea și formatarea datelor
        df = pd.read_csv(input_file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # 2. Împărțirea Cronologică (Time-Series Split)
        train_df = df[df['Date'].dt.year <= 2017]
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]
        test_df = df[df['Date'].dt.year >= 2021]

        features_to_drop = ['Date', 'Target_Direction']
        
        X_train = train_df.drop(columns=features_to_drop)
        y_train = train_df['Target_Direction']
        
        X_val = val_df.drop(columns=features_to_drop)
        y_val = val_df['Target_Direction']
        
        X_test = test_df.drop(columns=features_to_drop)
        y_test = test_df['Target_Direction']

        print(f"  -> Train Set: {len(X_train)} zile | Val Set: {len(X_val)} zile | Test Set: {len(X_test)} zile")

        # 3. Calcularea 'scale_pos_weight' (Echivalentul class_weight='balanced')
        # Formula: Numarul de exemple negative / Numarul de exemple pozitive
        negatives = len(y_train[y_train == 0])
        positives = len(y_train[y_train == 1])
        scale_ratio = negatives / positives

        # 4. Configurarea modelului XGBoost
        # Setările de bază pentru date financiare zgomotoase
        model = xgb.XGBClassifier(
            n_estimators=1000,          # Numărul maxim de arbori (dar ne vom opri mai devreme)
            learning_rate=0.05,         # Pași mici și precauți
            max_depth=4,                # Arbori scunzi (3-5 max în finanțe pt a evita overfitting)
            scale_pos_weight=scale_ratio, # Balansarea claselor
            eval_metric='auc',          # Urmărim ROC-AUC în timpul antrenamentului
            early_stopping_rounds=50,   # Dacă pe setul de VALIDARE scorul nu crește timp de 50 de arbori, oprește-te!
            random_state=42
        )
        
        # 5. Antrenarea modelului (Folosind setul de validare ca "arbitru")
        print("  -> Antrenez modelul (cu Early Stopping)...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False # Pune True dacă vrei să vezi cum evoluează scorul la fiecare arbore
        )

        print(f"  -> Antrenament oprit la arborele nr: {model.best_iteration}")

        # 6. Evaluarea pe Test Set
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 7. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== REZULTATE XGBOOST pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Precizie:  {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Acuratețe: {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nMatrice de Confuzie:")
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")
        print("=============================================\n")

        # 8. Salvarea modelului
        model_file_path = os.path.join(model_dir, f"xgboost_{ticker}.joblib")
        joblib.dump(model, model_file_path)
        print(f"  -> Model XGBoost salvat cu succes în: {model_file_path}")

    except Exception as e:
        print(f"  -> Eroare la antrenarea XGBoost: {e}")