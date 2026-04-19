import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

def run_svm_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str):
    print(f"--- Inițiere Antrenare SVM pentru {ticker} ---")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    input_file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_file_path):
        print(f"Eroare: Nu am găsit fișierul '{file_name}'.")
        return

    try:
        # 1. Încărcare și împărțire cronologică (Păstrăm același standard)
        df = pd.read_csv(input_file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Folosim doar Train pentru a respecta izolarea de timp din trecut
        # (SVM nu folosește Validation Set nativ pentru Early Stopping ca XGBoost)
        train_df = df[df['Date'].dt.year <= 2017]
        test_df = df[df['Date'].dt.year >= 2021]

        features_to_drop = ['Date', 'Target_Direction']
        
        X_train = train_df.drop(columns=features_to_drop)
        y_train = train_df['Target_Direction']
        
        X_test = test_df.drop(columns=features_to_drop)
        y_test = test_df['Target_Direction']

        print(f"  -> Train Set: {len(X_train)} zile | Test Set: {len(X_test)} zile")

        # 2. Configurarea Modelului SVM
        print("  -> Construiesc hiperplanul (Antrenare SVM)... (poate dura câteva secunde)")
        model = SVC(
            kernel='rbf',               # Funcția care permite granițe non-liniare (curbate)
            C=1.0,                      # Gradul de penalizare a greșelilor (1.0 e standard)
            gamma='scale',              # Cum se calculează raza "insulelor" de date
            class_weight='balanced',    # Critic pentru bias-ul pozitiv al pieței!
            probability=True,           # FORȚĂM SVM să emită probabilități (necesar pentru ROC-AUC)
            random_state=42
        )

        # 3. Antrenarea modelului
        model.fit(X_train, y_train)

        # 4. Evaluarea pe Test Set
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")
        y_pred = model.predict(X_test)
        
        # Extragem probabilitățile de creștere
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 5. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== REZULTATE SVM pe TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Precizie:  {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Acuratețe: {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nMatrice de Confuzie:")
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")
        print("========================================\n")

        # 6. Salvarea modelului
        model_file_path = os.path.join(model_dir, f"svm_{ticker}.joblib")
        joblib.dump(model, model_file_path)
        print(f"  -> Model SVM salvat cu succes în: {model_file_path}")

    except Exception as e:
        print(f"  -> Eroare la antrenarea SVM: {e}")