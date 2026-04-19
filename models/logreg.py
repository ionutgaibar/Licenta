import os
import pandas as pd
import joblib  # Pentru salvarea modelului antrenat pe disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

def run_logreg_pipeline(ticker: str, input_dir: str, model_dir: str, start_date: str, end_date: str):
    """
    Antrenează un model de Regresie Logistică folosind o împărțire cronologică a datelor.
    """
    print(f"--- Inițiere Antrenare Baseline (Logistic Regression) pentru {ticker} ---")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = f"{ticker}_{start_date}_to_{end_date}.csv"
    input_file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_file_path):
        print(f"Eroare: Nu am găsit fișierul cu features '{file_name}'.")
        return

    try:
        # 1. Încărcarea datelor
        df = pd.read_csv(input_file_path)
        
        # Ne asigurăm că data este un obiect datetime pentru a putea filtra pe ani
        df['Date'] = pd.to_datetime(df['Date'])

        # 2. Împărțirea Cronologică (Time-Series Split)
        print("  -> Execut împărțirea cronologică a datelor...")
        train_df = df[df['Date'].dt.year <= 2017]
        val_df = df[(df['Date'].dt.year >= 2018) & (df['Date'].dt.year <= 2020)]
        test_df = df[df['Date'].dt.year >= 2021]

        # 3. Separarea Features (X) de Target (y)
        # Aruncăm 'Date' (nu e feature) și 'Target_Direction' (e rezultatul)
        features_to_drop = ['Date', 'Target_Direction']
        
        X_train = train_df.drop(columns=features_to_drop)
        y_train = train_df['Target_Direction']
        
        X_val = val_df.drop(columns=features_to_drop)
        y_val = val_df['Target_Direction']
        
        X_test = test_df.drop(columns=features_to_drop)
        y_test = test_df['Target_Direction']

        print(f"     Train Set: {len(X_train)} zile (Până în 2017)")
        print(f"     Val Set:   {len(X_val)} zile (2018 - 2020)")
        print(f"     Test Set:  {len(X_test)} zile (2021 - Prezent)")

        # 4. Inițializarea și Antrenarea Modelului
        # class_weight='balanced' este critic! Previne modelul să parieze mereu pe "Creștere" 
        # doar pentru că piața are un istoric pozitiv natural.
        # C=0.1 aplică o ușoară regularizare (penalizează complexitatea)
        model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000)
        
        print("  -> Antrenez modelul pe Train Set...")
        model.fit(X_train, y_train)

        # 5. Evaluarea pe Test Set (Simularea vieții reale)
        print("  -> Evaluez modelul pe Test Set (Out-of-Sample)...")
        
        # model.predict() returnează 0 sau 1 (folosind pragul standard de 50%)
        y_pred = model.predict(X_test)
        
        # model.predict_proba() returnează probabilitatea brută (ex: 62% șanse să crească)
        # Luăm a doua coloană [:, 1] care reprezintă probabilitatea clasei 1 (Creștere)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 6. Calcularea Metricilor
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== REZULTATE TEST SET (2021+) ===")
        print(f"ROC-AUC:   {roc_auc:.4f} (Capacitatea de separare a claselor)")
        print(f"Precizie:  {prec:.4f} (Din câte a zis 'Cumpără', atâtea au fost corecte)")
        print(f"Recall:    {rec:.4f} (A captat X% din totalul zilelor crescătoare)")
        print(f"Acuratețe: {acc:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nMatrice de Confuzie:")
        print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP - Pierderi)]")
        print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP - Câștiguri)]")
        print("==================================\n")

        # 7. Salvarea modelului pentru a fi folosit în producție (ex: pentru a prezice ziua de mâine)
        model_file_path = os.path.join(model_dir, f"log_reg_{ticker}.joblib")
        joblib.dump(model, model_file_path)
        print(f"  -> Model salvat cu succes în: {model_file_path}")

    except Exception as e:
        print(f"  -> Eroare la antrenarea modelului: {e}")