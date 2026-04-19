# Pipeline de Analiză Cantitativă și Machine Learning (SPY)

Acest proiect implementează un pipeline complet de cercetare cantitativă pentru predicția direcției prețului activului **SPY** (S&P 500 ETF), comparând performanța a patru algoritmi diferiți de Machine Learning.

## Arhitectura Sistemului

Proiectul este structurat pe module logice pentru a asigura izolarea datelor și reproductibilitatea experimentelor:

1.  **Data Loader**: Descărcarea datelor istorice via `yfinance`.
2.  **Feature Engineering**: 
    * Calcularea a peste 15 indicatori tehnici (`pandas_ta`).
    * Transformări ortogonale: Rații preț/medii mobile.
    * Tratarea volumului: Volum relativ și ROC (Rate of Change) pe OBV.
    * **Standardizare**: Aplicarea `StandardScaler` pentru a aduce toți indicatorii la aceeași scară.
3.  **Model Training & Evaluation**: Implementarea și testarea a 4 modele pe un set de date **Out-of-Sample (2021+)**.

## Modele Testate și Rezultate

Am evaluat modelele folosind o tăietură cronologică fixă (**Train: 1993-2017**, **Val: 2018-2020**, **Test: 2021-Prezent**).

| Model | ROC-AUC | Precizie | Recall | Observații |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.5201** | **0.5724** | 0.1268 | **Campionul**. Strategie de tip "Sniper". |
| **Regresie Logistică** | 0.5098 | 0.5593 | 0.5364 | Baseline solid, dar liniar. |
| **LSTM** | 0.5062 | 0.5511 | 0.5007 | Overkill pentru volumul actual de date. |
| **SVM (RBF)** | 0.4963 | 0.5355 | 0.3294 | Overfitting pe zgomotul istoric. |

## Concluzii Strategice

* **Zgomotul Pieței**: Predicția direcției de la o zi la alta ($T+1$) este extrem de dificilă. Scorul ROC-AUC de ~0.52 al XGBoost reprezintă un avantaj statistic real ("edge").
* **Importanța Scalării**: Fără standardizarea trăsăturilor, modelele precum SVM sau LSTM ar fi fost dominate de magnitudinea ADX-ului în detrimentul indicatorilor mai mici.
* **Early Stopping**: Utilizarea setului de validare a prevenit XGBoost să memoreze zgomotul, oprindu-se la momentul optim (arborele nr. 2).

## Tehnologii Utilizate
* **Python 3.x**
* **Pandas & NumPy**: Manipularea datelor.
* **Pandas_TA**: Analiză tehnică.
* **Scikit-Learn**: Scalare, Regresie Logistică, SVM și metrici.
* **XGBoost**: Gradient Boosting.
* **TensorFlow/Keras**: Rețele neurale (LSTM).

---
*Proiect realizat în scop educațional pentru explorarea piețelor financiare prin Machine Learning.*
