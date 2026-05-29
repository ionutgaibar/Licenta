# backtester.py
import os  # Importă modulul 'os' pentru interacțiunea cu sistemul de operare (lucrul cu fișiere și directoare).
import pandas as pd  # Importă 'pandas' sub aliasul 'pd', pentru manipularea și analiza datelor sub formă de tabele.
import numpy as np  # Importă 'numpy' cu aliasul 'np', utilizat pentru operațiuni matematice eficiente pe vectori și matrice.
import matplotlib.pyplot as plt  # Importă 'pyplot' din 'matplotlib' pentru a desena graficele financiare la final.
import joblib  # Importă 'joblib', folosit pentru a încărca de pe disc modelele de Machine Learning clasic (ex: XGBoost, SVM, LogReg).
import glob  # Importă modulul 'glob', care ajută la căutarea fișierelor dintr-un director folosind pattern-uri (ex: '*.joblib').
import math  # Importă modulul matematic standard, folosit aici pentru a calcula numărul necesar de rânduri pentru grid-ul de grafice.
import tensorflow as tf  # Importă 'tensorflow' sub aliasul 'tf', necesar pentru încărcarea și rularea modelului de Deep Learning (LSTM).

def run_backtester(  # Definește funcția principală care va simula performanța financiară a modelelor în timp.
    ticker: str,  # Parametru pentru simbolul acțiunii/activului financiar (ex: 'AAPL').
    clean_dir: str,  # Parametru ce indică folderul cu datele brute curățate (pentru a prelua prețurile reale).
    processed_dir: str,  # Parametru ce indică folderul cu datele procesate (indicatorii tehnici scalați).
    model_dir: str,  # Parametru pentru locația unde au fost salvate modelele antrenate anterior.
    start_date: str,  # Parametru pentru data de început a intervalului analizat.
    end_date: str,  # Parametru pentru data de final a intervalului analizat.
    initial_capital: float = 1000.0  # Setează un buget de pornire standard de 1.000$ pentru simulare.
    ):
    print(f"--- Inițiere Backtesting Multi-Model pentru {ticker} ---")  # Afișează un mesaj în consolă la startul simulării.

    models = find_models(model_dir)  # Apelează o funcție ajutătoare (definită mai jos) pentru a găsi și încărca rutele tuturor modelelor salvate.

    features_file = os.path.join(  # Formează calea completă către fișierul cu feature-uri.
        processed_dir,  # Folderul sursă pentru features.
        f"{ticker}_{start_date}_to_{end_date}.csv"  # Numele fișierului, construit dinamic din parametri.
    )

    cleaned_file = os.path.join(  # Formează calea completă către fișierul curățat.
        clean_dir,  # Folderul sursă pentru datele curățate.
        f"{ticker}_{start_date}_to_{end_date}.csv"  # Numele fișierului curățat.
    )

    # 1. Încărcarea Datelor (Features scalate + Prețuri reale)
    df_features = pd.read_csv(features_file)  # Citește datele standardizate și indicatorii tehnici într-un DataFrame.
    df_features['Date'] = pd.to_datetime(df_features['Date'])  # Convertește coloana text 'Date' în format datetime real pentru a permite alinierea.

    df_cleaned = pd.read_csv(cleaned_file)  # Citește prețurile financiare brute curățate din CSV.
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])  # Transformă și aici data din text în format datetime nativ pandas.

    # Păstrăm doar data și prețul absolut din fișierul curățat
    df_prices = df_cleaned[['Date', 'Close']]  # Creează un subset doar cu coloana de date calendaristice și prețul de închidere (banii reali).

    # Aliniem temporal (merge) datele de features cu prețul Close real
    # Folosim 'inner' pentru a ne asigura că rândurile șterse (ex: warm-up EMA) nu reapar
    df = pd.merge(df_features, df_prices, on='Date', how='inner')  # Combină tabelul cu indicatori și cel cu prețuri pe baza coloanei 'Date', păstrând doar zilele comune amândurora.

    # 2. Pregătirea setului de test (Test Set: 2021+)
    test_data = df[df['Date'].dt.year >= 2021].copy()  # Creează setul final de test izolând strict perioada de la 2021 încolo, folosind '.copy()' pentru a evita avertismente de rescriere în pandas.

    # Calculăm randamentul pentru ziua următoare (Target_Next_Return) pe baza 'Close' REAL
    test_data['Daily_Return'] = test_data['Close'].pct_change()  # Calculează modificarea procentuală zilnică a portofoliului (cât a variat prețul de azi față de ieri).
    test_data['Target_Next_Return'] = test_data['Daily_Return'].shift(-1)  # Trage randamentul de 'mâine' pe rândul de 'azi', pentru ca modelul să știe cât câștigă/pierde dacă intră în tranzacție azi.

    # Curățăm ultimul rând care va avea NaN la Target_Next_Return
    test_data = test_data.dropna(subset=['Target_Next_Return'])  # Șterge exclusiv ultima zi din setul de test, deoarece nu știm prețul de 'mâine' pentru ea și ar strica simularea.

    # Izolăm strict caracteristicile pentru ML (scoatem meta-datele și prețul Close)
    cols_to_drop = ['Date', 'Target_Direction', 'Target_Next_Return', 'Daily_Return', 'Close']  # Listează toate coloanele care conțin informații de piață sau absolute ce nu trebuie procesate de model.
    features = test_data.drop(columns=cols_to_drop, errors='ignore')  # Elimină coloanele din listă, lăsând obiectul 'features' format strict din indicatorii matematici așteptați de modele.

    # 3. Inițializarea metricilor pentru Buy & Hold (Benchmark)
    test_data['Equity_BuyHold'] = initial_capital * (1 + test_data['Target_Next_Return']).cumprod()  # Simulează contul Buy&Hold: adaugă randamentul zilnic la 1, calculează produsul cumulativ și înmulțește cu mia de start.

    results = {}  # Inițializează un dicționar gol unde vom salva performanțele fiecărui model pentru a le pune într-un tabel final.

    # 4. Iterarea prin fiecare model
    for model_name, model_path in models.items():  # Parcurge dicționarul de modele găsite, obținând pe rând numele modelului și adresa fișierului pe disc.
        if not os.path.exists(model_path):  # Măsură de siguranță: verifică dacă fișierul fizic mai este la locația respectivă.
            print(f"  -> Avertisment: Nu am găsit modelul {model_name} la ruta {model_path}. Îl sărim.")  # Anunță utilizatorul că fișierul lipsește.
            continue  # Trece automat la următorul model din listă, sărind peste instrucțiunile de mai jos.

        print(f"  -> Evaluez modelul: {model_name}...")  # Afișează că a început generarea simulării pentru modelul curent.

        # Încărcare și Predicție în funcție de tipul modelului
        if model_path.endswith('.keras') or model_path.endswith('.h5'):  # Verifică după extensie dacă modelul curent este o rețea neurală (Keras/TensorFlow).
            # Logica pentru LSTM (Deep Learning)
            model = tf.keras.models.load_model(model_path)  # Încarcă în memorie arhitectura și greutățile rețelei LSTM folosind funcția nativă TensorFlow.

            # 1. Extragem automat dimensiunea ferestrei (time_steps) cerută de LSTM (ex: 10)
            time_steps = model.input_shape[1]  # Interoghează modelul antrenat pentru a afla de câte zile de istoric are nevoie simultan (dimensiunea 1 din structura de input).

            # 2. Transformăm datele 2D în secvențe 3D
            features_array = features.values  # Transformă tabelul pandas curent într-o matrice rapidă de valori NumPy.
            X_lstm = []  # Pregătește lista care va reține pachetele de timp 3D.
            for i in range(len(features_array) - time_steps + 1):  # Parcurge fiecare rând permis din setul de date de testare.
                X_lstm.append(features_array[i : i + time_steps])  # Extrage fereastra temporală curentă ('time_steps' zile consecutive) și o lipește la listă.
            X_lstm = np.array(X_lstm)  # Transformă lista de ferestre într-un cub de date tridimensional optimizat pentru TensorFlow.

            # 3. Generăm predicțiile
            y_pred_proba = model.predict(X_lstm, verbose=0).ravel()  # Pasează tensorul 3D în LSTM pentru a obține nivelurile de probabilitate (ascunzând bara de progres cu verbose=0), apoi aplatizează ('ravel') rezultatul.
            raw_signals = (y_pred_proba > 0.5).astype(int)  # Transformă orice probabilitate de peste 50% într-un semnal de cumpărare (1) și restul în (0).

            # 4. Aliniem predicțiile cu DataFrame-ul original
            # Deoarece LSTM a avut nevoie de 'time_steps' zile pentru prima predicție
            # primele 'time_steps - 1' zile nu au predicție (stăm în cash = 0)
            signals = np.zeros(len(features))  # Creează un vector plin cu zero-uri de exact lungimea totală a tabelului setului de test.
            signals[time_steps - 1:] = raw_signals  # Suprascrie acel vector cu semnalele reale, lăsând un 'gap' de zero-uri la început aferent primei ferestre temporale în care modelul era "orb".

        else:  # Dacă fișierul nu are extensie de Keras, înseamnă că e un model clasic.
            # Logica pentru Machine Learning Clasic (Sklearn/XGBoost)
            model = joblib.load(model_path)  # Folosește joblib pentru a deserializa și monta în RAM modele precum XGBoost, SVM sau Regresia Logistică.
            signals = model.predict(features)  # Modelele clasice prezic direct pe tabelul 2D (fără pachete de timp), returnând un șir de 0 și 1 egal cu lungimea tabelului.

        # Adăugăm semnalele în dataframe 
        col_signal = f'Signal_{model_name}'  # Definește un nume de coloană dinamic, personalizat pentru stocarea deciziilor modelului iterat curent.
        col_return = f'Return_{model_name}'  # Definește un nume de coloană dinamic pentru înregistrarea câștigurilor generate de acest model specific.
        col_equity = f'Equity_{model_name}'  # Definește un nume de coloană dinamic pentru soldul cumulat rezultat.

        test_data[col_signal] = signals  # Inserează vectorul complet de decizii (0 și 1) de mai devreme direct în DataFrame-ul de test principal.

        # Strategia: Ești investit doar când semnalul e 1
        test_data[col_return] = test_data[col_signal] * test_data['Target_Next_Return']  # Dacă semnalul e 1, înmulțește cu randamentul (primiți banii). Dacă e 0, randamentul pieței e neutralizat (0).
        test_data[col_equity] = initial_capital * (1 + test_data[col_return]).cumprod()  # Calculează dobânda compusă a strategiei aplicând produsele secvențiale ale randamentelor modelului peste bugetul inițial.

        # Calculare Metrici
        total_ret = (test_data[col_equity].iloc[-1] / initial_capital - 1) * 100  # Raportează soldul final absolut la capitalul inițial, scade unitatea și înmulțește cu 100 pentru procentajul de Profit/Pierdere net.
        sharpe = (test_data[col_return].mean() / test_data[col_return].std()) * np.sqrt(252) if test_data[col_return].std() != 0 else 0  # Calculează Sharpe Ratio (media câștigurilor împărțită la volatilitate), anualizând prin înmulțirea cu radical din 252 zile de tranzacționare (gestionează siguranța la divizare prin zero).

        rolling_max = test_data[col_equity].cummax()  # Urmărește cel mai înalt punct atins vreodată de portofoliu (Ath - All Time High) până în ziua curentă iterată.
        max_dd = ((test_data[col_equity] - rolling_max) / rolling_max).min() * 100  # Măsoară procentual căderea soldului curent față de vârful absolut, extrăgând minimul (cea mai severă "groapă").

        results[model_name] = {  # Salvează parametrii calculați în dicționarul general creat mai sus, folosind numele modelului curent drept cheie.
            'Profit (%)': round(total_ret, 2),  # Stochează randamentul final, limitat frumos la 2 zecimale.
            'Sharpe Ratio': round(sharpe, 2),  # Stochează riscul vs. recompensă, limitat la 2 zecimale.
            'Max Drawdown (%)': round(max_dd, 2)  # Stochează drawwdown-ul (riscul extrem suportat), formatat identic.
        }

    # 5. Calculare Metrici pentru Buy & Hold
    bh_ret = (test_data['Equity_BuyHold'].iloc[-1] / initial_capital - 1) * 100  # Evaluează randamentul final brut în cazul ignorării oricărei predicții, lăsând banii blocați complet în activ.
    bh_sharpe = (test_data['Target_Next_Return'].mean() / test_data['Target_Next_Return'].std()) * np.sqrt(252)  # Calculează rata Sharpe pentru piața în sine folosind seria de randamente zilnice netratate.
    bh_rolling_max = test_data['Equity_BuyHold'].cummax()  # Identifică maximul istoric atins de piață pe parcursul simulării.
    bh_max_dd = ((test_data['Equity_BuyHold'] - bh_rolling_max) / bh_rolling_max).min() * 100  # Măsoară picajul cel mai dramatic suferit de instrument în perioada de testare.

    results['Buy & Hold (Piața)'] = {  # Adaugă "B&H" ca o intrare distinctă în dicționar, tratând-o ca pe încă un "model concurent".
        'Profit (%)': round(bh_ret, 2),  # Stochează randamentul final brut (Buy & Hold).
        'Sharpe Ratio': round(bh_sharpe, 2),  # Stochează Sharpe ratio pentru piață.
        'Max Drawdown (%)': round(bh_max_dd, 2)  # Stochează drawdown-ul maxim al pieței.
    } 

    # 6. Afișarea Tabelului de Rezultate 
    print("\n=== REZULTATE FINANCIARE FINALE (2021+) ===")
    results_df = pd.DataFrame(results).T  # Transpune (invarte rândurile cu coloanele prin proprietatea .T) dicționarul într-un obiect tabelar curat.
    print(results_df.to_string())  # Listează întregul tabel curat în consolă ca text frumos aliniat.
    print("===========================================\n")

    # 7. Trasarea Graficelor Individuale (Subplots 2x2)
    num_models = len(models.keys())  # Numără efectiv câte strategii de Machine Learning au fost rulate pentru a ști cât de mare e nevoia de "pânză" (canvas).

    # Setăm 2 coloane, iar numărul de rânduri se calculează automat
    cols = 2  # Fixează designul grafic cu 2 ferestre una lângă alta pe ecran.
    rows = math.ceil(num_models / cols)  # Împarte numărul de modele la 2 și rotunjește matematic în sus (ceil) pentru a crea suficiente rânduri verticale (ex: 3 modele dau 2 rânduri).

    # Creăm fereastra mare și sub-graficele
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))  # Instanțiază tabloul principal 'fig' și subdiviziunile 'axes', adaptând dimensiunea înălțimii la numărul de rânduri necesare.

    # Aplatizăm matricea de axe pentru a o itera ușor (funcționează și la 1x2, și la 2x2)
    if num_models > 1:  # Tratează cazul generic: avem minim 2 modele de afișat.
        axes = axes.flatten()  # Transformă structura "matrice de ecrane" a subplots într-o singură listă ordonată de ferestre grafice individuale, făcând iterarea liniară posibilă.
    else:  # Gestionează o excepție (dacă există accidental un singur model de desenat).
        axes = [axes]  # Creează manual o listă dintr-un ax solitar pentru ca sintaxa loop-ului de mai jos să nu se blocheze.

    culori = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']  # Creează o selecție predefinită de culori distincte pentru linia de evoluție a fiecărei strategii.

    # Iterăm prin fiecare model și desenăm graficul lui specific
    for idx, model_name in enumerate(models.keys()):  # Parcurge modelele rulând atât un contor numeric 'idx', cât și numele 'model_name'.
        ax = axes[idx]  # Selectează fereastra de grafic (ax) corespunzătoare indexului curent.

        # Plotăm piața (Benchmark-ul) pe FIECARE sub-grafic
        ax.plot(test_data['Date'], test_data['Equity_BuyHold'], label='Buy & Hold (Piața)', color='black', linewidth=1.5, linestyle='--')  # Desenează peste grafic linia neagră punctată și constantă, reprezentând pasivitatea.

        # Plotăm modelul curent
        if f'Equity_{model_name}' in test_data.columns:  # Validare suplimentară înainte de grafică.
            ax.plot(test_data['Date'], test_data[f'Equity_{model_name}'], label=f'Strategie: {model_name}', color=culori[idx % len(culori)], linewidth=2)  # Desenează curba principală groasă și colorată a evoluției portofoliului AI-ului, folosind restul modulo pentru culori infinite.

        ax.set_title(f"Performanță: {model_name} vs B&H", fontsize=12, fontweight='bold')  # Scrie pe deasupra fiecărui chenar din cine cu cine concurează.
        ax.set_xlabel("Data")  # Setează "Data" drept indicator pentru axa orizontală X.
        ax.set_ylabel("Valoare Cont (USD)")  # Fixează axa verticală Y ca măsurând în valoarea monetară.
        ax.legend(loc='upper left')  # Plasează micul bloc explicativ (legenda graficelor) mereu în stânga sus, ca să nu calce pe curbele din extrema dreaptă a graficului.
        ax.grid(True, alpha=0.3)  # Activează grila transparentă pe fundal (30% vizibilitate) care sprijină ochiul la citirea graficului.

    # Dacă avem un număr impar de modele (ex: 3), ștergem ultimul sub-grafic gol
    for i in range(num_models, len(axes)):  # Identifică dacă numărul total de cutii grafice create e mai mare decât nevoia de modele rulate.
        fig.delaxes(axes[i])  # Curăță automat ferestrele albe lăsate accidental goale, distrugându-le de pe figură.

    plt.suptitle(f"Analiză Individuală a Modelelor vs. Piață - Evoluția a {initial_capital}$ pe {ticker}", fontsize=16, fontweight='bold', y=1.02)  # Adaugă un Super Titlu colosal aplicat peste toate ferestrele mai mici, deplasat ușor în sus deasupra pe axa Y.
    plt.tight_layout()  # Redimensionează și așază totul proporțional ca etichetele, legendele și axele să nu se încalece una cu alta.
    plt.show()  # Transmite către modulul Matplotlib ultima comandă, "Afișează la ecran/console tabloul final".

def find_models(base_models_dir):  # Definește sub-funcția ajutătoare, cea care rezolvă găsirea automată a strategiilor.
    """
    Scanează recursiv folderul de modele și returnează un dicționar:  # Oferă detalii clare despre operațiunea iterativă pe discuri.
    {'Nume_Model': 'cale/catre/model.extensie'}  # Prezintă standardul final de ieșire așteptat din funcție.
    """
    models_dict = {}  # Stabilește variabila container goală pentru date.

    # Căutăm toate fișierele .joblib (ML Clasic) și .keras (Deep Learning)
    # Recursiv (**) prin toate subfolderele
    patterns = [  # Alcătuiește o mică colecție de reguli sintactice tipice pentru identificarea mașinii.
        os.path.join(base_models_dir, "**", "*.joblib"),  # Include prima regulă: orice fișier joblib în orice folder.
        os.path.join(base_models_dir, "**", "*.keras"),  # Extinde permisiunile pe fișierele cu format nou Keras.
        os.path.join(base_models_dir, "**", "*.h5")  # Adaugă regulă extra de compatibilitate pentru formatul HDF5 Keras antic.
    ]

    for pattern in patterns:  # Iterare buclă externă ce parcurge fiecare tip de regulă în parte.
        for file_path in glob.glob(pattern, recursive=True):  # Buclează prin modulele găsite de modulul Python Glob cu abilitate recursivă la activ.
            # Extragem numele fișierului fără extensie (ex: 'xgboost_SPY')
            file_name = os.path.basename(file_path)  # Folosește librăria os ca să separe pur și simplu ultimul nume de fișier din lanțul lung de adresă absolută.
            model_name = os.path.splitext(file_name)[0]  # Fracționează numele propriu-zis de sufixul său de sistem de tip .joblib / .keras preluând doar bucata de index zero (fără punct).

            models_dict[model_name] = file_path  # Populează și inserează noile elemente descoperite creând un cheie-valoare legat solid cu path-ul respectiv.

    return models_dict  # Scoate dicționarul general cu modele din izolare trimițându-l funcției supreme backtester de mai sus.