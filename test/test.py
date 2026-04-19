import yfinance as yf

start_date = "2022-12-01"
end_date = "2024-01-01"
ticker = "SPY"
asset = yf.Ticker(ticker)

dividende = asset.dividends
print("Ultimele dividende:\n", dividende.tail(5))

# Varianta 1: Ajustat (Dividendele sunt "topite" în preț)
df_adj = asset.history(start=start_date, end=end_date, auto_adjust=True)

# Varianta 2: Neajustat (Prețul brut de la bursă)
df_raw = asset.history(start=start_date, end=end_date, auto_adjust=False)

# Verificăm o dată specifică, de exemplu 2023-12-20
data_test = "2023-12-20"

pret_adj = df_adj.loc[data_test, 'Close']
pret_raw = df_raw.loc[data_test, 'Close']

print(f"Data: {data_test}")
print(f"Preț cu auto_adjust=True:  {pret_adj:.2f}")
print(f"Preț cu auto_adjust=False: {pret_raw:.2f}")

if pret_adj != pret_raw:
    print("\n✅ Funcționează! Prețul ajustat este diferit de cel brut.")
else:
    print("\n❌ Valorile sunt identice. Verifică dacă au fost dividende în acest interval!")