# Folosim o versiune oficială și ușoară (slim) de Python 3.10
FROM python:3.12-slim

# Setăm directorul de lucru în interiorul containerului
WORKDIR /app

# 1. Instalăm uneltele de sistem necesare pentru a compila cod C (obligatoriu pentru TA-Lib)
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Descărcăm, compilăm și instalăm librăria nativă TA-Lib în sistemul de operare Linux
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# 3. Copiem fișierul cu dependențele Python (Atenție: dacă în folder fișierul se numește doar "requirements", șterge ".txt" de mai jos)
COPY requirements.txt .

# 4. Instalăm pachetele tale din fișier (fără să salvăm cache-ul, pentru a ține imaginea mică)
# 4. Actualizăm pip, instalăm numpy mai întâi pentru a ajuta TA-Lib la compilare, apoi restul
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==2.2.6 && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copiem tot restul codului sursă din proiect (mai puțin ce este în .dockerignore)
COPY . .

# 6. Comanda care rulează pipeline-ul tău la pornirea containerului
CMD ["python", "main.py"]