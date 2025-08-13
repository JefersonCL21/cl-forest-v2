FROM python:3.10-slim

# Ambiente UTF-8 e pip mais silencioso/rápido
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Dependências básicas do sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python primeiro (melhor cache)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do app
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "streamlit run '1_🏠Inicio.py' --server.port=${PORT:-8080} --server.address=0.0.0.0"]
