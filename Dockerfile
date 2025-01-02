# Usar imagem base do Python
FROM python:3.9

# Definir o diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para GeoPandas e Fiona
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-dev \
    && apt-get clean

# Copiar o arquivo requirements.txt e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos para o container
COPY . .

# Expor a porta padrão do Streamlit (8080)
EXPOSE 8080

# Definir a porta padrão para o Cloud Run
ENV PORT 8080

# Comando para iniciar o Streamlit (responde na raiz "/")
CMD ["sh", "-c", "streamlit run 1_🏠Inicio.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
