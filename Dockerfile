# Use uma imagem base oficial do Python
FROM python:3.9

# Diretório de trabalho
WORKDIR /app

# Copiar requirements.txt e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos (inclusive main.py, 1_🏠Inicio.py, etc.)
COPY . .

# Expor a porta 8080 (porta padrão do Cloud Run)
EXPOSE 8080

# Ao iniciar o contêiner, roda o 'main.py'
ENTRYPOINT ["python", "main.py"]
