# Use uma imagem base oficial do Python
FROM python:3.9

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências do Python
RUN pip install -r requirements.txt

# Copie o restante do código da aplicação para o diretório de trabalho
COPY . .

# Exponha a porta que o Streamlit usará
EXPOSE 8080

# Defina o ponto de entrada para o contêiner iniciar o aplicativo Streamlit
ENTRYPOINT ["streamlit", "run", "1_🏠Inicio.py", "--server.port=8080", "--server.address=0.0.0.0"]
