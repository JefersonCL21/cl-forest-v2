# Use uma imagem base oficial do Python
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Agora, em vez de chamar o Streamlit diretamente,
# chamamos o script que levanta o mini servidor + Streamlit
ENTRYPOINT ["python", "main.py"]
