import os
from fastapi import FastAPI
from starlette.responses import JSONResponse
from threading import Thread
import streamlit.web.bootstrap

# Cria o app FastAPI
app = FastAPI()

@app.get("/healthz")
def health_check():
    return JSONResponse({"status": "ok"}, status_code=200)

def run_streamlit():
    port = os.environ.get("PORT", 8080)
    # Executa o Streamlit
    streamlit.web.bootstrap.run(
        "1_🏠Inicio.py",  # Substitua aqui se seu arquivo principal tiver outro nome
        "", [], 
        flag_options={"server.port": port, "server.address": "0.0.0.0"}
    )

if __name__ == "__main__":
    # Roda o FastAPI em uma thread separada, por exemplo na porta 8000
    Thread(target=lambda: app.run(host="0.0.0.0", port=8000)).start()
    run_streamlit()
