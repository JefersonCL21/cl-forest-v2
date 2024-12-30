import os
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse
from threading import Thread
import streamlit.web.bootstrap

# Instância do FastAPI
app = FastAPI()

# Endpoint de health check
@app.get("/healthz")
def health_check():
    return JSONResponse({"status": "ok"}, status_code=200)

def run_streamlit():
    """
    Função que inicia o Streamlit.
    Ele vai escutar na porta que o Cloud Run espera: 8080
    """
    port = os.environ.get("PORT", 8080)
    streamlit.web.bootstrap.run(
        "1_🏠Inicio.py",
        "", [],
        flag_options={"server.port": port, "server.address": "0.0.0.0"}
    )

if __name__ == "__main__":
    # Subimos o FastAPI em uma thread separada (porta 8000, por exemplo)
    Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000)).start()
    # Rodamos o Streamlit na porta 8080
    run_streamlit()
