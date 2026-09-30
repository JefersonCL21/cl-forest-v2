import base64
import json
import os
import re
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

import adicionarLogo
import importarDados

st.set_page_config(
    page_title="CL Forest Biometrics",
    page_icon=":seedling:",
    layout="wide",
)

#adicionar a logo da empresa
adicionarLogo.add_logo()

#adicionar o plano de fundo do side bar
side_bg = 'imagens//image.png'
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

sidebar_bg(side_bg)

# Deixar o menu escondido.
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Voos disponíveis: cada pasta tem metadados.json (URL dos tiles da ortofoto no
# Cloud Storage) e vetores/ com as detecções do RF-DETR Seg. Gerados por
# exportar_mapa_web.py no repositório UAV_SAF_Training_Data_Sucupira.
VOOS = {"27/01/2026": Path("dados/uav/2026-01-27")}


@st.cache_data
def carregarVoo(pasta):
    pasta = Path(pasta)
    meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
    camadas = {}
    for nome in ("poligonos", "caixas", "centroides"):
        with open(pasta / "vetores" / f"copas_{nome}.geojson", encoding="utf-8") as f:
            camadas[nome] = json.load(f)
    copas = pd.DataFrame([ft["properties"] for ft in camadas["poligonos"]["features"]])
    return meta, camadas, copas


def ordemNatural(nome):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(nome))]


def formatar(valor, casas=1):
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


st.sidebar.markdown("## Mapa UAV")
voo = st.sidebar.selectbox("Voo", list(VOOS))
meta, camadas, copas = carregarVoo(str(VOOS[voo]))

talhoes = importarDados.carregarDadosSHP()
talhoes = talhoes[talhoes.geometry.notna()].to_crs(epsg=4326)
nomes = sorted(talhoes["Name"].dropna().unique(), key=ordemNatural)
escolha = st.sidebar.selectbox("Talhão", ["Toda a fazenda"] + nomes)

# Resumo da seleção
sel = copas if escolha == "Toda a fazenda" else copas[copas["talhao"] == escolha]
st.markdown(f"### Detecção de mogno africano · voo de {voo}")
c1, c2, c3 = st.columns(3)
c1.metric("Copas detectadas", formatar(len(sel), 0))
c2.metric("Área média de copa", f"{formatar(sel['area_m2'].mean())} m²" if len(sel) else "–")
c3.metric("DAP médio estimado", f"{formatar(sel['dap_cm_est'].mean())} cm" if len(sel) else "–")

# URL dos tiles: variável de ambiente (teste local) ou metadados do voo
url_tiles = os.environ.get("CL_UAV_TILES_URL") or meta.get("url_tiles")
w, s, e, n = meta["limites_wgs84"]

m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=15, tiles=None,
               max_zoom=23, control_scale=True)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    max_native_zoom=20,
    max_zoom=23,
).add_to(m)

if url_tiles:
    folium.TileLayer(
        tiles=url_tiles,
        attr=f'Ortofoto UAV {voo}',
        name=f'Ortofoto {voo}',
        overlay=True,
        control=True,
        max_native_zoom=meta["zoom_max"],
        max_zoom=23,
        bounds=[[s, w], [n, e]],
    ).add_to(m)
else:
    st.warning("URL dos tiles da ortofoto não configurada (metadados.json → url_tiles).")

folium.GeoJson(
    talhoes[["Name", "geometry"]].to_json(),
    name='Talhões',
    style_function=lambda f: {'color': '#FFFFFF', 'weight': 2, 'fill': False},
    tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Talhão:']),
).add_to(m)

folium.GeoJson(
    camadas["poligonos"],
    name='Segmentação (copas)',
    style_function=lambda f: {'color': '#FFD60A', 'weight': 1.5, 'fillOpacity': 0},
    highlight_function=lambda f: {'color': '#FF9F0A', 'weight': 3},
    tooltip=folium.GeoJsonTooltip(
        fields=['talhao', 'area_m2', 'dap_cm_est', 'confianca'],
        aliases=['Talhão:', 'Área da copa (m²):', 'DAP estimado (cm):', 'Confiança:'],
    ),
).add_to(m)

folium.GeoJson(
    camadas["caixas"],
    name='Caixas (detecção)',
    show=False,
    style_function=lambda f: {'color': '#32D7FF', 'weight': 1, 'dashArray': '4 3', 'fill': False},
).add_to(m)

folium.GeoJson(
    camadas["centroides"],
    name='Centroides',
    show=False,
    marker=folium.CircleMarker(radius=3, color='#FFFFFF', weight=1,
                               fill=True, fill_color='#FF3B30', fill_opacity=1),
).add_to(m)

if escolha == "Toda a fazenda":
    m.fit_bounds([[s, w], [n, e]])
else:
    x0, y0, x1, y1 = talhoes[talhoes["Name"] == escolha].total_bounds
    m.fit_bounds([[y0, x0], [y1, x1]])

folium.LayerControl(position='topleft', collapsed=False).add_to(m)
folium_static(m, width=1200, height=780)

alom = meta.get("alometria", {})
equacao = alom.get("equacao", "").replace(".", ",").replace(" * sqrt(A)", "·√A")
st.caption(
    f"Ortofoto de {voo} ({formatar(meta['resolucao_cm'])} cm/px). Copas segmentadas com RF-DETR Seg; "
    f"DAP estimado pela área da copa A em m² ({equacao}, n = {alom.get('n', '')} árvores medidas)."
)
