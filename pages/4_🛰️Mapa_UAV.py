import base64
import json
import os
import re
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from branca.element import MacroElement
from jinja2 import Template

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
            /* menos espaço no topo para o mapa caber na tela */
            .block-container {padding-top: 1.5rem; padding-bottom: 0.5rem;}
            /* mapa com a altura da tela */
            iframe[data-testid="stIFrame"] {height: calc(100vh - 90px) !important; min-height: 450px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

ALTURA_MAPA = 850


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
    # posição (UTM, m) e DAP de cada copa para o índice de competição calculado no navegador
    cen = gpd.GeoDataFrame.from_features(camadas["centroides"]["features"], crs="EPSG:4326")
    cen = cen.to_crs(cen.estimate_utm_crs())
    pos = {i: (round(p.x, 2), round(p.y, 2)) for i, p in zip(cen["id_copa"], cen.geometry)}
    competicao = {
        "ids": copas["id_copa"].tolist(),
        "x": [pos[i][0] for i in copas["id_copa"]],
        "y": [pos[i][1] for i in copas["id_copa"]],
        "dap": copas["dap_cm_est"].tolist(),
    }
    return meta, camadas, copas, competicao


def ordemNatural(nome):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(nome))]


def formatar(valor, casas=1):
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


CLASSES_DAP = list(range(5, 60, 5))


def popupTalhao(nome, area_ha, sel):
    """Resumo do talhão (aparece ao clicar): números + distribuição do DAP em classes de 5 cm."""
    linhas = [("Área do talhão", f"{formatar(area_ha, 2)} ha")]
    if len(sel):
        linhas += [
            ("Copas detectadas", formatar(len(sel), 0)),
            ("Densidade", f"{formatar(len(sel) / area_ha, 0)} árv/ha"),
            ("Área média de copa", f"{formatar(sel['area_m2'].mean())} m²"),
            ("DAP médio estimado", f"{formatar(sel['dap_cm_est'].mean())} cm"),
        ]
    tabela = "".join(
        f"<tr><td style='color:#5E6E64;padding:2px 14px 2px 0'>{a}</td>"
        f"<td style='font-weight:600;text-align:right'>{b}</td></tr>" for a, b in linhas)
    if len(sel):
        cont, _ = np.histogram(sel["dap_cm_est"], bins=CLASSES_DAP)
        usadas = np.flatnonzero(cont)
        barras = "".join(
            f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>"
            f"<span style='width:46px;text-align:right;color:#5E6E64'>{CLASSES_DAP[i]}–{CLASSES_DAP[i + 1]}</span>"
            f"<span style='flex:1;background:#EEF3EF'><span style='display:block;height:12px;"
            f"width:{100 * cont[i] / cont.max():.0f}%;background:#6B8F71'></span></span>"
            f"<span style='width:28px;text-align:right'>{cont[i]}</span></div>"
            for i in range(usadas[0], usadas[-1] + 1))
        grafico = ("<div style='margin-top:10px;font-weight:600;color:#1F3D2B'>"
                   f"Árvores por classe de DAP (cm)</div>{barras}")
    else:
        grafico = "<div style='margin-top:6px;color:#5E6E64'>Nenhuma copa de mogno detectada.</div>"
    html = ("<div style='font-family:Arial,sans-serif;font-size:12px;width:250px'>"
            f"<div style='font-size:15px;font-weight:700;color:#1F3D2B;margin-bottom:6px'>Talhão {nome}</div>"
            f"<table>{tabela}</table>{grafico}</div>")
    return folium.Popup(html, max_width=290)


class ControleCompeticao(MacroElement):
    """Índice de competição de Hegyi calculado no navegador.

    CI_i = Σ (DAP_j / DAP_i) / d_ij para as vizinhas j a até R metros. A régua
    muda o raio sem recarregar a página; o contorno de cada copa recebe a cor
    da classe de competição e a legenda mostra quantas copas há em cada uma.
    """
    _template = Template("""
{% macro script(this, kwargs) %}
(function () {
  const mapa = {{ this._parent.get_name() }};
  const copas = {{ this.camada }};
  const grupo = {{ this.grupo }};  // item "Índice de competição" da caixa de camadas
  const A = {{ this.dados }};
  const LIM = [0.25, 0.5, 0.75];
  const CORES = ['#4DD0E1', '#FFE14D', '#FF9933', '#FF3B30'];
  const NOMES = ['baixa (< 0,25)', 'moderada (0,25 – 0,50)', 'alta (0,50 – 0,75)', 'muito alta (≥ 0,75)'];
  const n = A.ids.length, IDX = {};
  A.ids.forEach((id, i) => { IDX[id] = i; });
  let ci = new Float64Array(n);
  const classe = v => v < LIM[0] ? 0 : v < LIM[1] ? 1 : v < LIM[2] ? 2 : 3;
  const fmt = (v, c) => v.toFixed(c).replace('.', ',');

  function hegyi(R) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = A.x[j] - A.x[i]; if (dx > R || dx < -R) continue;
        const dy = A.y[j] - A.y[i]; if (dy > R || dy < -R) continue;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > 0 && d <= R) s += A.dap[j] / A.dap[i] / d;
      }
      out[i] = s;
    }
    return out;
  }

  // Índice ligado: contorno e preenchimento na cor da classe; desligado: segmentação normal.
  // (vale também ao tirar o destaque do mouse, que reaplica options.style)
  copas.options.style = f => {
    if (!mapa.hasLayer(grupo)) return {color: '#FFD60A', weight: 1.6, fillOpacity: 0};
    const c = CORES[classe(ci[IDX[f.properties.id_copa]])];
    return {color: c, weight: 1.6, fillColor: c, fillOpacity: {{ this.opacidade }}};
  };
  const redesenhar = () => copas.eachLayer(l => copas.resetStyle(l));

  const ctl = L.control({position: 'topright'});
  ctl.onAdd = function () {
    const div = L.DomUtil.create('div');
    div.style.cssText = 'font:12px Arial,sans-serif;color:#2B3A31;background:#fff;padding:9px 11px;' +
      'border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.4);width:220px';
    div.innerHTML =
      '<div style="font-weight:700;font-size:13px;color:#1F3D2B">Competição · índice de Hegyi</div>' +
      '<div style="display:flex;justify-content:space-between;margin:7px 0 2px">' +
      '<span>Raio de busca</span><b class="raio"></b></div>' +
      '<input class="regua" type="range" min="{{ this.raio_min }}" max="{{ this.raio_max }}" step="1" ' +
      'value="{{ this.raio }}" style="width:100%;margin:0">' +
      '<div class="classes" style="margin-top:6px"></div>' +
      '<div class="media" style="margin-top:6px;color:#5E6E64"></div>' +
      '<div style="margin-top:4px;color:#5E6E64;font-size:11px">CI = Σ (DAP<sub>j</sub> / DAP<sub>i</sub>) / d<sub>ij</sub></div>';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    return div;
  };
  ctl.addTo(mapa);
  const caixa = ctl.getContainer();
  caixa.style.display = mapa.hasLayer(grupo) ? '' : 'none';
  grupo.on('add', () => {
    caixa.style.display = '';
    if (!mapa.hasLayer(copas)) mapa.addLayer(copas);  // o índice é mostrado nas copas
    redesenhar();
  });
  grupo.on('remove', () => { caixa.style.display = 'none'; redesenhar(); });

  function atualizar(R) {
    ci = hegyi(R);
    const cont = [0, 0, 0, 0];
    let soma = 0;
    copas.eachLayer(l => {
      const v = ci[IDX[l.feature.properties.id_copa]];
      l.feature.properties.hegyi = Number(v.toFixed(2));
      cont[classe(v)]++;
      soma += v;
    });
    redesenhar();
    caixa.querySelector('.raio').textContent = R + ' m';
    caixa.querySelector('.classes').innerHTML = CORES.map((c, k) =>
      '<div style="display:flex;align-items:center;gap:8px;margin:3px 0">' +
      '<span style="width:22px;border-top:3px solid ' + c + '"></span>' +
      '<span style="flex:1">' + NOMES[k] + '</span><b>' + cont[k] + '</b></div>').join('');
    caixa.querySelector('.media').textContent = 'CI médio: ' + fmt(soma / n, 2) + ' · ' + n + ' copas';
  }
  caixa.querySelector('.regua').addEventListener('input', e => atualizar(Number(e.target.value)));
  atualizar({{ this.raio }});
})();
{% endmacro %}
""")

    def __init__(self, camada, grupo, dados, raio=10, raio_min=6, raio_max=20, opacidade=0.45):
        super().__init__()
        self._name = "ControleCompeticao"
        self.camada = camada.get_name()
        self.grupo = grupo.get_name()
        self.opacidade = opacidade
        self.dados = json.dumps(dados, separators=(",", ":"))
        self.raio, self.raio_min, self.raio_max = raio, raio_min, raio_max


voo = next(iter(VOOS))
meta, camadas, copas, competicao = carregarVoo(str(VOOS[voo]))

talhoes = importarDados.carregarDadosSHP()
talhoes = talhoes[talhoes.geometry.notna()].to_crs(epsg=4326)
nomes = sorted(talhoes["Name"].dropna().unique(), key=ordemNatural)
# área de cada talhão (soma das partes: T2 e T6 têm mais de um polígono)
area_ha = talhoes.to_crs(talhoes.estimate_utm_crs()).geometry.area.groupby(talhoes["Name"]).sum() / 1e4

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

# Talhões: clicar dentro do talhão (fora das copas) ou no nome abre o resumo.
grupo_talhoes = folium.FeatureGroup(name='Talhões')
for nome in nomes:
    partes = talhoes[talhoes["Name"] == nome]
    sel = copas[copas["talhao"] == nome]
    folium.GeoJson(
        partes[["Name", "geometry"]].to_json(),
        style_function=lambda f: {'color': '#FFFFFF', 'weight': 2, 'fillOpacity': 0},
        highlight_function=lambda f: {'weight': 3.5},
        popup=popupTalhao(nome, area_ha[nome], sel),
    ).add_to(grupo_talhoes)
    ponto = partes.geometry.iloc[partes.to_crs(epsg=3857).area.argmax()].representative_point()
    folium.Marker(
        [ponto.y, ponto.x],
        icon=folium.DivIcon(
            icon_size=(60, 18), icon_anchor=(30, 9),
            html=f"<div style='font:700 12px Arial,sans-serif;color:#fff;text-align:center;"
                 f"text-shadow:0 0 3px #000,0 0 3px #000;cursor:pointer'>{nome}</div>"),
        popup=popupTalhao(nome, area_ha[nome], sel),
    ).add_to(grupo_talhoes)
grupo_talhoes.add_to(m)

# Copas: a cor do contorno vem do índice de competição (ControleCompeticao, mais abaixo).
for ft in camadas["poligonos"]["features"]:
    ft["properties"]["hegyi"] = 0.0  # preenchido no navegador
segmentacao = folium.GeoJson(
    camadas["poligonos"],
    name='Segmentação (copas)',
    style_function=lambda f: {'color': '#FFD60A', 'weight': 1.6, 'fillOpacity': 0},
    highlight_function=lambda f: {'weight': 3.5},
    tooltip=folium.GeoJsonTooltip(
        fields=['talhao', 'area_m2', 'dap_cm_est', 'hegyi', 'confianca'],
        aliases=['Talhão:', 'Área da copa (m²):', 'DAP estimado (cm):', 'Índice de Hegyi:', 'Confiança:'],
        localize=True,
    ),
)
segmentacao.add_to(m)
# Liga/desliga o índice de competição (colore a segmentação; a régua e a legenda só aparecem ligado).
indice_competicao = folium.FeatureGroup(name='Índice de competição (Hegyi)', show=False)
indice_competicao.add_to(m)

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

m.fit_bounds([[s, w], [n, e]])
ControleCompeticao(segmentacao, indice_competicao, competicao, raio=10).add_to(m)

folium.LayerControl(position='topleft', collapsed=False).add_to(m)
# Sem largura fixa: o mapa ocupa toda a largura da página.
components.html(folium.Figure().add_child(m).render(), height=ALTURA_MAPA)
