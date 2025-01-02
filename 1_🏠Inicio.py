import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as fig
import geopandas as gpd
import altair as alt
import shapefile as shp
import folium
import pyproj
import plotly.graph_objs as go
import json
import plotly.express as px
from streamlit_metrics import metric, metric_row
from PIL import Image
import seaborn as sns
from math import floor
import adicionarLogo    
import adicionarPlanoFundoSideBar

# === CONFIGURAÇÕES DO STREAMLIT ===

st.set_page_config(
    page_title="CL Forest Biometrics",
    page_icon=":seedling:",
    layout="wide",
)

# === ADICIONAR LOGO DA EMPRESA ===
adicionarLogo.add_logo()

# === ADICIONAR PLANO DE FUNDO NO SIDEBAR ===
side_bg = 'imagens//image.png'
adicionarPlanoFundoSideBar.sidebar_bg(side_bg)

st.sidebar.title("Análise de dados")
st.sidebar.markdown("Essa aplicação fornece uma análise para os dados de inventário agroflorestal")

# === FUNÇÃO PRINCIPAL ===

def inicialPage():
    st.header("**Courageous Land!** :earth_americas:")
    st.markdown(
        """
        💻 Usamos tecnologia para modelar atributos biofísicos de árvores e povoamentos florestais 🌲🌳🌲
        - Visite nosso site [Courageous Land](https://www.courageousland.com/)
        """
    )

inicialPage()

# === ESCONDER MENU E FOOTER ===
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
