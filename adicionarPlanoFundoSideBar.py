import streamlit as st
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


#Plano de fundo
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      /* Aplica imagem no fundo sem afetar layout */
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()}) no-repeat center top / cover;
      }}
      /* Garante que a navegação/ícones fiquem visíveis acima do fundo */
      [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
          position: relative;
          z-index: 1;
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

