import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
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
import importarDados
import gerarMapa



df = importarDados.carregarDados()

# Filtra o dataframe com base na coluna "Especie"
filtro = df["Especie"] == "Euterpe oleraceae"

# Exibe apenas as linhas do dataframe que atendem ao filtro
novo_df = df.loc[filtro, :]
novo_df["Novo_TPF"] = novo_df.groupby(['TPF', 'Talhao'])['TPF'].transform('count')
plantios = novo_df.groupby(['TPF', 'Talhao'])['Novo_TPF'].mean()
plantios_df= plantios.reset_index()

plantios_df1 = plantios_df

teste = []
producao = []
producaoFruto = []
Data = []
anoInventario = 2023
horizonte = 12
quilosPlanta = 20
Talhao = []

for h in range(horizonte):
    for i, row in plantios_df1.iterrows():
        TPF, qntdade, talhao = row['TPF'], row['Novo_TPF'], row['Talhao']
        if TPF > 0:
            producao.append(qntdade * 0 *quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append((qntdade * 0 *quilosPlanta)*2)
        elif TPF == 0:
            producao.append(qntdade * 0.08 * quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 0.08 *quilosPlanta*2)
        elif TPF == -1:
            producao.append(qntdade * 0.15* quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 0.15 *quilosPlanta*2)
        elif TPF == -2:
            producao.append(qntdade * 0.4* quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 0.4 *quilosPlanta*2)
        elif TPF == -3:
            producao.append(qntdade * 0.75* quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 0.75 *quilosPlanta*2)
        elif TPF == -4:
            producao.append(qntdade * 1* quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 1 *quilosPlanta*2)
        elif TPF <-4:
            producao.append(qntdade * 1* quilosPlanta)
            Data.append(anoInventario + h)
            Talhao.append(talhao)
            producaoFruto.append(qntdade * 1 *quilosPlanta*2)
        else:
            pass
        
        plantios_df1.at[i, 'TPF'] = TPF - 1


resultado = pd.DataFrame({'Produção': producao, 'Data': Data, 'Talhao': Talhao, 'producaoFruto': producaoFruto})
#resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
resultado_df = resultado.groupby(['Data'])['Produção'].sum()
resultado_df= resultado_df.reset_index()
#resultado_df

fig = px.bar(resultado_df, x='Data', y='Produção',
text='Produção',
title='Produção de acaí (<i>Euterpe oleraceae</i>) ao longo dos anos',
labels={'Data': 'Ano', 'Produção': 'Produção de polpa (kg)'},
template='plotly_white')
fig.update_traces(textposition='outside')
fig.update_layout(yaxis_range=[0, 70000])
fig.update_layout(xaxis_tickmode='linear')


fig.show()


