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
from plotly.subplots import make_subplots
import json
import plotly.express as px
from streamlit_metrics import metric, metric_row
from PIL import Image
import seaborn as sns
from math import floor
import importarDados
import gerarMapa
import streamlit as st
from dateutil.rrule import rrule, YEARLY
import io
import datetime
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import pyproj
import mpld3
from mpld3 import plugins
import base64
from io import BytesIO
from folium.features import DivIcon
import plotly.graph_objects as go
from plotly.offline import plot

def teste():
    st.write("Número de indivíduos por talhão e espécie")

def exibirMapa():
    # Criação do mapa
    m = folium.Map(location=[0.6182, -60.3455], zoom_start=16)

    # Dicionário com os mapas base personalizados
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=True,
            control=True
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        )
    }

    # Adiciona os mapas base personalizados ao mapa
    basemaps['Google Maps'].add_to(m)
    basemaps['Google Satellite Hybrid'].add_to(m)

    # Adicione camadas, marcadores ou outras personalizações ao mapa, se desejar
    folium.Marker([0.6182, -60.3455], popup='Regenera').add_to(m)

    df = gpd.read_file("Talhao_Regenera.geojson")

    # Converte o GeoDataFrame para formato suportado pelo Folium
    geojson_data = df.to_crs(epsg="4326").to_json()

    # Adiciona a camada GeoJSON ao mapa
    folium.GeoJson(geojson_data).add_to(m)

    # Carregar o Excel em um DataFrame
    @st.cache  # Decorador para armazenar em cache
    def load_data():
        df_especies = pd.read_excel("dadosRegenera.xlsx")
        return df_especies
    df_especies = load_data()

    # Agrupar por talhão
    grouped = df_especies.groupby(['TALHAO'])

    # Mapeia os tipos de uso para cores
    uso_cores = {'Frutífera': 'blue', 'Madeireira': 'green'}  # adicione ou altere conforme necessário

    #quant_densidade = st.selectbox('Informações sobre o número de individuos por espécies',['Total de individuos','Densidade de individuos'])
    # Para cada grupo (talhão)...
    for name, group in grouped:
        # Separa as espécies por uso

        frutifera = group[group['uso'] == 'Frutífera'].sort_values('quantidade')
        madeireira = group[group['uso'] == 'Madeireira'].sort_values('quantidade')

        max_quantidade = group['quantidade'].max()
        area = round(group['area'].max(), 2)

        # Configuração dos botões para alternar entre os gráficos
        buttons = [
            dict(
                label=f'Área total {area}',
                method='update',
                args=[{'visible': [True, True, False, False]}]
            ),
            dict(
                label='Hectare',
                method='update',
                args=[{'visible': [False, False, True, True]}]
            )
        ]

        # Criação dos dois gráficos de barras
        fig = go.Figure()


        fig.add_trace(go.Bar(
            x=frutifera['quantidade'],
            y=frutifera['especie.planta'],
            orientation='h',
            marker_color=uso_cores['Frutífera'],
            name='Frutífera',
            text=round(frutifera['quantidade'], 0),
            hovertemplate='Quantidade: %{x}<br>Espécie: %{y}'
        ))

        fig.add_trace(go.Bar(
            x=madeireira['quantidade'],
            y=madeireira['especie.planta'],
            orientation='h',
            marker_color=uso_cores['Madeireira'],
            name='Madeireira',
            text=round(madeireira['quantidade'], 0),
            hovertemplate='Quantidade: %{x}<br>Espécie: %{y}'
        ))


        fig.add_trace(go.Bar(
            x=frutifera['densidade'],
            y=frutifera['especie.planta'],
            orientation='h',
            marker_color=uso_cores['Frutífera'],
            name='Frutífera',
            text=round(frutifera['densidade'], 0),
            hovertemplate='Densidade: %{x}<br>Espécie: %{y}',
            visible=False
        ))

        fig.add_trace(go.Bar(
            x=madeireira['densidade'],
            y=madeireira['especie.planta'],
            orientation='h',
            marker_color=uso_cores['Madeireira'],
            name='Madeireira',
            text=round(madeireira['densidade'], 0),
            hovertemplate='Densidade: %{x}<br>Espécie: %{y}',
            visible=False
        ))

        
        fig.update_layout(xaxis=dict(range=[0, max_quantidade+70]))



        # Configuração dos botões no layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=buttons,
                    direction='left',
                    pad={'r': 0, 't': 0},
                    showactive=True,
                    x=0.1,
                    xanchor='right',
                    y=1.15,
                    yanchor='bottom'
                )
            ]
        )

        # Ajuste a posição da legenda
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="right",
                x=1
            )
        )

        fig.update_layout(
            autosize=False,
            width=500,
            height=300,
            margin=dict(
                l=10,  # margem esquerda
                r=20,  # margem direita
                b=0,  # margem inferior
                t=20,  # margem superior
                pad=5 # padding
            )
        )


        # Converte o gráfico para HTML
        plot_html = plot(fig, output_type='div', include_plotlyjs='cdn')

        # Adiciona o gráfico HTML ao popup
        iframe = folium.IFrame(html=plot_html, width=550, height=360)
        popup = folium.Popup(iframe, max_width=500)

        # Achar a localização do talhão no DataFrame de talhões
        talhao_df = df.loc[df['TALHAO'] == name]

        talhao_row = talhao_df.iloc[0]

        folium.Marker(
            location=[talhao_row['geometry'].centroid.y, talhao_row['geometry'].centroid.x],
            popup=popup,
            icon=DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html="""
                <div style="font-size: 12pt; color: red;">
                <svg width="150" height="36">
                    <text x="0" y="15">%s</text>
                </svg>
                </div>
                """ % name,
            )
        ).add_to(m)
        
        
    # Adiciona o controle de camadas ao mapa
    folium.LayerControl(position='topleft').add_to(m)

    # Exibir o mapa no Streamlit
    folium_static(m, height=700)








