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


    
    
def gerarMapa(map_df):

    map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    # leases to geojson
    path = "geojson.json"

    map_df.to_file(path, driver = "GeoJSON")
    with open(path) as geofile:
        j_file = json.load(geofile)

#index geojson
    i=0
    for feature in j_file["features"]:
        feature ['id'] = str(i).zfill(2)
        i += 1
    
    # mapbox token
    mapboxt = 'MapBox Token'

    # define layers and plot map
    PlotarVar = go.Choroplethmapbox(z=map_df['Quantidade'], locations = map_df.index, colorscale = 'Viridis', geojson = j_file,
     text = map_df['Name'], marker_line_width=0.1,  marker_opacity=0.8)
    # Your choropleth data here

    layout = go.Layout(title_text ='Talhões inventariados', title_x =0.5, width=700, height=800,mapbox = dict(center= dict(lat=-13.332585, lon=-39.3103),accesstoken= mapboxt, zoom=14.5,style="stamen-terrain"))
        
    #st.write('Layer 1:', layer1)
    fig = go.Figure(data=PlotarVar, layout=layout)
    # display streamlit

    return st.plotly_chart(fig)