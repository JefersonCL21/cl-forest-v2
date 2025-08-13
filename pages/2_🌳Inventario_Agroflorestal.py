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




st.set_page_config(
    page_title="CL Forest Biometrics",
    page_icon=":seedling:",
    layout="wide",
)

#adicionar a logo da empresa

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://i.imgur.com/7dH5wkT.jpeg);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-size: 230px 200px;
                background-position: 6px 5px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Courageous Land";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 20px;
                position: relative;
                top: 80px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

#adicionar o plano de fundo do side bar
side_bg = 'imagens//image.png'
#Plano de fundo
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

#importar dados do excel de inventário
@st.cache_data
def carregarDados():    
    df = pd.read_csv("dados/GERAL_V2.csv")  

    return df 
df = carregarDados()

@st.cache_data
def carregarDados_DAP_HT():    
    df = pd.read_csv("dados/CRESCIMENTO_DAP_HT.csv")  

    return df 
dfCrescimento_DAP_HT = carregarDados_DAP_HT()

polygon = importarDados.carregarDadosSHP()
map_df = polygon
map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

page = st.sidebar.selectbox('Escolha uma propriedade',['Sucupira Agroflorestas','Regenera'])

if page == 'Sucupira Agroflorestas':
    col1, col2, col3= st.columns(3)


    with col1:
        Uso = st.sidebar.radio('Uso da espécie', ('Inventário resumido', 'Madeira', 'Frutíferas', 'Especiárias', 'IRP', 'Análise de solos', 'Formigueiros', 'Imagens'), horizontal=False)

    with col2:
        if Uso == 'Inventário resumido':
            pass

        elif Uso == 'Madeira':
            Variavel = st.sidebar.radio('Variável analisada', ('Número de indivíduos', 'DAP médio (cm)', 'Altura (m)', 'Area basal (m²)', 'Volume (m³)'), horizontal=False)
        elif Uso == 'Frutíferas':
            Variavel = st.sidebar.radio('Variável analisada', ('TPF', 'Fenofase', 'Vigor'), horizontal=False)

        elif Uso == 'Especiárias':
            Variavel = st.sidebar.radio('Variável analisada', ('Fenofase', 'Vigor'), horizontal=False)

        else:
            pass

    Subcol1, Subcol2= st.columns(2)
    
    if Uso == 'Inventário resumido':

        with st.expander("Espécies inventariadas"):
            listaEspecies1 = df['Especie'].unique()           
            resumo_especie = st.multiselect(
                                    'Espécies selecionadas', listaEspecies1, listaEspecies1
                                    )

            
            res_dados = df.loc[df['Especie'].isin(resumo_especie)]
            res_dados = res_dados.groupby(['Especie','Popular', 'Uso', 'ordem']).size().reset_index(name='n')
            res_dados = res_dados.sort_values(by='ordem',ascending=False)
        
        Sub_res_col1, Sub_res_col2, Sub_res_col3= st.columns([3, 10, 3])
        with Sub_res_col1:
            pass
        with Sub_res_col2:
            fig_res = px.bar(res_dados, y='Popular', x='n',
                color='Uso',
                orientation='h',
                text='n',
                width=800,
                height=1200,
                color_discrete_sequence=[ "#2F4F4F","#556B2F","#1D250E"])
            fig_res.update_traces(textposition='outside', textfont_size=32)
            #fig_res.update_layout(plot_bgcolor="#FFFAFA")
            st.plotly_chart(fig_res, use_container_width=True)

            res_dados = res_dados.reset_index()
            st.write(res_dados[['Popular', 'Especie', 'n']])
            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(res_dados)

            st.download_button(
                label="Download - CSV",
                data=csv,
                file_name='Euterpe_oleraceae.csv',
                mime='text/csv',
            )

        with Sub_res_col3:
            pass

        

    elif Uso == 'Madeira' and Variavel == 'Número de indivíduos':

        listaEspecies = df['Especie'].unique()           
        especie = st.multiselect(
                                'Escolha uma espécie', listaEspecies, ['Dalbergia nigra']
                                )
        Sub_ind_col1, Sub_ind_col2= st.columns(2)

        with Sub_ind_col1:           

            if Variavel == 'Número de indivíduos':
                plantios = df.loc[df['Especie'].isin(especie)].groupby(['Talhao']).count()
                map_df['Quantidade'] = 1        
                for j in range(0, len(map_df.index)):
                    for i in range(0, len(plantios.index)):
                        if (map_df.iloc[j, 1] == plantios.index[i]):
                            map_df.loc[j, 'Quantidade'] = plantios.loc[plantios.index[i], 'Especie']

                gerarMapa.gerarMapa(map_df = map_df)

                enh_qualFuste = st.expander("Qualidade do fuste e posição sociológica", expanded=False)
                with enh_qualFuste:  
                    st.text('''
                                . QF1 são árvores com boa forma física, sadias, fuste comercial com pelo 
                                menos duas toras de 4 metros.
                                . QF2 são árvores com forma aceitável, sadias, fuste comercial com pelo 
                                menos 1 tora de 4 metros. 
                                . QF3 são árvores com formas totalmente irregular, sem condições para 
                                aproveitamento industrial. 

                                . Emergentes - são indivíduos bem desenvolvidos que apresentam copas que atingiram 
                                níveis mais elevados de cobertura do solo, recebendo luz diretamente e lateralmente;
                                . Dossel médio - são indivíduos que apresentam copas de tamanho médio em relação ao 
                                nível geral da floresta e que suportam competição laterais, já que recebem luz 
                                diretamente vinda de cima, mas escassa lateralmente, e;
                                . Dominados – são indivíduos que quase não recebem luz e suas copas são bem menores 
                                do que as apresentadas.

                                ''')


        
        with Sub_ind_col2:
            dados2 = df[df['Especie'].isin(especie)].groupby(['Talhao', 'Popular', 'Q_F']).size().reset_index(name='counts')
            dados2['prop'] = dados2.groupby(['Talhao'])['counts'].apply(lambda x: x / x.sum()).reset_index(drop=True)
            dados2 = dados2.sort_values(by=['Q_F', 'prop'])
            dados2['talhao1'] = pd.Categorical(dados2['Talhao'], categories=dados2['Talhao'].unique(), ordered=True)

            dados2['Q_F'] = pd.Categorical(dados2['Q_F'], categories=dados2['Q_F'].unique(), ordered=True)
            dados2 =  dados2.reset_index()
            

            fig_Q_F = px.bar(dados2, y='Talhao', x='prop', color='Q_F',
            orientation='h', text="counts",
            color_discrete_sequence=["#1D250E","#556B2F",  "#2F4F4F"])
            fig_Q_F.update_layout(title={'text': "Qualidade de fuste", 'x':0.5, 'xanchor': 'center'},
            xaxis_title="Proporção",
            yaxis_title="Talhões")
            #fig_Q_F.update_layout(plot_bgcolor="#FFFAFA")
            fig_Q_F.update_layout(height=500, width = 800)   


            st.plotly_chart(fig_Q_F, use_container_width=True)        
            
            dados3 = df[df['Especie'].isin(especie)].groupby(['Talhao', 'Popular', 'P_S']).size().reset_index(name='counts')
            dados3['prop'] = dados3.groupby(['Talhao'])['counts'].apply(lambda x: x / x.sum()).reset_index(drop=True)
            dados3 = dados3.sort_values(by=['P_S', 'prop'])
            dados3['Talhao'] = pd.Categorical(dados3['Talhao'], categories=dados3['Talhao'].unique(), ordered=True)

            dados3['P_S'] = pd.Categorical(dados3['P_S'], categories=dados3['P_S'].unique()[::-1], ordered=True)

            fig_P_S = px.bar(dados3, y='Talhao', x='prop', color='P_S', orientation='h', text="counts",
            color_discrete_sequence=[ "#2F4F4F","#556B2F","#1D250E"])
            fig_P_S.update_layout(title={'text': "Posição sociológica", 'x':0.5, 'xanchor': 'center'},
            xaxis_title="Proporção",
            yaxis_title="Talhões")
            #fig_P_S.update_layout(plot_bgcolor="#FFFAFA")
            fig_P_S.update_layout(height=500, width = 800)
            

            st.plotly_chart(fig_P_S, use_container_width=True)


                            
    elif Uso == 'Madeira' and Variavel == 'DAP médio (cm)':

        Sub_Esp_col1, Sub_Talhao_col2= st.columns(2)

        with Sub_Esp_col1: 
            listaEspecies = df.loc[df['Uso'].isin(['Madeireira'])]['Especie'].unique()
            listaEspecies.sort()          
            especie = st.multiselect(
                'Escolha uma espécie', listaEspecies, ['Dalbergia nigra']
                )

            df["DAP_MED"] = df.loc[(df['Especie'].isin(especie)) & (~df['DAP'].isnull())].groupby(['Talhao'])['DAP'].transform('mean')
            df1 = df.loc[df['Especie'].isin(especie)]  

            
            plantios = df1.loc[df['Especie'].isin(especie)] 
            plantios = plantios.groupby(['Talhao', 'Especie'])['DAP_MED'].mean().reset_index()

        with Sub_Talhao_col2:
        
            columns = df1.loc[df['Especie'].isin(especie)]['Talhao'].unique().tolist()                
            column_name = st.multiselect('Escolha um Talhão', columns, columns[0])
            Talhao = df1.loc[df1['Talhao'].isin(column_name)]
        

        Sub_box_col1, Sub_box_Espaco, Sub_hist_col2= st.columns([4,0.5,3])

        with Sub_box_col1:
                
                
            map_df['Quantidade'] = 0        
            for j in range(0, len(map_df.index)):
                for i in range(0, len(plantios.index)):
                    if (map_df.loc[j, 'Name'] == plantios.loc[i, 'Talhao']):
                        map_df.loc[j, 'Quantidade'] = plantios["DAP_MED"][i]
            
            gerarMapa.gerarMapa(map_df = map_df) 

        with Sub_hist_col2:

            Talhao = df1.loc[df1['Talhao'].isin(column_name)]
                
            fig1 = px.box(Talhao, x='Talhao', y='DAP', color="Especie", 
            width=800, height=550,
            color_discrete_sequence=["#1D250E", "#006400","#2F4F4F", "#1D250E","#556B2F", "#808000", "#556B2F", "#A9A9A9"], 
            title="Boxplot",
                            labels={
                    "Talhao": " ",
                    "DAP": "DAP (cm)",
                    
                },
            )
            #fig1.update_layout(plot_bgcolor="#FFFAFA")                                
                
            st.plotly_chart(fig1, use_container_width=True)

    #Função para fluxograma inicia aqui 
            def classe(data=df1, numberClass=13):
                minData = np.floor(data.min())
                maxData = np.round(data.max()) + 3

                classes = np.linspace(minData, maxData, num=numberClass, endpoint=True)
                LI = np.round(classes[:-1],1)
                LS = np.round(classes[1:],1)
                clas = [f"[{LI[i]}, {LS[i]})" for i in range(numberClass-1)]
                clas_LI = [LI[i] for i in range(numberClass-1)]
                df = pd.DataFrame({'LI': LI, 'LS': LS, 'clas': clas, 'clas_LI': clas_LI})
                return df

            # Filtre os dados com base na espécie e remova as linhas com valor nulo em 'DAP'
            dadosHist = df.query('Especie in @especie and not DAP.isnull() and Talhao in @column_name')

            
            # Crie uma coluna que indica a classe para cada linha com base nos limites das classes
            dadosBar = classe(data=dadosHist['DAP'])
            dadosHist['classe'] = pd.cut(dadosHist['DAP'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas'].tolist())
            dadosHist['LI'] = pd.cut(dadosHist['DAP'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas_LI'].tolist())
            dadosHist['LI'] = pd.to_numeric(dadosHist['LI'])

            invQualitativo= st.radio('', ['Normal', 'Qualidade Fuste', 'Posição sociológica'], horizontal=True, index=0, key='radio_1')
            
            if invQualitativo == 'Posição sociológica':
                # Agrupe as linhas por classe e conte o número de linhas em cada classe
                dadosBar2 = dadosHist.groupby(['classe', 'P_S'])['DAP'].agg(['count']).reset_index()
                dadosBar2['LI'] = dadosHist.groupby(['classe'])['LI'].first().reset_index(drop=True)

                dadosBar2 = dadosBar2.rename(columns={'count': 'n'})
                
                # Remova a coluna 'DAP' do dataframe de saída
                dadosBar2['LI'] = pd.to_numeric(dadosBar2['LI'])
                dadosBar2 = dadosBar2[['classe', 'n', 'LI', 'P_S']].sort_values('LI')            
                dadosBar2['P_S'] = dadosBar2['P_S'].astype(str)
                

                fig2 = px.bar(dadosBar2, x='classe', y = 'n', color="P_S", text="n", color_discrete_sequence=["#1D250E", "#006400","#808000"])
                fig2.update_layout(legend_title_text='P_S')
                #fig2.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig2, use_container_width=True) 

            elif invQualitativo == 'Qualidade Fuste':
                # Agrupe as linhas por classe e conte o número de linhas em cada classe
                dadosBar2 = dadosHist.groupby(['classe', 'Q_F'])['DAP'].agg(['count']).reset_index()
                dadosBar2['LI'] = dadosHist.groupby(['classe'])['LI'].first().reset_index(drop=True)

                dadosBar2 = dadosBar2.rename(columns={'count': 'n'})
                
                # Remova a coluna 'DAP' do dataframe de saída
                dadosBar2['LI'] = pd.to_numeric(dadosBar2['LI'])
                dadosBar2 = dadosBar2[['classe', 'n', 'LI', 'Q_F']].sort_values('LI')            
                dadosBar2['Q_F'] = dadosBar2['Q_F'].astype(str)
                

                fig2 = px.bar(dadosBar2, x='classe', y = 'n', color="Q_F", text="n", color_discrete_sequence=["#808000", "#006400","#1D250E"])
                fig2.update_layout(legend_title_text='Q_F')
                #fig2.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig2, use_container_width=True)

            else:
                # Agrupe as linhas por classe e conte o número de linhas em cada classe
                dadosBar2 = dadosHist.groupby(['classe'])['DAP'].agg(['count']).reset_index()
                dadosBar2['LI'] = dadosHist.groupby(['classe'])['LI'].first().reset_index(drop=True)

                dadosBar2 = dadosBar2.rename(columns={'count': 'n'})
                
                # Remova a coluna 'DAP' do dataframe de saída
                dadosBar2['LI'] = pd.to_numeric(dadosBar2['LI'])
                dadosBar2 = dadosBar2[['classe', 'n', 'LI']].sort_values('LI')          
                
                
                fig2 = px.bar(dadosBar2, x='classe', y = 'n')                
                fig2.update_traces(marker_color="#1D250E")

                st.plotly_chart(fig2, use_container_width=True) 
            
            

#Função para fluxograma termina aqui  

    elif Uso == 'Madeira' and Variavel == 'Altura (m)':

            Sub_Esp_col1, Sub_Talhao_col2= st.columns(2)

            with Sub_Esp_col1: 
                listaEspecies = df['Especie'].unique()           
                especie = st.multiselect(
                    'Escolha uma espécie', listaEspecies, ['Dalbergia nigra']
                    )

                df["HT_MED"] = df.loc[(df['Especie'].isin(especie)) & (~df['HT_Est'].isnull())].groupby(['Talhao'])['HT_Est'].transform('mean')
                df1 = df.loc[df['Especie'].isin(especie)]        
                
                plantios = df1.loc[df['Especie'].isin(especie)] 
                plantios = plantios.groupby(['Talhao', 'Especie'])['HT_MED'].mean().reset_index()

            with Sub_Talhao_col2:
            
                columns = df1['Talhao'].unique().tolist()                
                column_name = st.multiselect('Escolha um Talhão', columns, ['T1'])
                Talhao = df1.loc[df1['Talhao'].isin(column_name)]
            

            Sub_box_col1, Sub_hist_col2= st.columns(2)
            with Sub_box_col1:
                    
                    
                map_df['Quantidade'] = 0        
                for j in range(0, len(map_df.index)):
                    for i in range(0, len(plantios.index)):
                        if (map_df.loc[j, 'Name'] == plantios.loc[i, 'Talhao']):
                            map_df.loc[j, 'Quantidade'] = plantios['HT_MED'][i]
                
                gerarMapa.gerarMapa(map_df = map_df) 

            with Sub_hist_col2:

                Talhao = df1.loc[df1['Talhao'].isin(column_name)]
                    
                fig1 = px.box(Talhao, x='Talhao', y='HT_Est', color="Especie", width=800, height=550, title="Boxplot",
                color_discrete_sequence=["#1D250E", "#006400","#2F4F4F", "#1D250E","#556B2F", "#808000", "#556B2F", "#A9A9A9"],
                                labels={
                        "Talhao": " ",
                        "HT_Est": "Altura (m)",
                        
                    },
                )
                #fig1.update_layout(plot_bgcolor="#FFFAFA")                               
                    
                st.plotly_chart(fig1, use_container_width=True)

        #Função para fluxograma inicia aqui 
                def classe(data=df1, numberClass=13):
                    minData = np.floor(data.min())
                    maxData = np.round(data.max()) + 3

                    classes = np.linspace(minData, maxData, num=numberClass, endpoint=True)
                    LI = np.round(classes[:-1],1)
                    LS = np.round(classes[1:],1)
                    clas = [f"[{LI[i]}, {LS[i]})" for i in range(numberClass-1)]
                    df = pd.DataFrame({'LI': LI, 'LS': LS, 'clas': clas})
                    return df

                # Filtre os dados com base na espécie e remova as linhas com valor nulo em 'HT_Est'
                dadosHist = df.query('Especie in @especie and not HT_Est.isnull()')

                # Crie uma coluna que indica a classe para cada linha com base nos limites das classes
                dadosBar = classe(data=dadosHist['HT_Est'])
                dadosHist['classe'] = pd.cut(dadosHist['HT_Est'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas'].tolist())

                # Agrupe as linhas por classe e conte o número de linhas em cada classe
                dadosBar2 = dadosHist.groupby('classe')['HT_Est'].agg(['count', 'mean']).reset_index()
                dadosBar2 = dadosBar2.rename(columns={'count': 'n', 'mean': 'ordem'}).sort_values('ordem')

                # Remova a coluna 'HT_Est' do dataframe de saída
                dadosBar2 = dadosBar2[['classe', 'n', 'ordem']]
                
                fig2 = px.bar(dadosBar2, x='classe', y='n')
                fig2.update_traces(marker_color="#1D250E")
                #fig2.update_layout(plot_bgcolor="#FFFAFA")
                                                                            
                st.plotly_chart(fig2, use_container_width=True)

    elif Uso == 'Madeira' and Variavel == 'Area basal (m²)':
        
        col_AB1,  col_AB2 = st.columns(2)

        with col_AB1:
            listaEspecies = df['Especie'].unique()
            especie = st.multiselect(
                'Escolha uma espécie', listaEspecies, ['Dalbergia nigra']
                )

            df["AB"] = df.loc[(df['Especie'].isin(especie)) & (~df['DAP'].isnull())].groupby(['Talhao'])['DAP'].transform(lambda x: x **2 * np.pi / 40000)
            df1 = df.loc[df['Especie'].isin(especie)]

        with col_AB2:

            columns = df1['Talhao'].unique().tolist()                
            column_name = st.multiselect('Escolha um Talhão', columns, ['T1'])
            Talhao = df1.loc[df1['Talhao'].isin(column_name)]



        plantios = df1.loc[df['Especie'].isin(especie)] 
        plantios1 = plantios.groupby(['Talhao']).sum("AB")
        
        plantios1= plantios1.reset_index()


        AB_Mapa_col1, AB_hist_col2 = st.columns(2)

        with AB_Mapa_col1:

            map_df['Quantidade'] = 0 # Depois que arrumar os shp substituir por None       
            for j in range(0, len(map_df.index)):
                for i in range(0, len(plantios1.index)):
                    if (map_df.loc[j, 'Name'] == plantios1.loc[i, 'Talhao']):
                        map_df.loc[j, 'Quantidade'] = plantios1['AB'][i]

        
            gerarMapa.gerarMapa(map_df = map_df)

        with AB_hist_col2:

            Talhao = df1.loc[df1['Talhao'].isin(column_name)]
            fig1 = px.box(Talhao, x='Talhao', y='AB', color="Especie", width=800, height=550, title="Boxplot",
            color_discrete_sequence=["#1D250E", "#006400","#2F4F4F", "#1D250E","#556B2F", "#808000", "#556B2F", "#A9A9A9"],
                            labels={
                    "Talhao": " ",
                    "AB": "Área basal (m²)",
                    
                },
            )
            #fig1.update_layout(plot_bgcolor="#FFFAFA")                               
                
            st.plotly_chart(fig1, use_container_width=True)

            #Iniciando boxplot

            def classe(data=df1, numberClass=13):
                minData = np.floor(data.min())
                maxData = np.round(data.max() + 0.0001, 4)  # Use a small value to ensure maximum value is included in the last interval

                rangeData = maxData - minData
                classes = np.linspace(minData, maxData + 0.05 * rangeData, num=numberClass, endpoint=True)  # Increase upper limit by 10% of range
                LI = np.round(classes[:-1], 4)
                LS = np.round(classes[1:], 4)

                clas = [f"[{LI[i]}, {LS[i]})" for i in range(numberClass-1)]

                df = pd.DataFrame({'LI': LI, 'LS': LS, 'clas': clas})
                return df

            # Filtre os dados com base na espécie e remova as linhas com valor nulo em 'AB'
            dadosHist = df.query('Especie in @especie and not AB.isnull()')

            # Crie uma coluna que indica a classe para cada linha com base nos limites das classes
            dadosBar = classe(data=dadosHist['AB'])
            dadosHist['classe'] = pd.cut(dadosHist['AB'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas'].tolist())

            # Agrupe as linhas por classe e conte o número de linhas em cada classe
            dadosBar2 = dadosHist.groupby('classe')['AB'].agg(['count', 'mean']).reset_index()
            dadosBar2 = dadosBar2.rename(columns={'count': 'n', 'mean': 'ordem'}).sort_values('ordem')

            # Remova a coluna 'AB' do dataframe de saída
            dadosBar2 = dadosBar2[['classe', 'n', 'ordem']]        

            fig2 = px.bar(dadosBar2, x='classe', y='n')
            fig2.update_traces(marker_color="#1D250E")
            #fig2.update_layout(plot_bgcolor="#FFFAFA")
                                                                        
            st.plotly_chart(fig2, use_container_width=True)  


        st.markdown(
            '<hr style="border-top: 0.5px solid "#1D250E";">',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))
        m1.write('')
        m2.metric(label ='Área basal (m²)',value = round(plantios["AB"].sum(),2))
        m3.metric(label ='Área basal média por talhão (m²)',value = round(plantios["AB"].sum()/len(plantios["Talhao"].unique()),2))
        delta1 = round((round(plantios.loc[plantios['Talhao'].isin(column_name)]["AB"].sum(),2) - round(plantios["AB"].sum()/len(plantios["Talhao"].unique()),2)),2)
        m4.metric(label = 'Área basal do talhão selecionado (m²)',value =round(plantios.loc[plantios['Talhao'].isin(column_name)]["AB"].sum(),2), delta = delta1)
        m1.write('')        
#Iniciando volume
    elif Uso == 'Madeira' and Variavel == 'Volume (m³)':
        col_vol1,  col_vol2, col_vol3 = st.columns([4, 4, 2])

        with col_vol1:
            listaEspecies = df['Especie'].unique()
            especie = st.multiselect(
                'Escolha uma espécie', listaEspecies, ['Dalbergia nigra']
                )
            
            df1 = df.loc[df['Especie'].isin(especie)]
        
        with col_vol2:
            columns = df1['Talhao'].unique().tolist()                
            column_name = st.multiselect('Escolha um Talhão', columns, ['T1'])
            Talhao = df1.loc[df1['Talhao'].isin(column_name)]

        with col_vol3:
            FF = float(st.number_input("Fator de forma", value=0.6, help="Escolha um fator de forma.",  min_value=0.1))

#Área basal 
        df["Volume"] = FF * df["HT_Est"] *df.loc[(df['Especie'].isin(especie)) & (~df['DAP'].isnull())].groupby(['Talhao'])['DAP'].transform(lambda x: x **2 * np.pi / 40000)
        df1 = df.loc[df['Especie'].isin(especie)]

        plantios = df1.loc[df['Especie'].isin(especie)] 
        plantios1 = plantios.groupby(['Talhao']).sum("Volume")
        
        plantios1= plantios1.reset_index()

        vol_Mapa_col1, vol_hist_col2 = st.columns(2)

        with vol_Mapa_col1:

            map_df['Quantidade'] = 0 # Depois que arrumar os shp substituir por None       
            for j in range(0, len(map_df.index)):
                for i in range(0, len(plantios1.index)):
                    if (map_df.loc[j, 'Name'] == plantios1.loc[i, 'Talhao']):
                        map_df.loc[j, 'Quantidade'] = plantios1['Volume'][i]

            
            gerarMapa.gerarMapa(map_df = map_df)
        st.markdown(
            '<hr style="border-top: 0.5px solid "#1D250E";">',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))
        m1.write('')
        m2.metric(label ='Volume total (m³)',value = round(plantios["Volume"].sum(),2))
        m3.metric(label ='Volume médio por talhão (m³)',value = round(plantios["Volume"].sum()/len(plantios["Talhao"].unique()),2))
        delta1 = round((round(plantios.loc[plantios['Talhao'].isin(column_name)]["Volume"].sum(),2) - round(plantios["Volume"].sum()/len(plantios["Talhao"].unique()),2)),2)
        m4.metric(label = 'Volume do talhão selecionado (m³)',value =round(plantios.loc[plantios['Talhao'].isin(column_name)]["Volume"].sum(),2), delta = delta1)
        
        m1.write('')

        st.markdown(
            '<hr style="border-top: 0.5px solid "#1D250E";">',
            unsafe_allow_html=True
        )


        with vol_hist_col2:

            Talhao = df1.loc[df1['Talhao'].isin(column_name)]
            fig1 = px.box(Talhao, x='Talhao', y='Volume', color="Especie", width=800, height=550, title="Boxplot",
            color_discrete_sequence=["#1D250E", "#006400","#2F4F4F", "#1D250E","#556B2F", "#808000", "#556B2F", "#A9A9A9"],
                            labels={
                    "Talhao": " ",
                    "Volume": "Volume (m³)",
                    
                },
            )
            #fig1.update_layout(plot_bgcolor="#FFFAFA")                               
                
            st.plotly_chart(fig1, use_container_width=True)

            
    #Função para fluxograma inicia aqui 
            def classe(data=df1, numberClass=13):
                minData = np.floor(data.min())
                maxData = np.round(data.max() + 0.0001, 4)  # Use a small value to ensure maximum value is included in the last interval

                rangeData = maxData - minData
                classes = np.linspace(minData, maxData + 0.02 * rangeData, num=numberClass, endpoint=True)  # Increase upper limit by 10% of range
                LI = np.round(classes[:-1], 4)
                LS = np.round(classes[1:], 4)

                clas = [f"[{LI[i]}, {LS[i]})" for i in range(numberClass-1)]

                df = pd.DataFrame({'LI': LI, 'LS': LS, 'clas': clas})
                return df

            # Filtre os dados com base na espécie e remova as linhas com valor nulo em 'Volume'
            dadosHist = df.query('Especie in @especie and not Volume.isnull()')

            # Crie uma coluna que indica a classe para cada linha com base nos limites das classes
            dadosBar = classe(data=dadosHist['Volume'])
            dadosHist['classe'] = pd.cut(dadosHist['Volume'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas'].tolist())

            # Agrupe as linhas por classe e conte o número de linhas em cada classe
            dadosBar2 = dadosHist.groupby('classe')['Volume'].agg(['count', 'mean']).reset_index()
            dadosBar2 = dadosBar2.rename(columns={'count': 'n', 'mean': 'ordem'}).sort_values('ordem')

            # Remova a coluna 'Volume' do dataframe de saída
            dadosBar2 = dadosBar2[['classe', 'n', 'ordem']]          
                        
            
            fig2 = px.bar(dadosBar2, x='classe', y='n')
            fig2.update_traces(marker_color="#1D250E")
            #fig2.update_layout(plot_bgcolor="#FFFAFA")
                                                                        
            st.plotly_chart(fig2, use_container_width=True)

            
        st.markdown('<h1>Simulação de produção, desbastes e receitas</h1>', unsafe_allow_html=True)


        with st.expander(' ', expanded=True):


            parametros_tab, sucupira_tab = st.tabs(["Simulção", "Sucupira"])

            with parametros_tab:
                if parametros_tab:
                    colunasDesbastes1,  colunasDesbastes2 = st.columns(2)
                    with colunasDesbastes1:
                        #Dados de crescimento e projeção
                        listaEspeciesCrescimento_DAP_HT = dfCrescimento_DAP_HT['Especie'].unique()

                        especie_Crescimento_DAP_HT = st.multiselect(
                            'Escolha uma espécie', listaEspeciesCrescimento_DAP_HT, ['Khaya grandifoliola']
                            )
                        
                        cicloCorte = int(st.number_input("Ciclo de corte?", min_value=16, max_value=30, value=20))            
                                        
                    with colunasDesbastes2:
                        FF1 = float(st.number_input("Fator de forma ", value=0.5, help="Escolha um fator de forma.",  min_value=0.1))

                        precoMadeira = float(st.number_input("Preço da madeira (R$/m³) ", value=2500.0, help="Escolha o preço da madeira.",  min_value=1000.0))
                        
                        densidadeInicial = int(st.number_input("Densidade incial?", min_value=0, max_value=1800, value=400))

                    def desbastes():
                        # Perguntar ao usuário quantos desbastes serão realizados
                        with colunasDesbastes1:
                            num_desbastes = st.number_input("Quantos desbastes serão realizados?", min_value=1, max_value=3, value=1)

                        # Criar a sequência de colunas e controles deslizantes para cada desbaste
                        colunas_desbaste = [st.columns(3) for _ in range(num_desbastes)]

                        desbastes = []
                        for i, (col_ano, col_intensidade, colPrecoProduto) in enumerate(colunas_desbaste):
                            # Adicionar um controle deslizante para o ano do desbaste
                            with col_ano:
                                ano_val = st.number_input(f"Ano do desbaste {i+1}", min_value=1, max_value=30, value=9)

                            # Adicionar um controle deslizante para a intensidade do desbaste
                            with col_intensidade:
                                intensidade_val = st.slider(f"Intensidade do desbaste {i+1}", 0.0, 1.0, .5)

                            with colPrecoProduto:
                                Preco_produto = float(st.number_input(f"Preço do produto do desbaste {i+1} (R$/m³)?", min_value=0.00, max_value=5000.00, value=100.00))

                            # Adicionar os valores do desbaste à lista
                            desbastes.append((ano_val, intensidade_val, Preco_produto))
                        
                            

                        # Imprimir os valores selecionados pelo usuário para cada desbaste
                        for i, (ano_val, intensidade_val, Preco_produto) in enumerate(desbastes):
                            st.write(f"Desbaste {i+1}: ano={ano_val}, intensidade={intensidade_val}")

                        return desbastes

                    desbastes = desbastes()


                for i, (ano_val, intensidade_val, Preco_produto) in enumerate(desbastes):
                    # Calcular o volume para os dados de crescimento em altura e DAP
                    dfCrescimento_DAP_HT["Volume"] = FF1 * dfCrescimento_DAP_HT["HT_Est"] * dfCrescimento_DAP_HT['DAP'].transform(lambda x: x **2 * np.pi / 40000)
                    dfCrescimento_DAP_HT["VolumeDesbastado"] = dfCrescimento_DAP_HT["Volume"].copy()
                    dfCrescimento_DAP_HT["Produto"] = " "
                    dfCrescimento_DAP_HT["precoProduto"] = dfCrescimento_DAP_HT["Volume"] 
                    # adiciona uma coluna "Volume" ao dataframe        

                    # multiplica o volume por densidadeInicial ou 200 dependendo da idade
                    if(i==0):
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < ano_val, 'Volume'] *= densidadeInicial
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Volume'] *= densidadeInicial * (1 - intensidade_val)

                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < ano_val, 'VolumeDesbastado'] *= 0
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'VolumeDesbastado'] *= densidadeInicial * intensidade_val

                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'precoProduto'] *= densidadeInicial * (intensidade_val) * Preco_produto
                        

                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Produto'] = f"Desbaste {i+1}"

                    elif(i==1):
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0], 'Volume'] *= densidadeInicial
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'Volume'] *= densidadeInicial * (1 - desbastes[i-1][1])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Volume'] *= densidadeInicial* (1 - desbastes[i-1][1]) * (1 - intensidade_val)


                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0], 'VolumeDesbastado'] *= 0
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'VolumeDesbastado'] *= densidadeInicial * (desbastes[i-1][1])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'VolumeDesbastado'] *= densidadeInicial* (1 - desbastes[i-1][1]) * (intensidade_val)

                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'precoProduto'] *= densidadeInicial * (desbastes[i-1][1])* (desbastes[i-1][2])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'precoProduto'] *= densidadeInicial* (1 - desbastes[i-1][1]) * (intensidade_val)* Preco_produto

                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'Produto'] = f"Desbaste {i}"
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Produto'] = f"Desbaste {i+1}"
                                        
                    else:
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < desbastes[i-2][0], 'Volume'] *= densidadeInicial
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-2][0]) & (dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0]), 'Volume'] *= densidadeInicial * (1 - desbastes[i-2][1])
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'Volume'] *= densidadeInicial * (1 - desbastes[i-2][1])* (1 - desbastes[i-1][1])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Volume'] *= densidadeInicial * (1 - desbastes[i-2][1])* (1 - desbastes[i-1][1]) * (1 - intensidade_val)

                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < desbastes[i-2][0], 'VolumeDesbastado'] *= 0
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-2][0]) & (dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0]), 'VolumeDesbastado'] *= densidadeInicial * (desbastes[i-2][1])
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'VolumeDesbastado'] *= densidadeInicial * (desbastes[i-2][1])* (desbastes[i-1][1])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'VolumeDesbastado'] *= densidadeInicial * (desbastes[i-2][1])* (desbastes[i-1][1]) * (intensidade_val)


                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] < desbastes[i-2][0], 'VolumeDesbastado'] *= 0
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-2][0]) & (dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0]), 'precoProduto'] *= densidadeInicial * (desbastes[i-2][1])* (desbastes[i-2][2])
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'precoProduto'] *= densidadeInicial * (desbastes[i-2][1])* (desbastes[i-1][1])* (desbastes[i-1][2])
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'precoProduto'] *= densidadeInicial * (desbastes[i-2][1])* (desbastes[i-1][1]) * (intensidade_val) * Preco_produto

                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-2][0]) & (dfCrescimento_DAP_HT['idade'] < desbastes[i-1][0]), 'Produto'] = f"Desbaste {i-1}"
                        dfCrescimento_DAP_HT.loc[(dfCrescimento_DAP_HT['idade'] >= desbastes[i-1][0]) & (dfCrescimento_DAP_HT['idade'] < ano_val), 'Produto'] = f"Desbaste {i}"
                        dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['idade'] >= ano_val, 'Produto'] = f"Desbaste {i+1}"
                    

                
                dfCrescimento_DAP_HT1 = dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['Especie'].isin(especie_Crescimento_DAP_HT)]
                dfCrescimento_DAP_HT1 = dfCrescimento_DAP_HT1.loc[dfCrescimento_DAP_HT1['idade'] <= cicloCorte]

                dfDesbaste_DAP_HT = dfCrescimento_DAP_HT1.loc[dfCrescimento_DAP_HT['idade'].isin([desbastes[i][0] for i, _ in enumerate(desbastes)])]
                dfCorteRaso_DAP_HT1 = dfCrescimento_DAP_HT1.loc[dfCrescimento_DAP_HT1['idade'].isin([cicloCorte])]
                dfCorteRaso_DAP_HT1["Produto"] = "Corte raso"
                dfCorteRaso_DAP_HT1["precoProduto"] = dfCorteRaso_DAP_HT1["Volume"] * precoMadeira
                dfCorteRaso_DAP_HT1["VolumeDesbastado"] = dfCorteRaso_DAP_HT1["Volume"]


                df_Multiprodutos =  pd.concat([dfDesbaste_DAP_HT, dfCorteRaso_DAP_HT1], axis=0)
                        
                col_grafico1, col_grafico2 = st.columns(2)
                
                # Criar subplots
                fig = make_subplots(specs=[[{"secondary_y": False}]])

                # Adicionar linha do DAP
                fig.add_trace(go.Scatter(x=dfCrescimento_DAP_HT1['idade'], y=dfCrescimento_DAP_HT1['Volume'], mode="lines", name="Volume"), secondary_y=False)

                # Atualizar eixos
                fig.update_xaxes(title_text="Idade (anos)")
                fig.update_yaxes(title_text="Volume (m³)", secondary_y=False)
                fig.update_yaxes(title_text="Volume (m³)", secondary_y=True)

                # Atualizar layout
                fig.update_layout(title="Simulação de desbaste", showlegend=True)

                fig2 = px.bar(df_Multiprodutos, x='Produto', y='precoProduto', title='Receita bruta')

                # Mostrar gráfico
                with col_grafico1:
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                fig3 = px.pie(df_Multiprodutos, values='VolumeDesbastado', names='Produto')

                with col_grafico2:
                    st.plotly_chart(fig3, use_container_width=True)    

                    fig4 = go.Figure(data=[go.Table(
                            header=dict(values= ['Produto', 'VolumeDesbastado', 'precoProduto'],
                                        fill_color="#1D250E",
                                        font_color="white",
                                        align='center'),
                            cells=dict(values=[df_Multiprodutos['Produto'], round(df_Multiprodutos['VolumeDesbastado'], 2), round(df_Multiprodutos['precoProduto'], 2)],
                                    fill_color='lavender',
                                    align='center'))
                                                ])

                    st.plotly_chart(fig4, use_container_width=True)

                    @st.cache_data
                    def convert_df(df):
                        # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode('utf-8')

                    csv2 = convert_df(df_Multiprodutos)
                    st.download_button(
                        label="Download - CSV",
                        data=csv2,
                        file_name='ReceitaMultiprodutos.csv',
                        mime='text/csv',
                    )         

            with sucupira_tab:

                especiesReceitaMadeireiras = ['Khaya grandifoliola', 'Cariniana legalis', 'Dalbergia nigra', 'Handroanthus serratifolius', 'Khaya ivorensis', 'Plathymenia reticulata']
                intensidade_padrao = {'Khaya grandifoliola': 0.3, 'Cariniana legalis': 0.1, 'Dalbergia nigra': 0.45, 'Handroanthus serratifolius': 0.1, 'Khaya ivorensis': 0.3, 'Plathymenia reticulata': 0.3}
                preco_padrao = {'Khaya grandifoliola': 2000.00, 'Cariniana legalis': 2000.00, 'Dalbergia nigra': 5000.00, 'Handroanthus serratifolius': 2500.00, 'Khaya ivorensis': 2000.00, 'Plathymenia reticulata': 2000.00}
                ciclo_corte_padrao = {'Khaya grandifoliola': 20, 'Cariniana legalis': 35, 'Dalbergia nigra': 45, 'Handroanthus serratifolius': 45, 'Khaya ivorensis': 20, 'Plathymenia reticulata': 30}
                Volume_padrao = {'Khaya grandifoliola': 0.8, 'Cariniana legalis': 1.0, 'Dalbergia nigra': 0.6, 'Handroanthus serratifolius': 0.7, 'Khaya ivorensis': 0.8, 'Plathymenia reticulata': 0.7}

                if sucupira_tab:

                    @st.cache_data
                    def carregarDados1():    
                        df = pd.read_csv("dados/dadosReceitaMadeira.csv")  

                        return df 
                    dadosReceitaMadeira = carregarDados1()

                    colParemetros1 = st.multiselect(
                            'Espécies', especiesReceitaMadeireiras, ['Khaya grandifoliola']
                            )

                    num_desbastesSucupira = 1 #int(st.number_input(" ", min_value=1, max_value=1, value=1))
                    colunas_desbaste = [st.columns(4) for _ in range(num_desbastesSucupira)]

                    

                    valores_por_especie = {}  # Crie um dicionário vazio para armazenar os valores para cada espécie

                    for j in colParemetros1:
                        valores_por_especie[j] = {}  # Crie um novo dicionário para a espécie atual
                        for i, (col_ano1, col_intensidade1, colPrecoProduto, colVolumeMadeira) in enumerate(colunas_desbaste):
                            with col_ano1:
                                ciclo_corte = int(st.number_input(f"Ciclo de corte (anos)? {j}", min_value=5, max_value=70, value=ciclo_corte_padrao[j]))
                                valores_por_especie[j]['Ciclo de Corte'] = ciclo_corte
                            with col_intensidade1:
                                intensidade_val1 = st.slider(f"Intensidade do desbaste? {j}", 0.0, 1.0, intensidade_padrao[j])
                                # Aqui você deve substituir 'Nome da coluna' pelo nome real da coluna
                                valores_por_especie[j]['Desbastes'] = intensidade_val1
                            with colPrecoProduto:
                                Preco_produto = float(st.number_input(f"Preço(R$/m³)? {j}", min_value=0.00, max_value=5000.00, value=preco_padrao[j]))
                                valores_por_especie[j]['Preço m³'] = Preco_produto
                            with colVolumeMadeira:
                                Volume_padrao_madeira = float(st.number_input(f"Volume (m³) {j}", min_value=0.00, max_value=4.00, value=Volume_padrao[j]))
                                valores_por_especie[j]['VolSerrArv (m³)'] = Volume_padrao_madeira


                    # Atualize as colunas de acordo com a espécie
                    for especie, valores in valores_por_especie.items():
                        for coluna, valor in valores.items():
                            dadosReceitaMadeira.loc[dadosReceitaMadeira['Especie'] == especie, coluna] = valor

                    dadosReceitaMadeira['QuantCortada'] = dadosReceitaMadeira['Total'] * dadosReceitaMadeira['Desbastes']
                    dadosReceitaMadeira['QuantCorteRaso'] = dadosReceitaMadeira['Total'] - dadosReceitaMadeira['QuantCortada']
                    dadosReceitaMadeira['VolumeTotal (m³)'] = dadosReceitaMadeira['QuantCorteRaso'] * dadosReceitaMadeira['VolSerrArv (m³)']
                    dadosReceitaMadeira['Valor total'] = dadosReceitaMadeira['VolumeTotal (m³)'] * dadosReceitaMadeira['Preço m³']
                    dadosReceitaMadeira['AnoCortePrevisto'] =  dadosReceitaMadeira['Ciclo de Corte'] - dadosReceitaMadeira['Idade'] + datetime.datetime.now().year

                    

                    especies_unicas = dadosReceitaMadeira['Especie'].unique()

                    df_AnoCortePrevisto = dadosReceitaMadeira.groupby(['AnoCortePrevisto', 'Especie']).sum().reset_index()
                    
                    soma_AnoCortePrevisto = df_AnoCortePrevisto.groupby('AnoCortePrevisto').sum().reset_index()

                    soma_AnoCortePrevisto['Especie'] = 'Total'
                    df_AnoCortePrevisto1 = pd.concat([df_AnoCortePrevisto, soma_AnoCortePrevisto], ignore_index=True)
                    df_AnoCortePrevisto1 = df_AnoCortePrevisto1.sort_values(by='Valor total', ascending=False)

                    # Crie um slider que retorna dois valores (um intervalo).

                    if st.checkbox('Agrupar datas de colheita de madeira'):
                        valores = st.slider(
                            ' ',   # deixair sem texto
                            2030,  # valor mínimo
                            2070,  # valor máximo
                            (2030, 2070)  # valores iniciais (min, max)
                        )

                        min, max = valores
                        filtro = df_AnoCortePrevisto1['AnoCortePrevisto'].between(min, max)
                        df_AnoCortePrevisto_filtrado = df_AnoCortePrevisto1.loc[filtro]

                        # Agora somamos todos os valores dentro do intervalo para cada especie
                        df_AnoCortePrevisto_filtradoSomado = df_AnoCortePrevisto_filtrado.groupby('Especie').sum().reset_index()

                        # Criamos um novo DataFrame com o ano máximo como o único valor no eixo x
                        df_final = pd.DataFrame({
                            'AnoCortePrevisto': [max] * len(df_AnoCortePrevisto_filtradoSomado),
                            'Especie': df_AnoCortePrevisto_filtradoSomado['Especie'],
                            'Valor total': df_AnoCortePrevisto_filtradoSomado['Valor total']
                        })

                        graficoAgrupadoCol1,  graficoAgrupadoCol2, graficoAgrupadoCol3 = st.columns([1, 3, 1])

                        with graficoAgrupadoCol2:
                            fig = px.bar(df_final, x="AnoCortePrevisto", y="Valor total", color="Especie", barmode="group")
                            st.plotly_chart(fig, use_container_width=True)

                    else:
                        fig = px.bar(df_AnoCortePrevisto1, x="AnoCortePrevisto", y="Valor total", color="Especie", barmode="group")
                        st.plotly_chart(fig, use_container_width=True)


                        baixarDados1,  baixarDados2, baixarDados3 = st.columns([1, 15, 1])

                        with baixarDados2:
                            dadosReceitaMadeira

                            st.write('Baixar dados')          


        st.markdown(
            '<hr style="border-top: 0.5px solid "#1D250E";">',
            unsafe_allow_html=True
        )


#Iniciando frutiferas
    elif Uso == 'Frutíferas' and Variavel == 'TPF':


        with st.expander("TPF"):
            st.info('''
            O TPF é o tempo de frutificação esperado para as frutíferas. Nesse caso, 
            o TPF adotado foram para as seguintes espécies: Açaí (Euterpe oleraceae), cacau
            (Theobroma cacao) e cupuaçu (Theobroma grandiflorum).
            ''')
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("Euterpe oleraceae")
                st.image("imagens/Acai.jpeg")

            with col2:
                st.header("Theobroma cacao")
                st.image("imagens/cacau.jpeg")

            with col3:
                st.header(("Theobroma grandiflorum"))
                st.image("imagens/cup.jpeg")

        
        acai_tab, cacau_tab, cup_tab, juc_tab, all_tab = st.tabs(["Açaí", "Cacau", "Cupuaçu", "Juçara", "Todas as frutíferas"])

        

#Açaí
        with acai_tab:

            if acai_tab:
                enh_expander = st.expander("Fatores que influencia a produção", expanded=False) 
                with enh_expander:

                    Horizonte_col1, producaoKG_col2, preco_col3 = st.columns(3) 
                    with Horizonte_col1:   
                        seed = int(st.number_input("Horizonte de produção", value=2040, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with producaoKG_col2:
                        quilosPlanta = float(st.number_input("Produção média esperada por planta (kg)", value=10.00, help="Escolha um intevalo para a produção esperada.",  min_value=0.00))

                    with preco_col3:
                        Preco = float(st.number_input("Preço de venda (R$/kg)", value=10.00, help="Escolha um intevalo para o preço comercializado.",  min_value=0.00))

                    st.markdown('<h5 style="color: gray;">Constante de produção</h3>', unsafe_allow_html=True)

                    ano_col1, ano_col2, ano_col3, ano_col4 = st.columns(4)
                    
                    with ano_col1:
                            
                            ano1_val = st.slider('Ano 1',0.0, 1.0, (0.08)) 

                    with ano_col2:
                            
                            ano2_val = st.slider('Ano 2',0.0, 1.0, (0.15))                          

                    with ano_col3:
                            
                            ano3_val = st.slider('Ano 3',0.0, 1.0, (0.4))

                    with ano_col4:
                            
                            ano4_val = st.slider('Ano 4',0.0, 1.0, (0.75))       

            col_acai1, col_acai2 = st.columns(2)
            
            with col_acai1:
                
                filtro = df["Especie"] == "Euterpe oleraceae"
                # Exibe apenas as linhas do dataframe que atendem ao filtro
                novo_df = df.loc[filtro, :]
                #resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
                resultado2_df = novo_df.groupby(['TPF']).count()
                resultado2_df= resultado2_df.reset_index()


                fig6 = px.bar(resultado2_df, x='TPF', y='Talhao',
                text='Talhao',
                title='Tempo para frutificação do açaí (<i>Euterpe oleraceae</i>)',
                labels={'TPF': 'TPF', 'Talhao': 'Quantidade'},
                template='plotly_white')
                fig6.update_traces(textposition='outside')
                #fig.update_layout(yaxis_range=[0, 70000])
                fig6.update_layout(xaxis_tickmode='linear')
                fig6.update_traces(marker_color="#1D250E")
                fig6.update_layout(height=500, width = 800)
                #fig6.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig6, use_container_width=True)
                

            with col_acai2:
                
                
                filtro = df["Especie"] == "Euterpe oleraceae"

                # Exibe apenas as linhas do dataframe que atendem ao filtro
                novo_df = df.loc[filtro, :]
                novo_df["Novo_TPF"] = novo_df.groupby(['TPF', 'Talhao'])['TPF'].transform('count')
                plantios = novo_df.groupby(['TPF', 'Talhao'])['Novo_TPF'].mean()
                plantios_df= plantios.reset_index()

                plantios_df1 = plantios_df.copy()
            
                teste = []
                producao = []
                producaoFruto = []
                Data = []
                anoInventario = 2023
                horizonte = seed - 2022
                Talhao = []
                QuantidadeIndividuos = []

                for h in range(horizonte):
                    for i, row in plantios_df1.iterrows():
                        TPF, qntdade, talhao = int(row['TPF']), int(row['Novo_TPF']), row['Talhao']
                                            
                        if TPF > 0:
                            producao.append(qntdade * 0 *quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append((qntdade * 0 *quilosPlanta)*2)
                        elif TPF == 0:
                            producao.append(qntdade * ano1_val * quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.08 *quilosPlanta*2)
                        elif TPF == -1:
                            producao.append(qntdade * ano2_val* quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.15 *quilosPlanta*2)
                        elif TPF == -2:
                            producao.append(qntdade * ano3_val* quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.4 *quilosPlanta*2)
                        elif TPF == -3:
                            producao.append(qntdade * ano4_val* quilosPlanta)
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
                resultado_exibir = resultado.groupby(['Data', 'Talhao'])['Produção'].sum()
                resultado_exibir_acai= resultado_exibir.reset_index()
                resultado_exibir_acai['ValorProduto'] = resultado_exibir_acai['Produção'] * Preco
                resultado_exibir_acai['Espécie'] = 'Açaí'
                #resultado_df

                fig7 = px.bar(resultado_df, x='Data', y='Produção',                
                text='Produção',
                title='Produção de acaí (<i>Euterpe oleraceae</i>) ao longo dos anos',
                labels={'Data': 'Ano', 'Produção': 'Produção de polpa (kg)'},
                template='plotly_white')
                fig7.update_traces(textposition='outside')
                fig7.update_layout(yaxis_range=[0, resultado_df['Produção'].max()*1.2])
                fig7.update_layout(xaxis_tickmode='linear')
                fig7.update_traces(marker_color="#1D250E")
                fig7.update_traces(text=round(resultado_df['Produção'], 2))
                fig7.update_layout(height=500, width = 800)
                #fig7.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig7, use_container_width=True)
                
            dados_expander = st.expander("Exibir dados por talhão", expanded=False)
            with dados_expander:
                st.write(resultado_exibir_acai)
                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_exibir_acai)

                st.download_button(
                    label="Download - CSV",
                    data=csv,
                    file_name='Euterpe_oleraceae.csv',
                    mime='text/csv',
                )



# Cacau
        with cacau_tab:
            if cacau_tab:

                enh_expander = st.expander("Fatores que influencia a produção", expanded=False) 
                with enh_expander:

                    cacau_Horizonte_col1, cacau_producaoKG_col2, cacau_preco_col3 = st.columns(3) 
                    with cacau_Horizonte_col1:   
                        cacau_seed = int(st.number_input("Horizonte de produção ", value=2040, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with cacau_producaoKG_col2:
                        cacau_quilosPlanta = float(st.number_input("Produção média esperada por planta (kg) ", value=1.5, help="Escolha um intevalo para a produção esperada.",  min_value=0.0))

                    with cacau_preco_col3:
                        cacau_Preco = float(st.number_input("Preço de venda (R$/kg) ", value=20.00, help="Escolha um intevalo para o preço comercializado.",  min_value=0.00))

                    st.markdown('<h5 style="color: gray;">Constante de produção</h3>', unsafe_allow_html=True)

                    ano_cacau_col1, ano_cacau_col2, ano_cacau_col3, ano_cacau_col4 = st.columns(4)
                    
                    with ano_cacau_col1:
                            
                            cacau_ano1_val = st.slider('Ano 1 ',0.0, 1.0, (0.08)) 

                    with ano_cacau_col2:
                            
                            cacau_ano2_val = st.slider('Ano 2 ',0.0, 1.0, (0.15))                          

                    with ano_cacau_col3:
                            
                            cacau_ano3_val = st.slider('Ano 3 ',0.0, 1.0, (0.4))

                    with ano_cacau_col4:
                            
                            cacau_ano4_val = st.slider('Ano 4 ',0.0, 1.0, (0.75))       

            col_cacau1, col_cacau2 = st.columns(2)
            
            with col_cacau1:
                
                filtro = df["Especie"] == "Theobroma cacao"
                # Exibe apenas as linhas do dataframe que atendem ao filtro
                novo_df = df.loc[filtro, :]
                #resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
                resultado2_df = novo_df.groupby(['TPF']).count()
                resultado2_df= resultado2_df.reset_index()


                fig6 = px.bar(resultado2_df, x='TPF', y='Talhao',
                text='Talhao',
                title='Tempo para frutificação do cacau (<i>Theobroma cacao</i>)',
                labels={'TPF': 'TPF', 'Talhao': 'Quantidade'},
                template='plotly_white')
                fig6.update_traces(textposition='outside')
                #fig.update_layout(yaxis_range=[0, 70000])
                fig6.update_layout(xaxis_tickmode='linear')
                fig6.update_traces(marker_color="#1D250E")
                fig6.update_layout(height=500, width = 800)
                #fig6.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig6, use_container_width=True)
                

            with col_cacau2:
                
                
                filtro = df["Especie"] == "Theobroma cacao"

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
                horizonte = cacau_seed - 2022
                Talhao = []

                for h in range(horizonte):
                    for i, row in plantios_df1.iterrows():
                        TPF, qntdade, talhao = int(row['TPF']), int(row['Novo_TPF']), row['Talhao']
                        if TPF > 0:
                            producao.append(qntdade * 0 *cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append((qntdade * 0 *cacau_quilosPlanta)*2)
                        elif TPF == 0:
                            producao.append(qntdade * cacau_ano1_val * cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.08 *cacau_quilosPlanta*2)
                        elif TPF == -1:
                            producao.append(qntdade * cacau_ano2_val* cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.15 *cacau_quilosPlanta*2)
                        elif TPF == -2:
                            producao.append(qntdade * cacau_ano3_val* cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.4 *cacau_quilosPlanta*2)
                        elif TPF == -3:
                            producao.append(qntdade * cacau_ano4_val* cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.75 *cacau_quilosPlanta*2)
                        elif TPF == -4:
                            producao.append(qntdade * 1* cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 1 *cacau_quilosPlanta*2)
                        elif TPF <-4:
                            producao.append(qntdade * 1* cacau_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 1 *cacau_quilosPlanta*2)
                        else:
                            pass
                        
                        plantios_df1.at[i, 'TPF'] = TPF - 1


                resultado = pd.DataFrame({'Produção': producao, 'Data': Data, 'Talhao': Talhao, 'producaoFruto': producaoFruto})
                #resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
                resultado_df = resultado.groupby(['Data'])['Produção'].sum()
                resultado_df= resultado_df.reset_index()
                #resultado_df

                resultado_exibir = resultado.groupby(['Data', 'Talhao'])['Produção'].sum()
                resultado_exibir_cacau= resultado_exibir.reset_index()
                resultado_exibir_cacau['ValorProduto'] = resultado_exibir_cacau['Produção'] * cacau_Preco
                resultado_exibir_cacau['Espécie'] = 'Cacau'
                #resultado_df

                fig7 = px.bar(resultado_df, x='Data', y='Produção',                
                text='Produção',
                title='Produção de cacau (<i>Theobroma cacao</i>) ao longo dos anos',
                labels={'Data': 'Ano', 'Produção': 'Produção de amêndoa (kg)'},
                template='plotly_white')
                fig7.update_traces(textposition='outside')
                fig7.update_layout(yaxis_range=[0, resultado_df['Produção'].max()*1.2])
                fig7.update_layout(xaxis_tickmode='linear')
                fig7.update_traces(marker_color="#1D250E")
                fig7.update_traces(text=round(resultado_df['Produção'], 2))
                fig7.update_layout(height=500, width = 800)
                #fig7.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig7, use_container_width=True)

            dados_expander = st.expander("Exibir dados por talhão", expanded=False)
            with dados_expander:
                st.write(resultado_exibir_cacau)
                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_exibir_cacau)

                st.download_button(
                    label="Download - CSV",
                    data=csv,
                    file_name='Theobroma_cacao.csv',
                    mime='text/csv',
                )

# Cupuaçu
        with cup_tab:

            if cup_tab:

                enh_expander = st.expander("Fatores que influencia a produção", expanded=False) 
                with enh_expander:

                    cup_Horizonte_col1, cup_producaoKG_col2, cup_preco_col3 = st.columns(3) 
                    with cup_Horizonte_col1:   
                        cup_seed = int(st.number_input("Horizonte de produção  ", value=2040, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with cup_producaoKG_col2:
                        cup_quilosPlanta = float(st.number_input("Produção média esperada por planta (kg)  ", value=5.0, help="Escolha um intevalo para a produção esperada.",  min_value=0.0))

                    with cup_preco_col3:
                        cup_Preco = float(st.number_input("Preço de venda (R$/kg)  ", value=10.00, help="Escolha um intevalo para o preço comercializado.",  min_value=0.00))

                    st.markdown('<h5 style="color: gray;">Constante de produção</h3>', unsafe_allow_html=True)

                    ano_cup_col1, ano_cup_col2, ano_cup_col3, ano_cup_col4 = st.columns(4)
                    
                    with ano_cup_col1:
                            
                            cup_ano1_val = st.slider('Ano 1  ',0.0, 1.0, (0.08)) 

                    with ano_cup_col2:
                            
                            cup_ano2_val = st.slider('Ano 2  ',0.0, 1.0, (0.15))                          

                    with ano_cup_col3:
                            
                            cup_ano3_val = st.slider('Ano 3  ',0.0, 1.0, (0.4))

                    with ano_cup_col4:
                            
                            cup_ano4_val = st.slider('Ano 4  ',0.0, 1.0, (0.75))       

            col_cup1, col_cup2 = st.columns(2)
            
            with col_cup1:
                
                filtro = df["Especie"] == "Theobroma grandiflorum"
                # Exibe apenas as linhas do dataframe que atendem ao filtro
                novo_df = df.loc[filtro, :]
                #resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
                resultado2_df = novo_df.groupby(['TPF']).count()
                resultado2_df= resultado2_df.reset_index()


                fig6 = px.bar(resultado2_df, x='TPF', y='Talhao',
                text='Talhao',
                title='Tempo para frutificação do cupuaçu (<i>Theobroma grandiflorum</i>)',
                labels={'TPF': 'TPF', 'Talhao': 'Quantidade'},
                template='plotly_white')
                fig6.update_traces(textposition='outside')
                #fig.update_layout(yaxis_range=[0, 70000])
                fig6.update_layout(xaxis_tickmode='linear')
                fig6.update_layout(height=500, width = 800)
                fig6.update_traces(marker_color="#1D250E")
                #fig6.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig6, use_container_width=True)
                

            with col_cup2:
                
                
                filtro = df["Especie"] == "Theobroma grandiflorum"

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
                horizonte = cup_seed - 2022
                Talhao = []

                for h in range(horizonte):
                    for i, row in plantios_df1.iterrows():
                        TPF, qntdade, talhao = int(row['TPF']), int(row['Novo_TPF']), row['Talhao']
                        if TPF > 0:
                            producao.append(qntdade * 0 *cup_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append((qntdade * 0 *cup_quilosPlanta)*2)
                        elif TPF == 0:
                            producao.append(qntdade * cup_ano1_val * cup_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.08 *cup_quilosPlanta*2)
                        elif TPF == -1:
                            producao.append(qntdade * cup_ano2_val* cup_quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.15 *cup_quilosPlanta*2)
                        elif TPF == -2:
                            producao.append(qntdade * cup_ano3_val* quilosPlanta)
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(qntdade * 0.4 *quilosPlanta*2)
                        elif TPF == -3:
                            producao.append(qntdade * cup_ano4_val* quilosPlanta)
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

                resultado_exibir = resultado.groupby(['Data', 'Talhao'])['Produção'].sum()
                resultado_exibir_cup= resultado_exibir.reset_index()
                resultado_exibir_cup['ValorProduto'] = resultado_exibir_cup['Produção'] * cup_Preco
                resultado_exibir_cup['Espécie'] = 'Cupuaçu'
                #resultado_df


                fig7 = px.bar(resultado_df, x='Data', y='Produção',                
                text='Produção',
                title='Produção de cupuaçu (<i>Theobroma grandiflorum</i>) ao longo dos anos',
                labels={'Data': 'Ano', 'Produção': 'Produção de polpa (kg)'},
                template='plotly_white')
                fig7.update_traces(textposition='outside')
                fig7.update_layout(yaxis_range=[0, resultado_df['Produção'].max()*1.2])
                fig7.update_layout(xaxis_tickmode='linear')
                fig7.update_traces(marker_color="#1D250E")
                fig7.update_traces(text=round(resultado_df['Produção'], 2))
                fig7.update_layout(height=500, width = 800)
                #fig7.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig7, use_container_width=True)

            dados_expander = st.expander("Exibir dados por talhão", expanded=False)
            with dados_expander:
                st.write(resultado_exibir_cup)
                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_exibir_cup)

                st.download_button(
                    label="Download - CSV",
                    data=csv,
                    file_name='Theobroma_grandiflorum.csv',
                    mime='text/csv',
                )


        with juc_tab:
            if juc_tab:

                enh_expander = st.expander("Fatores que influencia a produção", expanded=False) 
                with enh_expander:

                    #st.markdown('<h5 style="color: gray;">Constante de produção</h3>', unsafe_allow_html=True)
                    juc_Horizonte_col1, juc_producaoKG_col2, juc_preco_col3, juc_incremento_col4,  juc_DAP_MIN, juc_DAP_Max = st.columns(6) 
                    with juc_Horizonte_col1:   
                        juc_seed = int(st.number_input("Horizonte de produção    ", value=2040, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with juc_producaoKG_col2:
                        juc_quilosPlanta = int(st.number_input("Produção média/planta (kg)    ", value=10, help="Escolha um intevalo para a produção esperada.",  min_value=0))

                    with juc_preco_col3:
                        juc_Preco = float(st.number_input("Preço de venda (R$/kg)    ", value=10.00, help="Escolha um intevalo para o preço comercializado.",  min_value=0.00))

                    with juc_incremento_col4:
                        incremento = float(st.number_input("IMA", value=0.8, help="Escolha um intevalo para o preço comercializado.",  min_value=0.0))

                    with juc_DAP_MIN:
                        DAP_MIN = float(st.number_input("DAP min", value=7.0, help="Escolha um intevalo para o preço comercializado.",  min_value=0.0))

                    with juc_DAP_Max:
                        DAP_MAX = float(st.number_input("DAP max", value=12.0, help="Escolha um intevalo para o preço comercializado.",  min_value=DAP_MIN))

                    

                # Filtra o dataframe com base na coluna "Especie"
                filtro = df["Especie"] == "Euterpe edulis"
                novo_df = df.loc[filtro, :]
                novo_df['Novo_DAP'] = novo_df['DAP']

                plantios_df1 = novo_df
                producao = []
                producaoFruto = []
                Data = []
                anoInventario = 2023
                horizonte = juc_seed - 2022
                Talhao = []
                dap_novo = []
                amplitude = (DAP_MAX - DAP_MIN) / 5
                        
                for h in range(horizonte):
                    for i, row in plantios_df1.iterrows():
                        DAP, talhao = row['Novo_DAP'], row['Talhao']
                        if DAP > 0 and DAP <= DAP_MIN:
                            producao.append(0 *juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)            
                            producaoFruto.append((0 *juc_quilosPlanta )*2)
                        elif DAP > DAP_MIN and DAP<= DAP_MIN + amplitude:
                            producao.append(0.08 * juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(0.08 *juc_quilosPlanta *2)
                        elif DAP > DAP_MIN + amplitude and DAP<= DAP_MIN + 2 * amplitude:
                            producao.append(0.15* juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(0.15 *juc_quilosPlanta *2)
                        elif DAP > DAP_MIN + 2 * amplitude and DAP<= DAP_MIN + 3 * amplitude:
                            producao.append(0.4* juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(0.4 *juc_quilosPlanta *2)
                        elif DAP > DAP_MIN + 3 * amplitude and DAP<= DAP_MIN + 4 * amplitude:
                            producao.append(0.75* juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(0.75 *juc_quilosPlanta *2)
                        elif DAP > DAP_MIN + 4 * amplitude and DAP<= DAP_MIN + 5 * amplitude:
                            producao.append(1* juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(1 *juc_quilosPlanta *2)
                        elif DAP > DAP_MIN + 5 * amplitude:
                            producao.append(1* juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)
                            producaoFruto.append(1 *juc_quilosPlanta *2)
                        else:
                            DAP = DAP + 2
                            producao.append(0 *juc_quilosPlanta )
                            Data.append(anoInventario + h)
                            Talhao.append(talhao)            
                            producaoFruto.append((0 *juc_quilosPlanta )*2)
                        
                        plantios_df1.at[i, 'Novo_DAP'] = DAP + incremento

                resultado = pd.DataFrame({'Produção': producao, 'Data': Data, 'Talhao': Talhao, 'producaoFruto': producaoFruto})
                #resultado_df = resultado.groupby(['Talhao','Data'])['Produção'].sum()
                resultado_df = resultado.groupby(['Data'])['Produção'].sum()
                resultado_df= resultado_df.reset_index()

                resultado_exibir = resultado.groupby(['Data', 'Talhao'])['Produção'].sum()
                resultado_exibir_juc = resultado_exibir.reset_index()
                resultado_exibir_juc['ValorProduto'] = resultado_exibir_juc['Produção'] * juc_Preco
                resultado_exibir_juc['Espécie'] = 'Juçara'
                #resultado_df

                fig8 = px.bar(resultado_df, x='Data', y='Produção',                
                text='Produção',
                title='Produção de açaí (<i>Euterpe Edulis</i>) ao longo dos anos',
                labels={'Data': 'Ano', 'Produção': 'Produção de polpa (kg)'},
                template='plotly_white')
                fig8.update_traces(textposition='outside')
                fig8.update_layout(yaxis_range=[0, resultado_df['Produção'].max()*1.2])
                fig8.update_layout(xaxis_tickmode='linear')
                fig8.update_traces(marker_color="#1D250E")
                fig8.update_traces(text=round(resultado_df['Produção'], 2))
                fig8.update_layout(height=500, width = 800)
                #fig8.update_layout(plot_bgcolor="#FFFAFA")

                st.plotly_chart(fig8, use_container_width=True)
            dados_expander = st.expander("Exibir dados por talhão", expanded=False)
            with dados_expander:
                st.write(resultado_exibir_juc)
                #st.dataframe(resultado_resultado_exibir)
                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_exibir_juc)

                st.download_button(
                    label="Download - CSV",
                    data=csv,
                    file_name='Euterpe_Edulis.csv',
                    mime='text/csv',
                )

#Com todas as frutas

        with all_tab:

            # Cria a caixa de seleção
            colPorcentagemComercializada1, colPorcentagemComercializada2 = st.columns(2)

            with colPorcentagemComercializada1:                
                porcentagemComercializada = float(st.slider('Porcentagem comercializada', 0.0, 1.0, 1.0))

            colReceita1, colReceita2 = st.columns(2)

            df_concat = pd.concat([resultado_exibir_acai, resultado_exibir_cacau, resultado_exibir_cup, resultado_exibir_juc], ignore_index=True)  
            df_ProducaoEspecieData = df_concat.groupby(['Espécie', 'Data']).sum().reset_index()
            df_ProducaoEspecieData['ValorProduto'] = df_ProducaoEspecieData['ValorProduto']  * porcentagemComercializada
            df_ProducaoEspecieData['Produção'] = df_ProducaoEspecieData['Produção']  * porcentagemComercializada

            df_sum = df_ProducaoEspecieData.groupby('Data').sum().reset_index()

            df_sum['Espécie'] = 'Total'
            df_Producao = pd.concat([df_ProducaoEspecieData, df_sum], ignore_index=True)
            df_Producao = df_Producao.sort_values(by='ValorProduto', ascending=False)
            df_Producao['ValorProduto'] = round(df_Producao['ValorProduto'], 2)
            df_Producao['Produção'] = round(df_Producao['Produção'], 2)

            with colReceita1:
                fig = px.bar(df_Producao, x="Data", y="ValorProduto", color="Espécie", barmode="group") 
                st.plotly_chart(fig)

                # Cria um slider que vai de 2023 até 2050
                
                ano_selecionado = int(st.slider('Selecione um ano', 2023, 2050, 2023))                
                fig2 =  px.pie(df_Producao.loc[df_Producao['Data'] == ano_selecionado], values='ValorProduto', names='Espécie', hole=.4)
                st.plotly_chart(fig2)
                
                

            # Transforma o dataframe em uma tabela pivot
            table = pd.pivot_table(df_Producao, values='ValorProduto', index='Data', columns='Espécie')

            # Adiciona rótulos às linhas e colunas
            table.index.name = 'Data'
            table.columns.name = 'Espécie'
            # Cria a coluna de rótulo
            rotulo = ['Ano'] + table.columns.tolist()

            #Para a produção
            # Transforma o dataframe em uma tabela pivot
            tableProducao = pd.pivot_table(df_Producao, values='Produção', index='Data', columns='Espécie')

            # Adiciona rótulos às linhas e colunas
            tableProducao.index.name = 'Data'
            tableProducao.columns.name = 'Espécie'

            # Cria a coluna de rótulo
            rotulo = ['Ano'] + tableProducao.columns.tolist()

            # Cria a figura da tabela e Formata a figura da tabela
            with colReceita2:

                st.markdown('<h3 style="text-align: center; display: flex; align-items: center; justify-content: center;">Receita bruta esperada (R$)</h3>', unsafe_allow_html=True)
                fig3 = go.Figure(data=[go.Table(
                    header=dict(values=rotulo,
                                fill_color='#1E90FF',
                                font=dict(color='white', size=14),
                                align='center'),
                    cells=dict(values=[table.reset_index()['Data']] + [table[col] for col in table.columns],
                            fill_color='lavender',
                            align='center'))
                ])

                
                st.plotly_chart(fig3)

                st.markdown('<h3 style="text-align: center; display: flex; align-items: center; justify-content: center;">Produção esperada (kg)</h3>', unsafe_allow_html=True)
                
                fig4 = go.Figure(data=[go.Table(
                    header=dict(values=rotulo,
                                fill_color='#1E90FF',
                                font=dict(color='white', size=14),
                                align='center'),
                    cells=dict(values=[tableProducao.reset_index()['Data']] + [tableProducao[col] for col in tableProducao.columns],
                            fill_color='lavender',
                            align='center'))
                ])
                
                
                st.plotly_chart(fig4)

                st.write('Baixar os dados organizados')

                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv_ProducaoEspecieData = convert_df(df_ProducaoEspecieData)
                st.download_button(
                    label="Download - CSV",
                    data=csv_ProducaoEspecieData,
                    file_name='Dados_Producao.csv',
                    mime='text/csv',
                ) 

    elif Uso == 'Frutíferas' and Variavel == 'Fenofase':

        with st.expander("Fenofase"):
            st.info("""
            A fenofase refere-se à condição fenológica em que a espécie encontra-se. Por exemplo,
            se está com frutos, flor, caixos, vegetativa ou uma combinação dessas variáveis.
            """)

    elif Uso == 'Frutíferas' and Variavel == 'Vigor':
        with st.expander("Vigor"):
            st.info("""
            O vigor reflete algumas manifestações da planta com relação ao seu desenvolvimeto
            e ao local de estabelecimento como estresse nutricional e hídrico, 
            ataque por pragas e doenças e iluminação.
            """)

    elif Uso == 'Especiárias' and Variavel == 'Fenofase':
        pass

    elif Uso == 'IRP':

        df["AB"] = df.loc[~df['DAP'].isnull()].groupby(['Talhao'])['DAP'].transform(lambda x: x **2 * np.pi / 40000)
        df1 = df.groupby('Talhao').agg({'area': 'mean', 'idade': 'mean', 'AB': 'sum'}).reset_index()

        contagem = df.groupby(["Talhao", "Uso"]).size().unstack(fill_value=0).reset_index()
    
        df3 = pd.merge(df1, contagem, on='Talhao')

        # definindo a função para dividir as colunas pela coluna 'area'
        df_novo = df3.copy()
        for coluna in df_novo.columns:
            if coluna not in ['Talhao', 'idade', 'area']:
                df_novo[coluna] = df_novo[coluna] / df_novo['area']

        def calcular_indice(frut, ab, idade, mad, const_frut, const_mad, const_area_basal):
            return ( frut**const_frut  * (ab ** const_area_basal / idade) - mad) / ( mad**const_mad * (ab / idade) + frut) 
        

        with st.expander(" ", expanded=True):

            st.write('## Índice de Relação Produtiva - IRP')
            c1, c2, c3 = st.columns(3)

            with c2:
                st.markdown("<span style='font-size: 20pt'>" +
                    r"$\frac{frut^{A} \cdot (\frac{ab^{C}}{idade}) - mad}{mad^{B} \cdot (\frac{ab}{idade}) + frut}$" +
                    "</span>", unsafe_allow_html=True)

            st.write("""
                ### Descrição das variáveis                

                1. `frut`: variável numérica que representa a quantidade de plantas frutíferas.

                2. `ab`: variável numérica que representa a área basal das espécies madeireiras.

                3. `idade`: variável numérica que representa a idade da planta em anos.

                4. `mad`: variável numérica que representa a quantidade de indivíduos para uso madeireiro.

                5. `A`: constante numérica que é utilizada para ajustar a importância da variável `frut` na função.

                6. `B`: constante numérica que é utilizada para ajustar a importância da variável `mad` na função.

                7. `C`: constante numérica que é utilizada para ajustar a importância da variável `ab` na função. 

                Essas variáveis são utilizadas para calcular um índice que depende da quantidade de frutíferas, da área basal, da idade
                e da quantidade de indivíduos de uso madeireiro, bem como de constantes numéricas que influenciam o peso relativo de 
                cada uma dessas variáveis no cálculo do índice.
                """)     

            frut, mad, area_basal = st.columns(3)
            with frut:
                const_frut = float(st.number_input("A", value=1.0, help="Escolha um peso para as frutíferas.",  min_value=0.0))

            with mad:
                const_mad = float(st.number_input("B", value=1.0, help="Escolha um peso para as madeireiras.",  min_value=0.0))

            with area_basal:
                const_area_basal = float(st.number_input("C", value=1.0, help="Escolha um peso para a área basal.",  min_value=0.0))

        g1, g2 = st.columns(2)
        with g1:

            dados2 = df.groupby(['Talhao', 'Uso']).size().reset_index(name='counts')
            dados2['prop'] = dados2.groupby(['Talhao'])['counts'].apply(lambda x: x / x.sum()).reset_index(drop=True)
            dados2 = dados2.sort_values(by=['Uso', 'prop'])
            dados2['talhao1'] = pd.Categorical(dados2['Talhao'], categories=dados2['Talhao'].unique(), ordered=True)

            dados2['Uso'] = pd.Categorical(dados2['Uso'], categories=dados2['Uso'].unique(), ordered=True)
            dados2 =  dados2.reset_index()            

            fig_Q_F = px.bar(dados2, y='Talhao', x='prop', color='Uso',
            orientation='h', text="counts",
            color_discrete_sequence=["#1D250E","#556B2F",  "#2F4F4F"])
            fig_Q_F.update_layout(title={'text': "Uso da espécie", 'x':0.5, 'xanchor': 'center'},
            xaxis_title="Proporção",
            yaxis_title="Talhões")
            #fig_Q_F.update_layout(plot_bgcolor="#FFFAFA")
            fig_Q_F.update_layout(height=600, width = 500)                         

            st.plotly_chart(fig_Q_F, use_container_width=True)

            fig1 = px.bar(df_novo.sort_values(by='AB', ascending=True), 
                        y='Talhao', x='AB',
                        orientation='h')
            
            fig1.update_traces(textposition='outside')
            #fig.update_layout(yaxis_range=[0, 70000])
            fig1.update_layout(xaxis_tickmode='linear')
            fig1.update_traces(marker_color="#1D250E")
            fig1.update_layout(height=600, width = 500)
            st.plotly_chart(fig1, use_container_width=True)
    

        with g2:

            # criar o gráfico
            df_novo['indice'] = calcular_indice(df_novo['Frutífera'], df_novo['AB'], df_novo['idade'], df_novo['Madeireira'], const_frut, const_mad, const_area_basal)
            # ordenar o dataframe pela coluna 'indice'
            df_novo= df_novo.sort_values(by='indice', ascending=True)

            fig = px.bar(df_novo, y='Talhao', x='indice',
                        orientation='h')
            fig.update_traces(textposition='outside')
            #fig.update_layout(yaxis_range=[0, 70000])
            fig.update_layout(xaxis_tickmode='linear')
            fig.update_traces(marker_color="#1D250E")
            fig.update_layout(height=600, width = 500)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(df_novo.sort_values(by='idade', ascending=True), 
                y='Talhao', x='idade',
                orientation='h')
            
            fig2.update_traces(textposition='outside')
            #fig.update_layout(yaxis_range=[0, 70000])
            fig2.update_layout(xaxis_tickmode='linear')
            fig2.update_traces(marker_color="#1D250E")
            fig2.update_layout(height=600, width = 500)
            st.plotly_chart(fig2, use_container_width=True)


        # Exibir o dataframe
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.write(df_novo)

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_novo)

        st.download_button(
            label="Download - CSV",
            data=csv,
            file_name='indice_agroflorestal.csv',
            mime='text/csv',
        )

    elif Uso == 'Análise de solos':


        #teste de importação
        @st.cache_data
        def carregarDadosAnaliseSolo(excel_bytes, sheet_name):
            import io
            if not excel_bytes:
                return pd.DataFrame()
            try:
                return pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name, engine='openpyxl')
            except Exception as e:
                st.error(f"Falha ao ler a aba '{sheet_name}'. Confira o arquivo: {e}")
                return pd.DataFrame()

        st.title('Aplicativo de Análise de Solo')

        # Upload do arquivo Excel
        uploaded_file = st.file_uploader("Escolha um arquivo Excel", type="xlsx")

        if uploaded_file is not None:
            # Ler bytes uma única vez e reutilizar
            if 'solo_excel_bytes' not in st.session_state or st.session_state.get('solo_prev', {}).get('file') != getattr(uploaded_file, 'name', None):
                st.session_state['solo_excel_bytes'] = uploaded_file.getvalue()
            import io
            try:
                xls = pd.ExcelFile(io.BytesIO(st.session_state['solo_excel_bytes']), engine='openpyxl')
            except Exception as e:
                st.error(f"Falha ao abrir o Excel. Confira o arquivo: {e}")
                st.stop()
            # Seleção da aba do Excel para cada análise
            sheet_analise = st.selectbox("Selecione a aba para Análise de Talhão:", xls.sheet_names)
            sheet_referencia = st.selectbox("Selecione a aba para Análise de Referência:", xls.sheet_names)
            sheet_area_talhao = st.selectbox("Selecione a aba para Área do Talhão:", xls.sheet_names)

            # Resetar confirmação se arquivo/abas mudarem
            if 'solo_prev' not in st.session_state:
                st.session_state['solo_prev'] = {}
            current_sel = {
                'file': getattr(uploaded_file, 'name', None),
                'analise': sheet_analise,
                'referencia': sheet_referencia,
                'area': sheet_area_talhao,
            }
            if st.session_state['solo_prev'] != current_sel:
                st.session_state['solo_prev'] = current_sel
                st.session_state['solo_dados_carregados'] = False

            # Aguardar seleção explícita do usuário antes de processar
            if 'solo_dados_carregados' not in st.session_state:
                st.session_state['solo_dados_carregados'] = False

            carregar = st.button('Carregar dados', type='primary', help='Selecione as três abas e clique para prosseguir.')

            if not st.session_state['solo_dados_carregados']:
                # Não prossegue até que o usuário confirme
                if carregar:
                    st.session_state['solo_dados_carregados'] = True
                else:
                    # Fica em branco até confirmação do usuário
                    st.stop()

            # Carregar os dados após confirmação do usuário
            excel_bytes = st.session_state.get('solo_excel_bytes')
            analiseTalhao = carregarDadosAnaliseSolo(excel_bytes, sheet_analise)
            analiseReferencia = carregarDadosAnaliseSolo(excel_bytes, sheet_referencia)
            areaTalhao = carregarDadosAnaliseSolo(excel_bytes, sheet_area_talhao)

            if not analiseTalhao.empty and not analiseReferencia.empty and not areaTalhao.empty:
                # Garantir que as colunas esperadas existam antes de prosseguir
                missing = []
                if 'Talhao' not in analiseTalhao.columns:
                    missing.append("Análise de Talhão: 'Talhao'")
                if 'especie' not in analiseReferencia.columns:
                    missing.append("Análise de Referência: 'especie'")
                if missing:
                    st.error("Confira as tabelas: campos obrigatórios ausentes -> " + ", ".join(missing))
                    st.stop()
                # Lista com os talhões
                talhoes = analiseTalhao['Talhao'].unique()
                especie = analiseReferencia['especie'].unique()

                col1, col2 = st.columns(2)

                with col1:
                    talhoes_selecionados = st.multiselect('Selecione os talhões:', talhoes)

                with col2:
                    especie_selecionada = st.selectbox('Selecione a espécie:', especie)

                colGrafico1, colTabela2 = st.columns(2)

                with  colGrafico1:
                    import plotly.express as px
                    import pandas as pd

                    # Inicializar a figura
                    fig = go.Figure()

                    # Preparar referência pela espécie selecionada e colunas comuns numéricas
                    ref_df = analiseReferencia.loc[analiseReferencia['especie'] == especie_selecionada]
                    if ref_df.empty:
                        st.error("Confira as tabelas: espécie selecionada não encontrada na Análise de Referência.")
                        st.stop()
                    ref_row = ref_df.iloc[:1, 1:].apply(pd.to_numeric, errors='coerce')

                    talhao_num_cols = list(analiseTalhao.columns[1:])
                    ref_num_cols = list(ref_row.columns)
                    common_cols = [c for c in talhao_num_cols if c in ref_num_cols]
                    if not common_cols:
                        st.error("Confira as tabelas: não há colunas numéricas comuns entre Talhão e Referência.")
                        st.stop()

                    # Adicionar linha zero (ideal) nas colunas comuns
                    fig.add_trace(go.Scatterpolar(
                        r=[0]*len(common_cols),
                        theta=common_cols,
                        fill='toself',
                        name='Ideal',
                        line=dict(color='grey', width=1)
                    ))

                    # Adicionar linhas para cada talhão selecionado
                    for talhao_selecionado in talhoes_selecionados:
                        df_porcentagem = analiseTalhao.loc[analiseTalhao['Talhao'] == talhao_selecionado].copy()
                        if df_porcentagem.empty:
                            continue
                        # Converter numéricas e alinhar pelas colunas comuns
                        df_porcentagem[common_cols] = df_porcentagem[common_cols].apply(pd.to_numeric, errors='coerce')
                        ref_vals = ref_row[common_cols].values
                        df_porcentagem.loc[:, common_cols] = ((df_porcentagem[common_cols] * 100) / ref_vals) - 100

                        fig.add_trace(go.Scatterpolar(
                            r=df_porcentagem[common_cols].iloc[0].values,
                            theta=common_cols,
                            fill='toself',
                            name=talhao_selecionado
                        ))

                    # Atualizar layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[-100, 100]
                            )
                        ),
                        showlegend=True,
                        height=650,  # Ajuste o valor de altura conforme necessário
                        title={
                            'text': "<b>Fertigrama</b>",
                            'y':1,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        title_font=dict(
                            size=30,
                        )
                    )

                    # Renderize a figura
                    st.plotly_chart(fig)
                
                with colTabela2:
                    st.write("")           
                    st.write("Análise de solos")            
                    st.write(analiseTalhao)

                    st.write(f"Valor referência para o {especie_selecionada}")
                    st.write(analiseReferencia)

                # (Controles de PCA movidos para dentro do expander)
                

                import pandas as pd
                from sklearn.decomposition import PCA
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                enh_PCA = st.expander("Análise de PCA e Agrupamento", expanded=False)
                with enh_PCA:
                    # Filtros e parâmetros da PCA dentro do expander
                    colFatorAgrupamento1, colPCA2, colcluster3, colVarPCA4 = st.columns(4)
                    with colFatorAgrupamento1:
                        colunas = analiseTalhao.columns
                        fatorAgrupamento = st.selectbox('Fator de agrupamento', colunas)
                        if analiseTalhao[fatorAgrupamento].dtype != 'object':
                            st.error('Erro: A coluna selecionada não é categórica')
                    # Determinar colunas numéricas para PCA (exclui a coluna de agrupamento)
                    numeric_cols = analiseTalhao.drop(columns=[fatorAgrupamento], errors='ignore').select_dtypes(include='number').columns.tolist()
                    if len(numeric_cols) < 3:
                        st.error('Para PCA e agrupamento, são necessárias pelo menos 3 variáveis numéricas.')
                        st.stop()
                    with colPCA2:
                        n_components = int(st.number_input("Total de PCA", value=3, help="Escolha o total de PCA para ser usada no agrupamento.", min_value=2, max_value=max(2, len(numeric_cols))))
                    with colcluster3:
                        totalCLUSTER = int(st.number_input("Total de cluster", value=4, help="Escolha o total de clusters.", min_value=2, max_value=5))
                    with colVarPCA4:
                        valorPCA = int(st.number_input("Carregamento-PCA", value=1, help="Escolha o componente para exibir os carregamentos.", min_value=1, max_value=n_components))

                    # Área de visualização: uma figura de clusters e um gráfico de carregamentos
                    colGraficoCluster, colGraficoVarImportancia = st.columns([1, 1])

                    analiseTalhaoPCA = analiseTalhao.copy()

                    # Selecionar apenas variáveis numéricas e tratar ausentes
                    subset = analiseTalhaoPCA[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    subset = subset.dropna(axis=0, how='any')
                    if subset.shape[0] < 2:
                        st.error('Dados insuficientes após remover valores ausentes. Forneça ao menos 2 linhas completas.')
                        st.stop()
                    if totalCLUSTER > subset.shape[0]:
                        st.error(f"Número de clusters ({totalCLUSTER}) maior que número de amostras ({subset.shape[0]}). Reduza os clusters.")
                        st.stop()
                    scaler = StandardScaler()
                    scaled_df = scaler.fit_transform(subset)

                    # Aplicar PCA
                    pca = PCA(n_components=min(n_components, scaled_df.shape[1]))
                    pca_result = pca.fit_transform(scaled_df)

                    # Adicionar resultados da PCA ao DataFrame de maneira iterativa
                    for i in range(n_components):
                        analiseTalhaoPCA[f'PCA{i+1}'] = pca_result[:, i]

                    # Criar uma lista com os nomes dos componentes
                    PCA_features = [f'PCA{i+1}' for i in range(min(n_components, scaled_df.shape[1]))]

                    # Realizar o agrupamento k-means (vamos supor 2 clusters para este exemplo)
                    kmeans = KMeans(n_clusters=totalCLUSTER, random_state=0).fit(analiseTalhaoPCA.loc[subset.index, PCA_features])

                    analiseTalhaoPCA['Cluster'] = kmeans.labels_
                
                import numpy as np
                from numpy.linalg import eig
                import plotly.graph_objects as go

                # Obter a variância explicada por cada componente principal
                explained_variance = pca.explained_variance_ratio_

                figPCA = go.Figure()

                # Cores para os diferentes clusters
                colors = ['red', 'green', 'blue', 'purple', 'orange']  # Customize de acordo com o número de clusters

                for i, cluster in enumerate(analiseTalhaoPCA['Cluster'].unique()):
                    cluster_data = analiseTalhaoPCA[analiseTalhaoPCA['Cluster'] == cluster]

                    # Plotar os pontos de cada cluster
                    figPCA.add_trace(go.Scatter(
                        x=cluster_data['PCA1'],
                        y=cluster_data['PCA2'],
                        mode='markers+text',
                        text=cluster_data[fatorAgrupamento],
                        marker=dict(
                            size=8,
                            color=colors[i],
                        ),
                        name=f'Cluster {cluster}',
                        textposition="bottom center"
                    ))

                    # Calcular a média e a matriz de covariância para os pontos no cluster
                    mean = cluster_data[['PCA1', 'PCA2']].mean().values
                    cov = np.cov(cluster_data[['PCA1', 'PCA2']].values.T)

                    # Calcular os autovalores e autovetores da matriz de covariância
                    eig_vals, eig_vecs = eig(cov)

                    # Adicionar a elipse para o cluster
                    figPCA.add_shape(
                        type='circle',
                        xref='x', yref='y',
                        x0=mean[0] - 2*np.sqrt(eig_vals[0]),
                        y0=mean[1] - 2*np.sqrt(eig_vals[1]),
                        x1=mean[0] + 2*np.sqrt(eig_vals[0]),
                        y1=mean[1] + 2*np.sqrt(eig_vals[1]),
                        line_color=colors[i],
                        opacity=0.2,
                        fillcolor=colors[i],
                        line_width=2,
                    )

                # Layout e exibição única do gráfico de clusters
                figPCA.update_layout(
                    title='PCA e Agrupamento K-means dos Talhões',
                    xaxis_title='PCA1 - {0:.1f}%'.format(explained_variance[0]*100),
                    yaxis_title='PCA2 - {0:.1f}%'.format(explained_variance[1]*100),
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                with colGraficoCluster:
                    st.plotly_chart(figPCA, use_container_width=True)

                # Obtenha os coeficientes de carregamento para o primeiro componente principal
                loadings = pca.components_[valorPCA-1]

                # Crie um índice para cada variável
                variables = colunas.drop(fatorAgrupamento)

                # Crie um DataFrame com as variáveis e os carregamentos
                df_loadings = pd.DataFrame({'Variable': variables, 'Loading': loadings})

                # Ordene o DataFrame pelos valores de carregamento
                df_loadings = df_loadings.sort_values(by='Loading')

                # Crie o gráfico de barras usando o DataFrame ordenado
                figImportancia = px.bar(df_loadings, x='Variable', y='Loading', title=f'Contribuição das variáveis para o PCA{valorPCA}', labels={'Variable': 'Variáveis', 'Loading': 'Carregamento'})

                # Altere a orientação do texto do eixo x para vertical
                figImportancia.update_layout(xaxis_tickangle=-90, height=360, margin=dict(l=10, r=10, t=40, b=40))


                with colGraficoVarImportancia:
                    st.plotly_chart(figImportancia, use_container_width=True)

                enh_qualFuste = st.expander("Determinação da Necessidade de Calagem para os talhões", expanded=False)

                # Função que calcula o valor de NC
                def calculate_NC(T, Va, PRNT, Ve, col_Incoporacao):   
                    
                    NC = (T * (Ve - Va) / PRNT) *  (col_Incoporacao / 20 )            
                    if NC < 0:
                        NC = 0
                    else:
                        pass
                    return NC
                

                def NC_areaTotal(NC, areaTotal): 
                    return NC * areaTotal
                
                with enh_qualFuste:  

                    col_texto, col_PRNT, col_Ve, col_precoCalcario, col_profundidadeIncoporacao, col_NG, col_precoGesso = st.columns(7)

                    with col_texto:
                        st.write('')
                        st.write('')                
                        st.write('NC = T(Ve – Va)/PRNT')
                    with col_PRNT:
                        col_PRNT = float(st.number_input('PRNT ',0.0, 100.0, (85.0)))
                        
                    with col_Ve:
                        col_Ve = int(st.number_input('Ve ',0.0, 100.0, (60.0)))              
                    with col_precoCalcario:
                        precoCalcario = int(st.number_input('Tonelada de calcário (R$)',0.0, 1000.0, (160.0)))              
                    with col_profundidadeIncoporacao:
                        col_Incoporacao = float(st.number_input('Profundidade (cm)',0.0, 60.0, (20.0)))    
                    with col_NG:
                        col_NG = float(st.number_input('Gessagem',0.0, 1.25, (0.25)))
                    with col_precoGesso:
                        precoGesso = int(st.number_input('Tonelada de gesso (R$)',0.0, 1000.0, (160.0))) 



                    # Calculando o valor de NC utilizando a função calculate_NC
                    analiseTalhao_NC = analiseTalhao.copy()
                    analiseTalhao_NC['areaTotal'] = areaTalhao['area']            
                    analiseTalhao_NC['NC'] = analiseTalhao_NC.apply(lambda row: calculate_NC(row['T'], row['Va'], col_PRNT, col_Ve, col_Incoporacao), axis=1)
                    analiseTalhao_NC['NC_HA'] = analiseTalhao_NC.apply(lambda row: NC_areaTotal(row['NC'], row['areaTotal']), axis=1)          
                    
                    
                    # Exibindo o resultado
                    #Criando colunas para colocar os gráficos
                    col_g1, col_g2 = st.columns(2)
                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='NC', ascending=False)
                    figNC = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['NC'], name='NC')
                            ])

                    # Atualizar os rótulos dos eixos
                    figNC.update_layout(
                        title = 'Calagem',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de calagem (t/ha)'
                    )

                    figNG = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['NC'] * col_NG, name='NC'),
                            ])

                    # Atualizar os rótulos dos eixos
                    figNG.update_layout(
                        title = 'Gessagem',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de Gessagem (t/ha)'
                    )

                    with col_g1:
                        st.plotly_chart(figNC)
                        st.plotly_chart(figNG)

                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='NC_HA', ascending=False)
                    figNC_areaTotal = go.Figure(data=[go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['NC_HA'])])
                    figNG_areaTotal = go.Figure(data=[go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['NC_HA'] * col_NG)])

                    # Atualizar os rótulos dos eixos
                    figNC_areaTotal.update_layout(
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de calagem total (t)'
                    )

                    figNG_areaTotal.update_layout(
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de gessagem total (t)'
                    )

                    with col_g2:
                        st.plotly_chart(figNC_areaTotal)
                        st.plotly_chart(figNG_areaTotal)

                    st.markdown(
                        '<hr style="border-top: 0.5px solid "#1D250E";">',
                        unsafe_allow_html=True
                    )

                    m1, m2, m3, m4, m5 = st.columns((2,1,1,1,1))
                    m1.write('')
                    m2.metric(label ='Total de toneladas de Calcário',value = round(analiseTalhao_NC['NC_HA'].sum(),2))
                    m2.metric(label ='Total de toneladas de Gesso',value = round(analiseTalhao_NC['NC_HA'].sum() * col_NG,2))
                    m3.metric(label ='Custo total (R$)',value = round(analiseTalhao_NC['NC_HA'].sum() * precoCalcario,2))
                    m3.metric(label ='Custo total (R$) ',value = round(analiseTalhao_NC['NC_HA'].sum() * col_NG * precoGesso,2))
                    m1.write('')    

                
                enh_Fosfatagem = st.expander("Determinação da Necessidade fosfatagem para os talhões", expanded=False)  


                def Qntd_AduboFosfatado(P2O5_kgha, col_Porc_Perda, Preencher): 
                    if col_Porc_Perda <= 0 or Preencher <= 0:
                        return 0
                    Qntd_AduboFosfatado_kgha =  P2O5_kgha * 100 / col_Porc_Perda
                    Qntd_AduboFosfatado_kgha  = 100 * Qntd_AduboFosfatado_kgha / Preencher
                    if Qntd_AduboFosfatado_kgha < 0:
                        Qntd_AduboFosfatado_kgha = 0
                    return Qntd_AduboFosfatado_kgha

                def P2O5_kgHa(col_fosfatagemIdeal, P_meh1): 
                    # Se o valor medido já for >= ideal, não há necessidade (trunca em zero)
                    P2O5_kgha = (col_fosfatagemIdeal - P_meh1) * 4.58  # 2x para P e 2.29x para P2O5
                    if P2O5_kgha < 0:
                        P2O5_kgha = 0
                    return P2O5_kgha
                
                with enh_Fosfatagem:
                    col1_fosfatagemIdeal, col2_Porc_Perda, col3_Preencher= st.columns(3)
                    with col1_fosfatagemIdeal:
                        col_fosfatagemIdeal = float(st.number_input('Nível ideal de fósforo',0.0, 50.0, (20.0)))

                    with col2_Porc_Perda:
                        col_Porc_Perda = float(st.number_input('Perda considerada (%)',0.0, 100.0, (20.0)))
                                    
                    with col3_Preencher:
                        Preencher = float(st.number_input('Recomendação no adubo a preencher (%)', 0.0, 100.0, (12.0)))

                    analiseTalhao_NC['P2O5 (kg/ha)'] = analiseTalhao_NC.apply(lambda row: P2O5_kgHa(col_fosfatagemIdeal, row['P meh-¹']), axis=1)
                    analiseTalhao_NC['P2O5 total'] = analiseTalhao_NC.apply(lambda row: NC_areaTotal(row['P2O5 (kg/ha)'], row['areaTotal']), axis=1)
                    analiseTalhao_NC['Qntd_AduboFosfatado (kg/ha)'] = analiseTalhao_NC.apply(lambda row: Qntd_AduboFosfatado(row['P2O5 (kg/ha)'], col_Porc_Perda, Preencher), axis=1)
                    analiseTalhao_NC['Qntd_AduboFosfatado_Total'] = analiseTalhao_NC.apply(lambda row: NC_areaTotal(row['Qntd_AduboFosfatado (kg/ha)'], row['areaTotal']), axis=1)            
                    

                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='P2O5 (kg/ha)', ascending=False)
                    figP2O5 = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['P2O5 (kg/ha)'], name='P2O5 (kg/ha)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figP2O5.update_layout(
                        title = 'P2O5',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção (kg/ha)'
                    )

                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='P2O5 total', ascending=False)
                    figP2O5_total = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['P2O5 total'], name='P2O5 total (kg)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figP2O5_total.update_layout(
                        title = 'P2O5',
                        xaxis_title='Talhão',
                        yaxis_title='Correção em área total (kg)'
                    )

                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='Qntd_AduboFosfatado (kg/ha)', ascending=False)
                    figAdubo_Fosfatado = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['Qntd_AduboFosfatado (kg/ha)'], name='Adubo Fosfatado (kg/ha)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figAdubo_Fosfatado.update_layout(
                        title = 'Adubo Fosfatado',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção (kg/ha)'
                    )


                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='Qntd_AduboFosfatado_Total', ascending=False)
                    figAdubo_Fosfatado_total = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['Qntd_AduboFosfatado_Total'], name='Adubo Fosfatado área total (kg)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figAdubo_Fosfatado_total.update_layout(
                        title = 'Adubo Fosfatado',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção área total'
                    )


                    #Criando colunas para colocar os gráficos
                    col_g1, col_g2 = st.columns(2)

                    with col_g1:
                        st.plotly_chart(figP2O5)           
                        st.plotly_chart(figAdubo_Fosfatado)
                    with col_g2:
                        st.plotly_chart(figP2O5_total)
                        st.plotly_chart(figAdubo_Fosfatado_total)

                enh_Potassagem = st.expander("Determinação da Necessidade potassagem para os talhões", expanded=False) 

                def Nc_RecomendadaK2O(k20_kg_ha, col_Potassagem_Preencher):
                    if col_Potassagem_Preencher <= 0:
                        return 0
                    valor = k20_kg_ha * 100 / col_Potassagem_Preencher
                    if valor < 0:
                        valor = 0
                    return valor

                def Nc_potassio(K, T):
                    K_CTC_ph7 = 100 * K / T
                    K_porc = 3 - K_CTC_ph7
                    k_cmol_dm3 = K_porc * T / 100
                    k_mg_dm3 = k_cmol_dm3 * 390
                    k_kg_ha = k_mg_dm3 * 2
                    k20_kg_ha = k_kg_ha * 1.205
                    if k20_kg_ha < 0:
                        k20_kg_ha = 0
                    return k20_kg_ha
                

                
                with enh_Potassagem:

                    col1_Potassagem, col2_Potassagem_Preencher = st.columns(2)
                    with col1_Potassagem:
                        col_Potassagem = float(st.number_input('Preço (R$/KG) ',0.0, 50.0, (20.0)))

                    with col2_Potassagem_Preencher:
                        col_Potassagem_Preencher = float(st.number_input('Recomendação no adubo a preencher (%) ',0.0, 100.0, (2.80)))

                    analiseTalhao_NC['K2O (kg/ha)'] = analiseTalhao_NC.apply(lambda row: Nc_potassio(row['K'], row['T']), axis=1)
                    analiseTalhao_NC['Recomendação de K2O (kg/ha)'] = analiseTalhao_NC.apply(lambda row: Nc_RecomendadaK2O(row['K2O (kg/ha)'], col_Potassagem_Preencher), axis=1)
                    analiseTalhao_NC['K2O total'] = analiseTalhao_NC.apply(lambda row: NC_areaTotal(row['K2O (kg/ha)'], row['areaTotal']), axis=1)
                    analiseTalhao_NC['Recomendação area total (kg)'] = analiseTalhao_NC.apply(lambda row: NC_areaTotal(row['Recomendação de K2O (kg/ha)'], row['areaTotal']), axis=1)
                    
                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='K2O (kg/ha)', ascending=False)
                    figK2O = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['K2O (kg/ha)'], name='K2O (kg/ha)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figK2O.update_layout(
                        title = 'K2O',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção (kg/ha)'
                    )

                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='K2O total', ascending=False)
                    figK2O_total = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['K2O total'], name='K2O total (kg)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figK2O_total.update_layout(
                        title = 'K2O',
                        xaxis_title='Talhão',
                        yaxis_title='Correção em área total (kg)'
                    )

                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='Recomendação de K2O (kg/ha)', ascending=False)
                    figAdubo_Potassio = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['Recomendação de K2O (kg/ha)'], name='Recomendação de K2O (kg/ha)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figAdubo_Potassio.update_layout(
                        title = 'Recomendação de K2O (kg/ha)',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção (kg/ha)'
                    )


                    # Criar o gráfico de barras
                    analiseTalhao_NC = analiseTalhao_NC.sort_values(by='Recomendação area total (kg)', ascending=False)
                    figAdubo_Potassio_total = go.Figure(data=[
                            go.Bar(x=analiseTalhao_NC['Talhao'], y=analiseTalhao_NC['Recomendação area total (kg)'], name='Adubo k2o área total (kg)')
                            ])

                    # Atualizar os rótulos dos eixos
                    figAdubo_Potassio_total.update_layout(
                        title = 'Adubo K20',
                        xaxis_title='Talhão',
                        yaxis_title='Necessidade de correção área total'
                    )


                    #Criando colunas para colocar os gráficos
                    col_g1, col_g2 = st.columns(2)

                    with col_g1:
                        st.plotly_chart(figK2O)           
                        st.plotly_chart(figAdubo_Potassio)
                    with col_g2:
                        st.plotly_chart(figK2O_total)
                        st.plotly_chart(figAdubo_Potassio_total)
                    
                enh_Calculos = st.expander("Exibir tabela com os valores de adubação por talhão", expanded=False)
                with enh_Calculos:
                    st.write(analiseTalhao_NC)    

        else:
            st.warning("Por favor, faça o upload de um arquivo Excel.")
            with st.expander("Instruções sobre as unidades dos elementos químicos e exemplo de planilha para importação."):
                st.write("""
                **Atenção:** Para garantir que a análise de solo seja realizada corretamente, as unidades dos elementos devem seguir o padrão abaixo:

                | Elemento | Unidade         |
                |----------|-----------------|
                | PH       | pH (-)          |
                | P meh-¹  | P meh-¹ (mg/dm³)|
                | S        | S (mg/dm³)      |
                | K        | K (cmolc/dm³)   |
                | Ca       | Ca (cmolc/dm³)  |
                | Mg       | Mg (cmolc/dm³)  |
                | MO       | MO (dag/kg ou %)|
                | B        | B (mg/dm³)      |
                | Cu       | Cu (mg/dm³)     |
                | Fe       | Fe (mg/dm³)     |
                | Zn       | Zn (mg/dm³)     |
                | Mn       | Mn (mg/dm³)     |
                | T        | T (cmolc/dm³)   |
                | Va       | V (%)           |
                | mt       | m (%)           |

                **Observação:** Certifique-se de que os dados na planilha sigam essas unidades para que a análise seja precisa.
                """)
                # Função para carregar a planilha
                @st.cache_data
                def carregar_planilha_solos():
                    # Caminho da planilha (mesmo caminho que você forneceu)
                    caminho_planilha = r"dados/solos.xlsx"
                    
                    # Ler a planilha em modo binário
                    try:
                        with open(caminho_planilha, "rb") as file:
                            planilha_bytes = file.read()
                        return planilha_bytes
                    except FileNotFoundError:
                        st.error("Arquivo não encontrado. Verifique o caminho da planilha.")
                        return None

                # Carregar a planilha
                planilha_bytes = carregar_planilha_solos()

                # Se a planilha foi carregada com sucesso, disponibilizar para download
                if planilha_bytes:
                    st.download_button(
                        label="Baixar Planilha Modelo (solos.xlsx)",
                        data=planilha_bytes,
                        file_name="solos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("Planilha disponível para download. Utilize o modelo para preencher os dados de análise de solo.")

#fim do teste de importação
            

    elif Uso == 'Formigueiros':

        import streamlit as st
        import pandas as pd
        import requests
        from io import StringIO
        from streamlit_folium import folium_static
        import folium
        import folium.plugins as plugins  # Importando os plugins

        # Obter os dados da planilha do Google Sheets
        URL = "https://docs.google.com/spreadsheets/d/1RMc28dzaQBwCBCHjHGDQhmvzqp0SAiS2fIameQXC89A/gviz/tq?tqx=out:csv&sheet=Formiga_Cad"
        response = requests.get(URL)
        data = response.content.decode('utf-8')
        df_formiga = pd.read_csv(StringIO(data))

        # Criar um mapa base
        m = folium.Map(location=[-13.331891, -39.311357], zoom_start=16)

        df_sucupira = gpd.read_file("geojson.json")

        # Converte o GeoDataFrame para formato suportado pelo Folium
        df_sucupira_data = df_sucupira.to_crs(epsg="4326").to_json()


        # Função de estilo com alfa 0
        def style_function(feature):
            return {
                'fillOpacity': 0,  # Alfa de 0
                'weight': 3,
                'color': 'blue'
            }
        # Adiciona a camada GeoJSON ao mapa
        folium.GeoJson(df_sucupira_data, style_function=style_function).add_to(m)

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

        # Dicionário de cores para cada tipo de formiga
        color_dict = {
            'Saúvas': 'blue',
            'Quenquéns': 'red',
        }

        # Adicionar checkbox no Streamlit para mapa de calor e marcadores
        show_heatmap = st.checkbox('Mostrar Mapa de Calor')
        show_markers = st.checkbox('Mostrar Marcadores')

        # Adicionar marcadores para cada ponto
        heat_data = []  # Dados para o mapa de calor
        for index, row in df_formiga.iterrows():
            lat, long = row['LatLong'].split(',')
            lat, long = float(lat), float(long)
            color = color_dict.get(row['TipoFormiga'], 'green')  # 'green' é a cor padrão

            if show_markers:
                folium.Marker(
                    location=[lat, long],
                    icon=folium.Icon(color=color),
                    popup=row['TipoFormiga']
                ).add_to(m)

            heat_data.append([lat, long])  # Adicionando coordenadas à lista de dados para o mapa de calor

        # Se o checkbox do mapa de calor estiver marcado, adicione o mapa de calor
        if show_heatmap:
            plugins.HeatMap(heat_data).add_to(m)

        # Mostrar o mapa no Streamlit
        folium_static(m, height=900, width=900)
        


    elif Uso == 'Imagens':

        st.write("Em desenvolvimento")

        # # Fiz esse comentário para trabalhar os dados depois
        # import streamlit as st
        # import requests
        # from PIL import Image
        # import pandas as pd
        # from io import StringIO
        # # Configurações iniciais para acessar o drive
        # import json
        # from google.oauth2.service_account import Credentials
        # from googleapiclient.discovery import build
        # from googleapiclient.http import MediaIoBaseDownload
        # import io
        # from PIL import Image
        # import streamlit as st
        # from io import BytesIO
        # import numpy as np
        # from PIL.ExifTags import TAGS, GPSTAGS  # Importação adicional
        # from streamlit_folium import folium_static

        # # Função para converter DMS para decimal
        # def dms_to_decimal(degrees, minutes, seconds, ref):
        #     decimal = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
        #     if ref in ['S', 'W']:
        #         decimal = -decimal
        #     return decimal

        # # Função para extrair informações de geolocalização (adicionada)
        # def get_geotagging(exif):
        #     if not exif:
        #         raise ValueError("Sem metadados EXIF")
        #     geotagging = {}
        #     for (idx, tag) in TAGS.items():
        #         if tag == 'GPSInfo':
        #             if idx not in exif:
        #                 raise ValueError("Sem dados EXIF de geolocalização existentes")
        #             for (key, val) in GPSTAGS.items():
        #                 if key in exif[idx]:
        #                     geotagging[val] = exif[idx][key]
        #     return geotagging

        # # Ler o arquivo JSON
        # with open("chaves/ee-jeferson-2-86ccf5737e11.json", "r") as f:
        #     credentials_json = json.load(f)

        # # Criar credenciais
        # credentials = Credentials.from_service_account_info(credentials_json, scopes=["https://www.googleapis.com/auth/drive"])


        # # Construir o serviço do Google Drive
        # drive_service = build('drive', 'v3', credentials=credentials)

        # # ID da pasta
        # folder_id = '1-w9k2eSpnv5v2S-uNSDO6vGqoAjC6dDB'

        # # Obter lista de arquivos na pasta
        # results = drive_service.files().list(
        #     q=f"'{folder_id}' in parents",
        #     fields="files(id, name)"
        # ).execute()
        # items = results.get('files', [])

        # # Inicializa o mapa
        # mapa_foto = folium.Map(location=[-20.7194, -41.4997], zoom_start=8)

        # # Dicionário com os mapas base personalizados
        # basemaps = {
        #     'Google Maps': folium.TileLayer(
        #         tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        #         attr='Google',
        #         name='Google Maps',
        #         overlay=True,
        #         control=True
        #     ),
        #     'Google Satellite Hybrid': folium.TileLayer(
        #         tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        #         attr='Google',
        #         name='Google Satellite',
        #         overlay=True,
        #         control=True
        #     )
        # }

        # # Adiciona os mapas base personalizados ao mapa
        # basemaps['Google Maps'].add_to(mapa_foto)
        # basemaps['Google Satellite Hybrid'].add_to(mapa_foto)

        # # Para cada arquivo na pasta
        # for item in items:
        #     # Informações do arquivo
        #     file_id = item['id']
        #     file_name = item['name']

        #     # Solicitação para baixar o arquivo
        #     request = drive_service.files().get_media(fileId=file_id)
        #     fh = io.BytesIO()
        #     downloader = MediaIoBaseDownload(fh, request)
        #     done = False
        #     while done is False:
        #         status, done = downloader.next_chunk()
            
        #     # Mover o ponteiro para o início do arquivo BytesIO
        #     fh.seek(0)

        #     # Tentar abrir a imagem com PIL
        #     try:
        #         img = Image.open(fh)
        #         exif_data = img._getexif()

        #         if exif_data is not None:
        #             geotagging = get_geotagging(exif_data)

        #         # Converter coordenadas para formato decimal
        #         latitude = dms_to_decimal(*geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
        #         longitude = dms_to_decimal(*geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])

        #         # Converter a imagem PIL para base64
        #         buffered = BytesIO()
        #         img.save(buffered, format='JPEG')
        #         img_str = base64.b64encode(buffered.getvalue()).decode()

        #         # Criar popup com a imagem
        #         html = f'<img src="data:image/jpeg;base64,{img_str}" width=500>'
        #         popup = folium.Popup(html)

        #         # Adicionar marcador ao mapa
        #         folium.Marker([latitude, longitude], popup=popup).add_to(mapa_foto)
                
        #     except Exception as e:
        #         st.write(f"Não foi possível abrir o arquivo {file_name}: {e}")

        # # Mostrar o mapa com os marcadores e imagens no popup
        # folium_static(mapa_foto, height=900, width=900)

    else:
        pass

    if Uso in ['Madeira', 'Frutíferas'] and Variavel in ['TPF', 'Volume (m³)']:
        with col3:
            if Uso == 'Frutíferas' and Variavel == 'TPF':
                
                # Exibindo o botão na sidebar

                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                

                # Configuração de estilo
                def main():
                    # Configuração de estilo
                    st.markdown(
                        """
                        <style>
                        .sidebar .button {
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 18px;
                            font-weight: bold;
                            text-align: center;
                            cursor: pointer;
                            color: #fff;
                            background-color: #ff3366;
                            border: none;
                            border-radius: 5px;
                            transition: background-color 0.3s;
                        }
                        .sidebar .button:hover {
                            background-color: #ff4488;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    # Exibindo o botão na sidebar
                    if st.sidebar.button(f"Gerar relatório {Uso}", key="my_button", help="Gerar relatório"):
                        progress_bar = st.sidebar.progress(0)
                        status_text = st.sidebar.empty()

                        # Salvando o DataFrame em um arquivo Excel temporário
                        temp_file_path = f"{Uso}_Receita.xlsx"

                        df_Producao.to_excel(temp_file_path, index=False)
                        progress_bar.empty()
                        status_text.markdown(get_download_link(temp_file_path, "Clique aqui para baixar o relatório"), unsafe_allow_html=True)

                def get_download_link(file_path, link_text):
                    with open(file_path, 'rb') as file:
                        contents = file.read()
                    st.sidebar.download_button(
                        label=link_text,
                        data=contents,
                        file_name=file_path,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                if __name__ == "__main__":
                    main()
            
            elif Uso == 'Madeira' and Variavel =='Volume (m³)': 
                # Exibindo o botão na sidebar

                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                

                # Configuração de estilo
                def main():
                    # Configuração de estilo
                    st.markdown(
                        """
                        <style>
                        .sidebar .button {
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 18px;
                            font-weight: bold;
                            text-align: center;
                            cursor: pointer;
                            color: #fff;
                            background-color: #ff3366;
                            border: none;
                            border-radius: 5px;
                            transition: background-color 0.3s;
                        }
                        .sidebar .button:hover {
                            background-color: #ff4488;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    # Exibindo o botão na sidebar
                    if st.sidebar.button(f"Gerar relatório {Uso}", key="my_button", help="Gerar relatório"):
                        progress_bar = st.sidebar.progress(0)
                        status_text = st.sidebar.empty()

                        # Salvando o DataFrame em um arquivo Excel temporário
                        temp_file_path = f"{Uso}_Receita.xlsx"

                        dadosReceitaMadeira.to_excel(temp_file_path, index=False)
                        progress_bar.empty()
                        status_text.markdown(get_download_link(temp_file_path, "Clique aqui para baixar o relatório"), unsafe_allow_html=True)

                def get_download_link(file_path, link_text):
                    with open(file_path, 'rb') as file:
                        contents = file.read()
                    st.sidebar.download_button(
                        label=link_text,
                        data=contents,
                        file_name=file_path,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )


            else:
                pass
    
elif page == 'Regenera':
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

    colSidebar1, colSidebar2, colSidebar3= st.columns(3)
    with colSidebar1:
        mostrar = st.sidebar.radio('Uso da espécie', ('Mapa', 'Análise de solos'), horizontal=False)

    if mostrar == 'Mapa':

        col1, col2, col3, col4 = st.columns([6, 3, 3, 1])
        

        with col1:
            #import regenera
            st.write("Número de indivíduos por talhão e espécie")

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
            #importar dados do excel de inventário
            @st.cache_data
            def carregarDadosRegenera():    
                df = pd.read_csv("dados/dadosRegenera.csv")  

                return df 
            df_especies = carregarDadosRegenera()

            # Agrupar por talhão
            grouped = df_especies.groupby("TALHAO")
            
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

        with col2:

            #Essa função é para ser modificada, fiz isso apenas para ser um referencial para os calculos espectrais e vigor das plantas
            def ParaCodarDepois():
                # Crie um selectbox com os nomes dos talhões
                df1 = gpd.read_file("Talhao_Regenera.geojson")
                selected_talhao = st.selectbox('Selecione um talhão', df1['TALHAO'].unique())

                # Filtrar o DataFrame com base no talhão selecionado
                df_selected = df1[df1['TALHAO'] == selected_talhao]

                import streamlit as st
                from PIL import Image

                #image = Image.open("imagens/cup.jpeg")
                
                #st.image(image, caption='Sunrise by the mountains')

                import cv2
                import numpy as np
                import matplotlib.pyplot as plt
                import streamlit as st

                # Carregar a imagem (certifique-se de ter importado a imagem conforme mencionado anteriormente)
                imagem = cv2.imread("imagens/cup.jpeg")

                # Converter a imagem para o formato RGB (se necessário)
                imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

                # Extrair as bandas vermelha, verde e azul da imagem
                red = imagem_rgb[:, :, 0].astype(np.float32)
                green = imagem_rgb[:, :, 1].astype(np.float32)
                blue = imagem_rgb[:, :, 2].astype(np.float32)

                # Calcular os índices de vegetação
                ndvi = (red - blue) / (red + blue)
                evi = 2.5 * ((red - blue) / (red + 6 * green - 7.5 * blue + 1))
                gndvi = (green - red) / (green + red)
                
                PRI = (blue - red) / (blue + red)
                sPRI = (PRI + 1) / 2
                co2flux = gndvi * sPRI
                co2fluxEVI = sPRI * (evi - gndvi)* 3.5
                #exg = 2 * green - red - blue
                #gli = green / blueS
                #ndre = (red - green) / (red + green)
                #grvi = (green - red) / (green + red)
                #rndvi = (red - green) / (red + green)
                #savi = ((red - blue) / (red + blue + 0.5)) * 1.5
                #tgi = -0.5 * green + 0.5 * blue + red

                # Aplicar a falsa cor
                cmap = plt.cm.jet  # Escolher mapa de cores (você pode escolher outro mapa de cores)
                ndvi_color = cmap(ndvi)  # Aplicar mapa de cores ao NDVI
                evi_color = cmap(evi)  # Aplicar mapa de cores ao EVI
                gndvi_color = cmap(gndvi)  # Aplicar mapa de cores ao GNDVI
                sPRI_color = cmap(sPRI)
                co2flux_color = cmap(co2flux)
                co2fluxEVI_color = cmap(co2fluxEVI)
                #exg_color = cmap(exg)  # Aplicar mapa de cores ao ExG
                #gli_color = cmap(gli)  # Aplicar mapa de cores ao GLI
                #ndre_color = cmap(ndre)  # Aplicar mapa de cores ao NDRE
                #grvi_color = cmap(grvi)  # Aplicar mapa de cores ao GRVI
                #rndvi_color = cmap(rndvi)  # Aplicar mapa de cores ao RNDVI
                #savi_color = cmap(savi)  # Aplicar mapa de cores ao SAVI
                #tgi_color = cmap(tgi)  # Aplicar mapa de cores ao TGI

                # Criar a figura com subplots
                fig, axs = plt.subplots(6, 1, figsize=(8, 18))  # Aumentar o número de subplots e ajustar o tamanho

                # Lista de índices de vegetação e títulos
                indices_vegetacao = [imagem_rgb, evi_color, gndvi_color, sPRI_color, co2flux_color, co2fluxEVI_color]  # Adicionar a imagem RGB à lista
                titulos = ['RGB', 'EVI', 'GNDVI', 'sPRI', 'co2flux', 'co2fluxEVI']  # Adicionar "RGB" aos títulos

                # Plotar as imagens de índices de vegetação em falsa cor nos subplots
                for i, ax in enumerate(axs.flat):
                    ax.imshow(indices_vegetacao[i])
                    ax.set_title(titulos[i])
                    ax.axis('off')

                # Ajustar o espaçamento entre os subplots
                plt.tight_layout()

                # Exibir a figura
                # plt.show() -> Comentado porque o Streamlit irá gerenciar a exibição

                # Usar o Streamlit para exibir as imagens
                st.pyplot(fig)

                import plotly.graph_objects as go

                # Dados de exemplo do EVI ao longo do tempo
                tempo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Valores de tempo
                evi = [0.2, 0.3, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 0.5, 0.3]  # Valores de EVI correspondentes

                # Criar figura e adicionar o gráfico do EVI
                figEVI = go.Figure()
                figEVI.add_trace(go.Scatter(x=tempo, y=evi, mode='lines+markers', name='EVI'))

                # Personalizar o layout do gráfico
                figEVI.update_layout(
                    title='Comportamento do EVI ao longo do tempo',
                    xaxis_title='Tempo',
                    yaxis_title='EVI',
                    showlegend=True,
                    hovermode='x'
                )

                st.plotly_chart(figEVI, use_container_width=True)


        with col3:
            #selected_especie = st.selectbox('Selecione uma especie', df_especies['especie.planta'].unique())
            pass


    elif mostrar == 'Análise de solos':

        st.write('Sistema para análise de solos')

        @st.cache_data
        def carregarDadosAnaliseSolo(dados):    
            df = pd.read_excel("dados/solos.xlsx", sheet_name=dados)  
            return df 

        analiseTalhao = carregarDadosAnaliseSolo("analise")
        analiseReferencia = carregarDadosAnaliseSolo("referencia")

        # Lista com os talhões
        talhoes = analiseTalhao['Talhao'].unique()
        especie = analiseReferencia['especie'].unique()

        col1, col2 = st.columns(2)

        with col1:
            talhoes_selecionados = st.multiselect('Selecione os talhões:', talhoes)

        with col2:
            especie_selecionada = st.selectbox('Selecione a espécie:', especie)

        colGrafico1, colTabela2 = st.columns(2)

        with  colGrafico1:
            import plotly.express as px
            import pandas as pd

            # Inicializar a figura
            fig = go.Figure()

            # Adicionar linha zero
            df_zero = analiseTalhao.iloc[:1, 1:].copy()
            df_zero.loc[:, :] = 0

            fig.add_trace(go.Scatterpolar(
                r=df_zero.iloc[0, :],
                theta=df_zero.columns,
                fill='toself',
                name='Ideal',
                line=dict(color='grey', width=1)
            ))

            # Adicionar linhas para cada talhão selecionado
            for talhao_selecionado in talhoes_selecionados:
                df_porcentagem = analiseTalhao.loc[analiseTalhao['Talhao'] == talhao_selecionado].copy()
                df_porcentagem.iloc[:, 1:] = ((df_porcentagem.iloc[:, 1:] * 100) / analiseReferencia.iloc[:, 1:].values) - 100

                fig.add_trace(go.Scatterpolar(
                    r=df_porcentagem.iloc[:, 1:].values[0],
                    theta=df_porcentagem.columns[1:],
                    fill='toself',
                    name=talhao_selecionado
                ))

            # Atualizar layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-100, 100]
                    )
                ),
                showlegend=True,
                height=650,  # Ajuste o valor de altura conforme necessário
                title={
                    'text': "<b>Fertigrama</b>",
                    'y':1,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                title_font=dict(
                    size=30,
                )
            )

            # Renderize a figura
            st.plotly_chart(fig)
        
        with colTabela2:
            st.write("")           
            st.write("Análise de solos")            
            st.write(analiseTalhao)

            st.write(f"Valor referência para o {especie_selecionada}")
            st.write(analiseReferencia)

        colFatorAgrupamento1, colPCA2, colcluster3, colVarPCA4= st.columns(4)

        with colFatorAgrupamento1:
            colunas = analiseTalhao.columns
            
            fatorAgrupamento = st.selectbox('Fator de agrupamento', colunas)
            # Checando se a coluna é categórica
            if analiseTalhao[fatorAgrupamento].dtype == 'object':
                pass
            else:
                st.error('Erro: A coluna selecionada não é categórica')
        with colPCA2:
            n_components = int(st.number_input("Total de PCA", value=3, help="Escolha o total de PCA para ser usada no agrupamento.",  min_value=2, max_value=len(colunas)-1))

        with colcluster3:
            totalCLUSTER = int(st.number_input("Total de cluster", value=4, help="Escolha o total de clusters.",  min_value=2, max_value=5))
        with colVarPCA4:
            valorPCA = int(st.number_input("Carregamento-PCA", value=1, help="Escolha o total de clusters.",  min_value=1, max_value=n_components))
        

        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        col1_, colGraficoCluster,  colGraficoVarImportancia, col2_ = st.columns([0.5,3,3,1])

        analiseTalhaoPCA = analiseTalhao.copy()

        # Selecionando todas as colunas exceto 'Talhao'
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(analiseTalhaoPCA.drop(fatorAgrupamento, axis=1))

        # Aplicar PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_df)

        # Adicionar resultados da PCA ao DataFrame de maneira iterativa
        for i in range(n_components):
            analiseTalhaoPCA[f'PCA{i+1}'] = pca_result[:,i]

        # Criar uma lista com os nomes dos componentes
        PCA_features = [f'PCA{i+1}' for i in range(n_components)]

        # Realizar o agrupamento k-means (vamos supor 2 clusters para este exemplo)
        kmeans = KMeans(n_clusters=totalCLUSTER, random_state=0).fit(analiseTalhaoPCA[PCA_features])

        analiseTalhaoPCA['Cluster'] = kmeans.labels_

        import numpy as np
        from numpy.linalg import eig
        import plotly.graph_objects as go

        # Obter a variância explicada por cada componente principal
        explained_variance = pca.explained_variance_ratio_

        figPCA = go.Figure()

        # Cores para os diferentes clusters
        colors = ['red', 'green', 'blue', 'purple', 'orange']  # Customize de acordo com o número de clusters

        for i, cluster in enumerate(analiseTalhaoPCA['Cluster'].unique()):
            cluster_data = analiseTalhaoPCA[analiseTalhaoPCA['Cluster'] == cluster]

            # Plotar os pontos de cada cluster
            figPCA.add_trace(go.Scatter(
                x=cluster_data['PCA1'],
                y=cluster_data['PCA2'],
                mode='markers+text',
                text=cluster_data[fatorAgrupamento],
                marker=dict(
                    size=8,
                    color=colors[i],
                ),
                name=f'Cluster {cluster}',
                textposition="bottom center"
            ))

            # Calcular a média e a matriz de covariância para os pontos no cluster
            mean = cluster_data[['PCA1', 'PCA2']].mean().values
            cov = np.cov(cluster_data[['PCA1', 'PCA2']].values.T)

            # Calcular os autovalores e autovetores da matriz de covariância
            eig_vals, eig_vecs = eig(cov)

            # Adicionar a elipse para o cluster
            figPCA.add_shape(
                type='circle',
                xref='x', yref='y',
                x0=mean[0] - 2*np.sqrt(eig_vals[0]),
                y0=mean[1] - 2*np.sqrt(eig_vals[1]),
                x1=mean[0] + 2*np.sqrt(eig_vals[0]),
                y1=mean[1] + 2*np.sqrt(eig_vals[1]),
                line_color=colors[i],
                opacity=0.2,  # Faz a elipse semi-transparente
                fillcolor=colors[i],
                line_width=2,
            )

        figPCA.update_layout(
            title='PCA e Agrupamento K-means dos Talhões',
            xaxis_title='PCA1 - {0:.1f}%'.format(explained_variance[0]*100),
            yaxis_title='PCA2 - {0:.1f}%'.format(explained_variance[1]*100)
        )

        with colGraficoCluster:
            st.plotly_chart(figPCA)


                # Obtenha os coeficientes de carregamento para o primeiro componente principal
        loadings = pca.components_[valorPCA-1]

        # Crie um índice para cada variável
        variables = colunas.drop(fatorAgrupamento)

        # Crie um DataFrame com as variáveis e os carregamentos
        df_loadings = pd.DataFrame({'Variable': variables, 'Loading': loadings})

        # Ordene o DataFrame pelos valores de carregamento
        df_loadings = df_loadings.sort_values(by='Loading')

        # Crie o gráfico de barras usando o DataFrame ordenado
        figImportancia = px.bar(df_loadings, x='Variable', y='Loading', title=f'Contribuição das variáveis para o PCA{valorPCA}', labels={'Variable': 'Variáveis', 'Loading': 'Carregamento'})

        # Altere a orientação do texto do eixo x para vertical
        figImportancia.update_layout(xaxis_tickangle=-90)


        with colGraficoVarImportancia:
            st.plotly_chart(figImportancia)

        

        

    else:
        pass        