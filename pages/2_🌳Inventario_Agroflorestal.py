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
@st.cache(allow_output_mutation=True)
def carregarDados():    
    df = pd.read_csv("dados/GERAL_V2.csv")  

    return df 
df = carregarDados()

@st.cache(allow_output_mutation=True)
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
        Uso = st.sidebar.radio('Uso da espécie', ('Inventário resumido', 'Madeira', 'Frutíferas', 'Especiárias', 'IRP'), horizontal=False)

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
            @st.cache
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
        

        Sub_box_col1, Sub_hist_col2= st.columns(2)
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
                df = pd.DataFrame({'LI': LI, 'LS': LS, 'clas': clas})
                return df

            # Filtre os dados com base na espécie e remova as linhas com valor nulo em 'DAP'
            dadosHist = df.query('Especie in @especie and not DAP.isnull()')

            # Crie uma coluna que indica a classe para cada linha com base nos limites das classes
            dadosBar = classe(data=dadosHist['DAP'])
            dadosHist['classe'] = pd.cut(dadosHist['DAP'], bins=dadosBar['LI'].tolist()+[dadosBar['LS'].iloc[-1]], labels=dadosBar['clas'].tolist())

            # Agrupe as linhas por classe e conte o número de linhas em cada classe
            dadosBar2 = dadosHist.groupby('classe')['DAP'].agg(['count', 'mean']).reset_index()
            dadosBar2 = dadosBar2.rename(columns={'count': 'n', 'mean': 'ordem'}).sort_values('ordem')

            # Remova a coluna 'DAP' do dataframe de saída
            dadosBar2 = dadosBar2[['classe', 'n', 'ordem']]

            fig2 = px.bar(dadosBar2, x='classe', y = 'n')
            fig2.update_traces(marker_color="#1D250E")
            #fig2.update_layout(plot_bgcolor="#FFFAFA")
                                                                        
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

                    @st.cache
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

                    @st.cache(allow_output_mutation=True)
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
                @st.cache
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
                st.write(resultado_exibir_cacau)
                @st.cache
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
                @st.cache
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
                @st.cache
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

                @st.cache
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

        @st.cache
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
    
    import regenera

    regenera.teste()

    regenera.exibirMapa()

