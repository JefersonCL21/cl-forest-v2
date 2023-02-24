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
import importarDados
import gerarMapa
import streamlit as st
from dateutil.rrule import rrule, YEARLY
import pickle


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

polygon = importarDados.carregarDadosSHP()
map_df = polygon
map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

page = st.sidebar.selectbox('Escolha uma propriedade',['Sucupira Agroflorestas','Regenera'])

if page == 'Sucupira Agroflorestas':
    col1, col2= st.columns(2)
    with col1:
        Uso = st.sidebar.radio('Uso da espécie', ('Inventário resumido', 'Madeira', 'Frutíferas', 'Especiárias'), horizontal=False)

    with col2:
        if Uso == 'Inventário resumido':
            pass

        elif Uso == 'Madeira':
            Variavel = st.sidebar.radio('Variável analisada', ('Número de indivíduos', 'DAP médio (cm)', 'Altura (m)', 'Area basal (m²)', 'Volume (m³)'), horizontal=False)
        elif Uso == 'Frutíferas':
            Variavel = st.sidebar.radio('Variável analisada', ('TPF', 'Fenofase', 'Vigor'), horizontal=False)

        else:
            Variavel = st.sidebar.radio('Variável analisada', ('Fenofase', 'Vigor'), horizontal=False)

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
            st.write(res_dados[['Popular', 'Especie', 'n']].style.background_gradient(cmap="Greys"))
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
            dados2['prop'] = dados2.groupby(['Talhao'])['counts'].apply(lambda x: x / x.sum())
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
            dados3['prop'] = dados3.groupby(['Talhao'])['counts'].apply(lambda x: x / x.sum())
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
            plantios = plantios.groupby(['Talhao', 'Especie', 'DAP_MED']).mean()

        with Sub_Talhao_col2:
        
            columns = df1.loc[df['Especie'].isin(especie)]['Talhao'].unique().tolist()                
            column_name = st.multiselect('Escolha um Talhão', columns, columns[0])
            Talhao = df1.loc[df1['Talhao'].isin(column_name)]
        

        Sub_box_col1, Sub_hist_col2= st.columns(2)
        with Sub_box_col1:
                
                
            map_df['Quantidade'] = 0        
            for j in range(0, len(map_df.index)):
                for i in range(0, len(plantios.index)):
                    if (map_df.iloc[j, 1] == plantios.index[i][0]):
                        map_df.loc[j, 'Quantidade'] = plantios.index[i][2]
            
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
                plantios = plantios.groupby(['Talhao', 'Especie', 'HT_MED']).mean()

            with Sub_Talhao_col2:
            
                columns = df1['Talhao'].unique().tolist()                
                column_name = st.multiselect('Escolha um Talhão', columns, ['T1'])
                Talhao = df1.loc[df1['Talhao'].isin(column_name)]
            

            Sub_box_col1, Sub_hist_col2= st.columns(2)
            with Sub_box_col1:
                    
                    
                map_df['Quantidade'] = 1        
                for j in range(0, len(map_df.index)):
                    for i in range(0, len(plantios.index)):
                        if (map_df.iloc[j, 1] == plantios.index[i][0]):
                            map_df.loc[j, 'Quantidade'] = plantios.index[i][2]
                
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

        
        acai_tab, cacau_tab, cup_tab, juc_tab = st.tabs(["Açaí", "Cacau", "Cupuaçu", "Juçara"])

        

#Açaí
        with acai_tab:

            if acai_tab:
                enh_expander = st.expander("Fatores que influencia a produção", expanded=False) 
                with enh_expander:

                    Horizonte_col1, producaoKG_col2, preco_col3 = st.columns(3) 
                    with Horizonte_col1:   
                        seed = int(st.number_input("Horizonte de produção", value=2030, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with producaoKG_col2:
                        quilosPlanta = int(st.number_input("Produção média esperada por planta (kg)", value=20, help="Escolha um intevalo para a produção esperada.",  min_value=0))

                    with preco_col3:
                        Preco = int(st.number_input("Preço de venda (R$/kg)", value=20, help="Escolha um intevalo para o preço comercializado.",  min_value=0))

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
                        print(TPF)                        
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
                resultado_resultado_exibir= resultado_exibir.reset_index()
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
                st.write(resultado_resultado_exibir.style.background_gradient(cmap="Greys"))
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_resultado_exibir)

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
                        cacau_seed = int(st.number_input("Horizonte de produção ", value=2030, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with cacau_producaoKG_col2:
                        cacau_quilosPlanta = int(st.number_input("Produção média esperada por planta (kg) ", value=20, help="Escolha um intevalo para a produção esperada.",  min_value=0))

                    with cacau_preco_col3:
                        cacau_Preco = int(st.number_input("Preço de venda (R$/kg) ", value=20, help="Escolha um intevalo para o preço comercializado.",  min_value=0))

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
                resultado_resultado_exibir= resultado_exibir.reset_index()
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
                st.write(resultado_resultado_exibir.style.background_gradient(cmap="Greys"))
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_resultado_exibir)

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
                        cup_seed = int(st.number_input("Horizonte de produção  ", value=2030, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with cup_producaoKG_col2:
                        cup_quilosPlanta = int(st.number_input("Produção média esperada por planta (kg)  ", value=20, help="Escolha um intevalo para a produção esperada.",  min_value=0))

                    with cup_preco_col3:
                        cup_Preco = int(st.number_input("Preço de venda (R$/kg)  ", value=20, help="Escolha um intevalo para o preço comercializado.",  min_value=0))

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
                resultado_resultado_exibir= resultado_exibir.reset_index()
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
                st.write(resultado_resultado_exibir.style.background_gradient(cmap="Greys"))
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_resultado_exibir)

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
                        juc_seed = int(st.number_input("Horizonte de produção    ", value=2030, help="Escolha um intevalo para a produção esperada.",  min_value=2022))
                        
                    with juc_producaoKG_col2:
                        juc_quilosPlanta = int(st.number_input("Produção média/planta (kg)    ", value=10, help="Escolha um intevalo para a produção esperada.",  min_value=0))

                    with juc_preco_col3:
                        juc_Preco = int(st.number_input("Preço de venda (R$/kg)    ", value=20, help="Escolha um intevalo para o preço comercializado.",  min_value=0))

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
                resultado_resultado_exibir= resultado_exibir.reset_index()
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
                st.write(resultado_resultado_exibir.style.background_gradient(cmap="Greys"))
                #st.dataframe(resultado_resultado_exibir)
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(resultado_resultado_exibir)

                st.download_button(
                    label="Download - CSV",
                    data=csv,
                    file_name='Euterpe_Edulis.csv',
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

    else:
        pass
