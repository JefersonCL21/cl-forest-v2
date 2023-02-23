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
import numpy as np
from scipy.optimize import curve_fit
# Criar um DataFrame de exemplo
import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CL Forest Biometrics",
    page_icon=":seedling:",
    layout="wide",
)

#adicionar a logo da empresa
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

especie = ['Khaya grandifoliola', 
             'Bowdichia virgilioides', 
             'Caesalpinia echinata', 
             'Cariniana legalis',
             'Dalbergia nigra',
             'Handroanthus chrysotrichus',
             'Handroanthus serratifolius',
             'Khaya ivorensis',
             'Plathymenia reticulata'
            ]

df = df[['Talhao', 'Especie', 'Popular', 'DAP', 'HT']]
df['HT'] = pd.to_numeric(df['HT'], errors='coerce')
df = df.loc[(df['Especie'].isin(especie))& (~df['HT'].isnull()) & (~df['DAP'].isnull()) ] 


modelos = st.sidebar.radio(' ', ('Modelos ajustados', 'Ajustar modelos'), horizontal=False)

selectbox = st.sidebar.selectbox("", ["Hipsométricos", "Volumétricos", "Biomassa"])

if modelos == 'Modelos ajustados' and selectbox == "Hipsométricos":

    enh_quacao = st.expander("Modelos ajustados para estimar altura total de árvores em agroflorestas", expanded=False)
    
    with enh_quacao:
        eq_col1, eq_col2, eq_col3, eq_col4 = st.columns(4)

        with eq_col1:
            st.markdown('<h5 style="color: gray;">Logístico</h3>', unsafe_allow_html=True)
            st.markdown("$HT = b_0/(1 + b_1*e^{-b_2*DAP})*\epsilon$")
        with eq_col2:
            st.markdown('<h5 style="color: gray;">Gompertz</h3>', unsafe_allow_html=True)
            st.markdown("$HT = b_0*e^{-e^{b_1-b_2*DAP}}*\epsilon$")
        with eq_col3:
            st.markdown('<h5 style="color: gray;">Curtis</h3>', unsafe_allow_html=True)
            st.markdown("$lnHT = b_0 + b_1*({1/DAP})+ \epsilon$")
        with eq_col4:
            st.markdown('<h5 style="color: gray;">Parabólico</h3>', unsafe_allow_html=True)
            st.markdown("$HT = b_0 + b_1*DAP + b_2*DAP^{2} + \epsilon$")

    def logistic_model(DAP, species):
        if species == "Khaya grandifoliola":
            b0 = 21.76844341
            b1 = 5.74124889
            b2 = 0.143963527
        elif species == "Bowdichia virgilioides":
            b0 = 21.79732276
            b1 = 3.776144935
            b2 = 0.105154042
        elif species == "Caesalpinia echinata":
            b0 = 8.990513346
            b1 = 2.162877862
            b2 = 0.232040133
        elif species == "Cariniana legalis":
            b0 = 10.57467764
            b1 = 3.091923738
            b2 = 0.180739727
        elif species == "Dalbergia nigra":
            b0 = 17.51162407
            b1 = 2.423401613
            b2 = 0.123758488
        elif species == "Handroanthus chrysotrichus":
            b0 = 8.009336547
            b1 = 7.02216711
            b2 = 0.438732334
        elif species == "Handroanthus serratifolius":
            b0 = 17.9521742
            b1 = 4.64531414
            b2 = 0.150851381
        elif species == "Khaya ivorensis":
            b0 = 27.874120800
            b1 = 5.231677313
            b2 = 0.076901282
        elif species == "Plathymenia reticulata":
            b0 = 18.3080410062121
            b1 = 3.44946237547807
            b2 = 0.101525454147303
        else:
            raise ValueError("Species not recognized")

        HT = b0 / (1 + b1 * np.exp(-b2 * DAP))
        return HT

    def gompertz_model(DAP, species):
        if species == "Khaya grandifoliola":
            b0 = 22.84499489
            b1 = 0.90572028
            b2 = 0.10073053
        elif species == "Bowdichia virgilioides":
            b0 = 22.57496185
            b1 = 0.596504514
            b2 = 0.07332119
        elif species == "Caesalpinia echinata":
            b0 = 9.331061448
            b1 = 0.27528858
            b2 = 0.174241468
        elif species == "Cariniana legalis":
            b0 = 11.10535017
            b1 = 0.506875092
            b2 = 0.129479685
        elif species == "Dalbergia nigra":
            b0 = 18.65171429
            b1 = 0.337405964
            b2 = 0.086086353
        elif species == "Handroanthus chrysotrichus":
            b0 = 8.512413427
            b1 = 1.048987267
            b2 = 0.301346331
        elif species == "Handroanthus serratifolius":
            b0 = 22.75351363
            b1 = 0.730806687
            b2 = 0.078943938
        elif species == "Khaya ivorensis":
            b0 = 34.43775768
            b1 = 0.789669916
            b2 = 0.041422041
        elif species == "Plathymenia reticulata":
            b0 = 19.804027
            b1 = 0.5597384
            b2 = 0.067651954
        else:
            raise ValueError("Species not recognized")

        HT = b0 * np.exp(-np.exp(b1 - b2 * DAP))
        return HT

    def parabolic_model(DAP, species):
        if species == "Khaya grandifoliola":
            b0 = -1.103468571
            b1 = 1.186903301
            b2 = -0.015690292
        elif species == "Bowdichia virgilioides":
            b0 = 2.698653641
            b1 = 0.751169154
            b2 = -0.007409894
        elif species == "Caesalpinia echinata":
            b0 = 2.270905087
            b1 = 0.728168192
            b2 = -0.021388202
        elif species == "Cariniana legalis":
            b0 = 1.662469507
            b1 = 0.66026658
            b2 = -0.012371877
        elif species == "Dalbergia nigra":
            b0 = 3.665754442
            b1 = 0.789337522
            b2 = -0.012375090
        elif species == "Handroanthus chrysotrichus":
            b0 = -1.975926648
            b1 = 1.673506602
            b2 = -0.074725336
        elif species == "Handroanthus serratifolius":
            b0 = 2.238091246
            b1 = 0.682264265
            b2 = -0.002044374
        elif species == "Khaya ivorensis":
            b0 = 2.368119239
            b1 = 0.571141995
            b2 = -0.001510177
        elif species == "Plathymenia reticulata":
            b0 = 2.483537519
            b1 = 0.629504792
            b2 = -0.006237206
        else:
            raise ValueError("Species not recognized")

        HT = b0 + b1 * DAP + b2 * (DAP ** 2)
        return HT

    def curtis_model(DAP, species):
        if species == "Khaya grandifoliola":
            b0 = 3.044098819
            b1 = -4.614315042
        elif species == "Bowdichia virgilioides":
            b0 = 3.013555196
            b1 = -7.295961829
        elif species == "Caesalpinia echinata":
            b0 = 2.260295062
            b1 = -2.719103577
        elif species == "Cariniana legalis":
            b0 = 2.476019547
            b1 = -4.989873432
        elif species == "Dalbergia nigra":
            b0 = 2.921989353
            b1 = -5.815057512
        elif species == "Handroanthus chrysotrichus":
            b0 = 2.444400613
            b1 = -4.673091866
        elif species == "Handroanthus serratifolius":
            b0 = 2.639323777
            b1 = -4.651464538
        elif species == "Khaya ivorensis":
            b0 = 3.228512313
            b1 = -12.56345354
        elif species == "Plathymenia reticulata":
            b0 = 3.0096409
            b1 = -9.301165045
        else:
            raise ValueError("Species not recognized")

        log_HT = np.exp(b0 + b1 * (1/DAP))
        return log_HT    


    # aplicando a função ao dataframe
    col_mod1, col_mod2 = st.columns(2)

    with col_mod1:
        especie1 = st.multiselect(
                    'Escolha uma espécie', especie, ['Dalbergia nigra']
                    )

    with col_mod2:
        mod = ['Logístico', 'Gompertz','Curtis','Parabólico']            
        modelos = st.multiselect(
                'Escolha um modelo', mod, ['Logístico']
                )

   
    for esp in especie1:
        df1 = df.loc[df['Especie'].isin([esp])] 
        modelo = []
        cor = []     
        rmse = []
        rmse_rel = []
        bias = []

        fig = px.scatter(df1, x='DAP', y='HT')
        fig.update_traces(marker_color="#1D250E")
        fig.update_layout(title=esp)
       
        if 'Logístico' in modelos:
            df1['HT_log'] = df1.apply(lambda x: logistic_model(x['DAP'], x['Especie']), axis=1)        
        
            fig.add_scatter(x=df1['DAP'], y=df1['HT_log'], mode='markers', name='Logístico', marker=dict(size=5, color="red"),) 
                              
            modelo.append('Logístico')
            cor.append(round(np.corrcoef(df1['HT'], y=df1['HT_log'])[0,1],3))
            rmse.append(round(np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_log'])) ** 2)),4))
            rmse_rel.append(round((np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_log'])) ** 2))) / np.mean(df1['HT'])*100, 2))
            bias.append(round(np.mean(np.array(df1['HT']) - np.array(df1['HT_log'])),2))

        if 'Gompertz' in modelos:
            df1['HT_Gom'] = df1.apply(lambda x: gompertz_model(x['DAP'], x['Especie']), axis=1)          
        
            fig.add_scatter(x=df1['DAP'], y=df1['HT_Gom'], mode='markers', name='Gompertz', marker=dict(size=5, color="Green"),)
            modelo.append('Gompertz')
            cor.append(round(np.corrcoef(df1['HT'], y=df1['HT_Gom'])[0,1],3))
            rmse.append(round(np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Gom'])) ** 2)),4))
            rmse_rel.append(round((np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Gom'])) ** 2))) / np.mean(df1['HT'])*100, 2))
            bias.append(round(np.mean(np.array(df1['HT']) - np.array(df1['HT_Gom'])),2))

        if 'Curtis' in modelos:
            df1['HT_Curt'] = df1.apply(lambda x: curtis_model(x['DAP'], x['Especie']), axis=1)          
        
            fig.add_scatter(x=df1['DAP'], y=df1['HT_Curt'], mode='markers', name='Curtis', marker=dict(size=4, color="blue"),)
            modelo.append('Curtis')
            cor.append(round(np.corrcoef(df1['HT'], y=df1['HT_Curt'])[0,1],3))
            rmse.append(round(np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Curt'])) ** 2)),4))
            rmse_rel.append(round((np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Curt'])) ** 2))) / np.mean(df1['HT'])*100, 2))
            bias.append(round(np.mean(np.array(df1['HT']) - np.array(df1['HT_Curt'])),2))
        if 'Parabólico' in modelos:
            df1['HT_Par'] = df1.apply(lambda x: parabolic_model(x['DAP'], x['Especie']), axis=1)          
        
            fig.add_scatter(x=df1['DAP'], y=df1['HT_Par'], mode='markers', name='Parabólico', marker=dict(size=6, color="#8B4513"),)
            modelo.append('Parabólico')
            cor.append(round(np.corrcoef(df1['HT'], y=df1['HT_Par'])[0,1],3))
            rmse.append(round(np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Par'])) ** 2)),4))
            rmse_rel.append(round((np.sqrt(np.mean((np.array(df1['HT']) - np.array(df1['HT_Par'])) ** 2))) / np.mean(df1['HT'])*100, 2))
            bias.append(round(np.mean(np.array(df1['HT']) - np.array(df1['HT_Par'])),2))        

        col_Res1, col_Res2 = st.columns(2)

        with col_Res1:
            st.plotly_chart(fig) 
        with col_Res2:
            resultados = pd.DataFrame({'Modelo': modelo,'r': cor, 'RMSE': rmse, 'RMSE_rel': rmse_rel, 'Bias': bias})

            fig2 = go.Figure(data=[go.Table(
                header=dict(values=list(resultados.columns),
                            fill_color="#1D250E",
                            font_color="white",
                            align='center'),
                cells=dict(values=[resultados.Modelo, resultados.r, resultados.RMSE, resultados.RMSE_rel, resultados.Bias],
                        fill_color='lavender',
                        align='center'))
            ])

            st.plotly_chart(fig2)


    data = {
    "Especie": ["Khaya grandifoliola", "Bowdichia virgilioides", "Caesalpinia echinata", "Cariniana legalis",
                "Dalbergia nigra", "Handroanthus chrysotrichus", "Handroanthus serratifolius", "Khaya ivorensis",
                "Plathymenia reticulata", "Khaya grandifoliola", "Bowdichia virgilioides", "Caesalpinia echinata",
                "Cariniana legalis", "Dalbergia nigra", "Handroanthus chrysotrichus", "Handroanthus serratifolius",
                "Khaya ivorensis", "Plathymenia reticulata", "Khaya grandifoliola", "Bowdichia virgilioides",
                "Caesalpinia echinata", "Cariniana legalis", "Dalbergia nigra", "Handroanthus chrysotrichus",
                "Handroanthus serratifolius", "Khaya ivorensis", "Plathymenia reticulata", "Khaya grandifoliola",
                "Bowdichia virgilioides", "Caesalpinia echinata", "Cariniana legalis", "Dalbergia nigra",
                "Handroanthus chrysotrichus", "Handroanthus serratifolius", "Khaya ivorensis",
                "Plathymenia reticulata"],
    "Modelo": ["Parabólico", "Parabólico", "Parabólico", "Parabólico", "Parabólico", "Parabólico", "Parabólico",
                "Parabólico", "Parabólico", "Logístico", "Logístico", "Logístico", "Logístico", "Logístico",
                "Logístico", "Logístico", "Logístico", "Logístico", "Gompertz", "Gompertz", "Gompertz", "Gompertz",
                "Gompertz", "Gompertz", "Gompertz", "Gompertz", "Gompertz", "Curtis", "Curtis", "Curtis",
                "Curtis", "Curtis", "Curtis", "Curtis", "Curtis", "Curtis"],
                'b0': [-1.103468571, 2.698653641, 2.270905087, 1.662469507, 3.665754442,
              -1.975926648, 2.238091246, 2.368119239, 2.483537519, 21.41571318,
              21.79732276, 8.990513346, 10.57467764, 17.51162407, 8.009336547,
              17.9521742, 27.8741208, 18.30804101, 22.18278669, 22.57496185,
              9.331061448, 11.10535017, 18.65171429, 8.512413427, 22.75351363,
              34.43775768, 19.80402714, 3.351592289, 3.013555196, 2.260295062,
              2.476019547, 2.921989353, 2.444400613, 2.639323777, 3.228512313,
              3.0096409],
                'b1': [1.186903301, 0.751169154, 0.728168192, 0.66026658, 0.789337522, 1.673506602, 0.682264265,
            0.571141995, 0.629504792, 6.871833269, 3.776144935, 2.162877862, 3.091923738, 2.423401613,
            7.02216711, 4.64531414, 5.231677313, 3.449462375, 1.088884968, 0.596504514, 0.27528858, 
            0.506875092, 0.337405964, 1.048987267, 0.730806687, 0.789669916, 0.559738448, -11.14719061,
            -7.295961829, -2.719103577, -4.989873432, -5.815057512, -4.673091866, -4.651464538, -12.56345354, 
            -9.301165045],
             'b2': [1.143588852, 0.891053782, 3.230319113, 1.881754923, 2.151041452, 5.262996542, 0.801898567, 1.058495846, 
             1.480786358, 0.157175976, 0.105154042, 0.232040133, 0.180739727, 0.123758488, 0.438732334, 0.150851381, 0.076901282, 
             0.101525454, 0.115312051, 0.07332119, 0.174241468, 0.129479685, 0.086086353, 0.301346331,  0.078943938, 
             0.041422041, 0.067651954, None, None, None, None, None, None, None, None, None]
                        }

    df_exportar = pd.DataFrame(data)
    

    enh_parametros = st.expander("Exportar os parametros dos modelos ajustados para estimar altura total de árvores em agroflorestas", expanded=False)
    with enh_parametros:
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        col_Parametros1, col_Estimados2 = st.columns(2)


        with col_Estimados2:

            if  len(especie1) > 0:
                df1
                csv1 = convert_df(df1)

                st.download_button(
                    label="Download - CSV",
                    data=csv1,
                    file_name='ParametrosModelosAjustados.csv',
                    mime='text/csv',
                )
            else:
                pass



        with col_Parametros1:

            st.write(df_exportar)
            csv = convert_df(df_exportar)
            st.download_button(
                label="Download - CSV",
                data=csv,
                file_name='ParametrosModelosAjustados.csv',
                mime='text/csv',
            )

    enh_aplicar_mod = st.expander("Aplicar os modelos em um novo banco de dados", expanded=False)  
    with enh_aplicar_mod:
        col_1, col_2 = st.columns(2)
        with col_1:
            
            file = st.file_uploader("Importar dados", type=["xlsx"])
            if file is not None:
                @st.cache(allow_output_mutation=True)
                def load_data():
                    df = pd.read_excel(file)
                    return df  
                df_novo1 = load_data()
                

   
        with col_2:            
            # Adicionar o selectbox ao aplicativo Streamlit
            if file is not None:
                esp = st.selectbox("Espécie:", especie)
                df_novo1["Especie"]  = esp
                 
                df_novo = df_novo1.copy()

        with col_2:
            if file is not None:
                mod = ['Logístico', 'Gompertz','Curtis','Parabólico']            
                modelos = st.multiselect(
                        'Modelo:', mod, ['Logístico']
                        )
        with col_2:
            if file is not None:
                
                col_DAP = st.selectbox("DAP (cm):", df_novo.columns)

                if not col_DAP:
                    st.error("Você não selecionou uma opção válida.")
                else:
                    if not df_novo[col_DAP].dtype in [np.int64, np.float64]:
                        st.error("Os dados selecionados não são numéricos.")
                    else:
                        df_novo["DAP_"] = df_novo[col_DAP]
                                    

        if file is not None:
            fig3= px.scatter()
           

            if 'Logístico' in modelos:
                df_novo['HT_log'] = df_novo.apply(lambda x: logistic_model(x['DAP_'], x['Especie']), axis=1)
                fig3.add_scatter(x=df_novo['DAP_'], y=df_novo['HT_log'], mode='markers', name='Logístico', marker=dict(size=5, color="red"),)        

            if 'Gompertz' in modelos:
                df_novo['HT_Gom'] = df_novo.apply(lambda x: gompertz_model(x['DAP_'], x['Especie']), axis=1)
                fig3.add_scatter(x=df_novo['DAP_'], y=df_novo['HT_Gom'], mode='markers', name='Gompertz', marker=dict(size=5, color="Green"),)          

            if 'Curtis' in modelos:
                df_novo['HT_Curt'] = df_novo.apply(lambda x: curtis_model(x['DAP_'], x['Especie']), axis=1)
                fig3.add_scatter(x=df_novo['DAP_'], y=df_novo['HT_Curt'], mode='markers', name='Curtis', marker=dict(size=4, color="blue"),)          

            if 'Parabólico' in modelos:
                df_novo['HT_Par'] = df_novo.apply(lambda x: parabolic_model(x['DAP_'], x['Especie']), axis=1) 
                fig3.add_scatter(x=df_novo['DAP_'], y=df_novo['HT_Par'], mode='markers', name='Parabólico', marker=dict(size=6, color="#8B4513"),)

                # Store the initial value of widgets in session state
       

        col1, col2 = st.columns(2)
        if file is not None:
            with col1:
                st.markdown(
                    '<hr style="border-top: 0.5px solid "#1D250E";">',
                    unsafe_allow_html=True
                )
                st.write('Baixar os dados')
                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv2 = convert_df(df_novo)
                st.download_button(
                    label="Download - CSV",
                    data=csv2,
                    file_name='Dados.csv',
                    mime='text/csv',
                )

        with col2:
           if file is not None:
                st.plotly_chart(fig3)            

                     
      
elif modelos == 'Ajustar modelos' and selectbox == "Hipsométricos":


    if selectbox == "Hipsométricos":

        col1, col2, col3 = st.columns([1,4,1])
        with col2:

            st.set_option('deprecation.showfileUploaderEncoding', False)

            def Logistico(x, b0, b1, b2):
                y = b0/(1 + b1*np.exp(-b2*x))
                return y

            def Gompertz(x, b0, b1, b2):
                y = b0*np.exp(-np.exp(b1-b2*x))
                return y

            def Richards(x, b0, b1, b2, b3):
                y = b0/(1 + np.exp(b1-b2*x))**(1/b3)
                return y

            st.title("Modelos alométricos para estimar altura")
            file = st.file_uploader("Upload database", type=["xlsx"])
            if file is not None:
                @st.cache(allow_output_mutation=True)
                def load_data():
                    df = pd.read_excel(file)
                    return df  
                df = load_data()
                independent_var = st.sidebar.selectbox('Independent variable',list(df.columns))
                dependent_var = st.sidebar.selectbox('Dependent variable',list(df.columns))
                x = df[independent_var]
                y = df[dependent_var]

            def compare_models():

                if file is not None:

                    func_options = ["Logístico", "Gompertz", "Richards"]
                    func_selected = st.multiselect("Select the function to use", func_options)
                    
                    with st.sidebar.expander("Logístico"): 
                        st.markdown("$y = b_0/(1 + b_1*e^{-b_2*x})$")                    
                        
                        log_1 = float(st.number_input('b0 =', value=20.0))
                        log_2 = float(st.number_input('b1 =', value=5.0))        
                        log_3 = float(st.number_input('b2 =', value=0.1))

                    with st.sidebar.expander("Gompertz"): 
                        st.markdown("$y = b_0*e^{-e^{b_1-b_2*x}}$")                    
                        gom_1 = float(st.number_input('b0 = ', value=20.0))
                        gom_2 = float(st.number_input('b1 = ', value=0.86))        
                        gom_3 = float(st.number_input('b2 = ', value=0.097))

                    with st.sidebar.expander("Richards"): 
                        st.markdown("$y = b_0/(1 + e^{b_1-b_2*x})^{1/b_3}$")                    
                        ric_1 = float(st.number_input('b0 =  ', value=22.0))
                        ric_2 = float(st.number_input('b1 =  ', value=1.86))        
                        ric_3 = float(st.number_input('b2 =  ', value=0.13))
                        ric_4 = float(st.number_input('b3 =  ', value=2))

                    p0_dict = {"Logístico": [log_1, log_2, log_3], "Gompertz": [gom_1, gom_2, gom_3], "Richards": [ric_1, ric_2, ric_3, ric_4]}


                
                    results = {}
                    for func_name in func_selected:
                        if func_name == "Logístico":
                            func = Logistico

                        
                        elif func_name == "Gompertz":
                            func = Gompertz
                            
                        else:
                            func = Richards



                            

                        p0 = p0_dict[func_name]

                        # Fit the model
                        try:
                            popt, pcov = curve_fit(func, x, y, p0=p0)
                        except:
                            st.error("O modelo não ajustou. Deseja escolher outros chutes para os parâmetros iniciais?")



                        #Calculate the R^2 and RMSE
                        y_pred = func(x, *popt)
                        r2 = round(np.corrcoef(y, y_pred)[0,1]**2, 4)
                        rmse = round(np.sqrt(np.mean((y_pred-y)**2)), 4)

                        #Calculate relative RMSE
                        mean_y = np.mean(y)
                        rrmse = rmse/mean_y*100
                        rrmse = round(rrmse, 4)

                        #Calculate bias
                        bias = round(np.mean(y_pred - y), 4)
                        

                        results[func_name] = {"R2": r2, "RMSE": rmse, "RMSE_rel": rrmse, "Bias": bias}
                        fig = px.scatter(x=x, y=y, labels={'x': independent_var, 'y': dependent_var})
                        fig.update_traces(marker_color="#1D250E")
                        fig.add_scatter(x=x, y=y_pred, mode='markers', name= func_name, marker=dict(size=3, color = 'red'),)                        
                        fig.update_layout(title=func_name)
                        st.plotly_chart(fig)


                        if func_name == "Logístico":
                            st.markdown("$y = b_0/(1 + b_1*e^{-b_2*x})$")
                            st.markdown(f"Parâmetros: b0 = {popt[0]:.4f}, b1 = {popt[1]:.4f}, b2 = {popt[2]:.4f}")
                        elif func_name == "Gompertz":
                            st.markdown("$y = b_0*e^{-e^{b_1-b_2*x}}$")
                            st.markdown(f"Parâmetros: b0 = {popt[0]:.4f}, b1 = {popt[1]:.4f}, b2 = {popt[2]:.4f}")
                        else:
                            st.markdown("$y = b_0/(1 + e^{b_1-b_2*x})^{1/b_3}$")
                            st.markdown(f"Parâmetros: b0 = {popt[0]:.4f}, b1 = {popt[1]:.4f}, b2 = {popt[2]:.4f}, b3 = {popt[3]:.4f}")                       
                    

                    

                    estatisticas = pd.DataFrame.from_dict(results, orient='index', columns=["R2", "RMSE", "RMSE_rel", "Bias"])

                    fig2 = go.Figure(data=[go.Table(
                        header=dict(values=list(estatisticas.columns),
                                    fill_color="#1D250E",
                                    font_color="white",
                                    align='center'),
                        cells=dict(values=[estatisticas.R2, estatisticas.RMSE, estatisticas.RMSE_rel, estatisticas.Bias],
                                fill_color='lavender',
                                align='center'))
                                            ])

                    st.plotly_chart(fig2) 

            
                        

        
            compare_models()


        

            
    elif selectbox == "Volumétricos":
        st.title("Modelo de Rede Neural para estimar altura")

    else:
        pass

else:
    pass

    