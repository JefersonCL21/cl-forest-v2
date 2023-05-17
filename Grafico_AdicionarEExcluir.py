            df["Volume"] = FF1 * df["HT_Est"] *df.loc[(df['Especie'].isin(especie_Crescimento_DAP_HT)) & (~df['DAP'].isnull())].groupby(['Talhao'])['DAP'].transform(lambda x: x **2 * np.pi / 40000)

        #df["Volume"] = FF1 * df["HT_Est"] *df.loc[(df['Especie'].isin(especie_Crescimento_DAP_HT)) & (~df['DAP'].isnull())].groupby(['Talhao'])['DAP'].transform(lambda x: x **2 * np.pi / 40000)
        
        dfCrescimento_DAP_HT1 = dfCrescimento_DAP_HT.loc[dfCrescimento_DAP_HT['Especie'].isin(especie_Crescimento_DAP_HT)]
        especiePovoamento =df.loc[df['Especie'].isin(especie_Crescimento_DAP_HT)]
        idadePovoamento = especiePovoamento['idade'].unique()
        idadePovoamento = (st.multiselect(
            'Escolha uma idade', idadePovoamento, []
            ))

        volumePovoamento = df.loc[(df['Especie'].isin(especie_Crescimento_DAP_HT)) & (df['idade'].isin(idadePovoamento))]
        
        # Criar subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Adicionar boxplot
        fig.add_trace(go.Box(y=volumePovoamento['Volume'], x=volumePovoamento['idade'], name="Volume", marker_color="blue"), secondary_y=False)

        # Adicionar linha do DAP
        fig.add_trace(go.Scatter(x=dfCrescimento_DAP_HT1['idade'], y=dfCrescimento_DAP_HT1['Volume'], mode="lines", name="Volume"), secondary_y=True)
        fig.add_trace(go.Box(y=volumePovoamento['Volume'], x=volumePovoamento['idade'], name="Volume", marker_color="blue"), secondary_y=False)

        # Atualizar eixos
        fig.update_xaxes(title_text="Idade (anos)")
        fig.update_yaxes(title_text="Volume (m³)", secondary_y=False)
        fig.update_yaxes(title_text="Volume (m³)", secondary_y=True)

        # Atualizar layout
        fig.update_layout(title="Gráfico Boxplot HT_Est e Linha DAP", showlegend=True)

        # Mostrar gráfico

        st.plotly_chart(fig, use_container_width=True)