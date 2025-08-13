import streamlit as st
import base64

"""
Esse script adiciona logo. A logo foi importada para o
site imgur e depois foi feito a importação.
"""


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://i.imgur.com/7dH5wkT.jpeg);
                background-repeat: no-repeat;
                padding-top: 230px;
                background-size: 230px 230px;
                /* Centraliza horizontalmente; 10px a partir do topo */
                background-position: center 0px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )




