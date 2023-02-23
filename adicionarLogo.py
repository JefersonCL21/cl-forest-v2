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




