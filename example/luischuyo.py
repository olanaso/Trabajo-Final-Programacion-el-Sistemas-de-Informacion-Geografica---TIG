import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd
import numpy as np


def app():
    st.title("Trabajo de luis Chuyo")

    st.header("Calculo de ULE de la Zona de XXXX")


    df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns = ['lat', 'lon'])
    st.map(df)

