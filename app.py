import streamlit as st
from multiapp import MultiApp
from apps import (home, gee, raulcardenas, luischuyo, yeisonfernadez, Kmeas)

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here

apps.add_app("Inicio", home.app)
apps.add_app("Trabajo Final", gee.app)
apps.add_app("Yeison Fernandez", yeisonfernadez.app)
apps.add_app("Raul Cardenas", raulcardenas.app)
apps.add_app("Luis Chuyo", luischuyo.app)
apps.add_app("Scaty", Kmeas.app)



# The main app
apps.run()
