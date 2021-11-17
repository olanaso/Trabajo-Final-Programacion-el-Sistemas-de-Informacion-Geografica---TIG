import streamlit as st
from multiapp import MultiApp
from apps import (home, gee, raulcardenas, luischuyo, yeisonfernadez, Kmeas,lineaTiempo,sckty)

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here


apps.add_app("Kmeas", Kmeas.app)
apps.add_app("sckty", sckty.app)





# The main app
apps.run()
