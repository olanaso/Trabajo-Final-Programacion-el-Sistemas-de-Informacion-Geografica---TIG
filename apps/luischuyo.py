import streamlit as st
import geopandas as gpd
import leafmap.foliumap as leafmap
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
from shapely.geometry import Polygon


#Permite subir un geojson
@st.cache
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf


def app():
    st.title("Trabajo de luis Chuyo")

    st.header("Calculo de ULE de la Zona de XXXX")



    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        m = geemap.Map(basemap="HYBRID", plugin_Draw=True, draw_export=True)
        m.add_basemap("ROADMAP")

        with st.expander("See a video demo"):
            st.video("https://youtu.be/VVRK_-dEjR4")

        data = st.file_uploader(
            "Draw a small ROI on the map, click the Export button to save it, and then upload it here. Customize timelapse parameters and then click the Submit button 😇👇",
            type=["geojson"],
        )

        if data:
            gdf = uploaded_file_to_gdf(data)
            st.session_state["roi"] = geemap.geopandas_to_ee(gdf)
            m.add_gdf(gdf, "ROI")
        else:
            polygon_geom = Polygon(
                [
                    [-74.672699, -8.600032],
                    [-74.672699, -8.254983],
                    [-74.279938, -8.254983],
                    [-74.279938, -8.600032],
                ]
            )
            crs = {"init": "epsg:4326"}
            gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
            st.session_state["roi"] = geemap.geopandas_to_ee(gdf)
            m.add_gdf(gdf, "ROI", zoom_to_layer=True)
        m.to_streamlit(height=600)
