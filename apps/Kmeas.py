import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans #Para utilizar el método de KMeans
import pydeck as pdk
import matplotlib.pyplot as plt
import numpy as np
import random
import json

from matplotlib import cm
import pandas as pd
#import geopandas as gpd

from sklearn.preprocessing import StandardScaler #Utilizado para la estandarización.
from sklearn.metrics import silhouette_samples
import math as math
import scipy.stats as stats #Para cálculo de probabilidades estadísticas.
from sklearn.model_selection import train_test_split #Para la partición de los sets de training y testing.


#Definimos la funcion que generar el grafico
def Grafico_de_silueta(X,n_cluster_list,init,n_init,max_iter,tol,semilla):
    try:
        print("x", X)
        print("n_cluster_list", n_cluster_list)
        print("init", init)
        print("n_init", n_init)
        print("max_iter", max_iter)
        print("tol", tol)
        print("semilla", semilla)
        cont = 0
        # fig = plt.figure()
        for i in n_cluster_list:
            cont += 1
            plt.subplot(3, 4, cont)
            km = KMeans(n_clusters=i,
                        init=init,  # elija k observaciones (filas) para los centroides iniciales
                        n_init=n_init,  # número de veces que el algoritmo se ejecutará
                        max_iter=max_iter,  # número máximo de iteraciones para una ejecución
                        tol=tol,  # tolerancia para declarar convergencia
                        random_state=semilla)  # semilla
            print("Punto 01")
            y_km = km.fit_predict(X)
            cluster_labels = np.unique(y_km)  # valores de clúster
            n_clusters = cluster_labels.shape[0]  # núnero de clústers
            silhouette_vals = silhouette_samples(X, y_km,
                                                 metric='euclidean')  # valores de silueta teniendo en cuenta la distancia euclideana

            y_ax_lower, y_ax_upper = 0, 0
            yticks = []  # objeto tipo lista vacío
            fig = plt.figure()
            print("Punto 02")
            for i, c in enumerate(cluster_labels):
                print("Punto 03")
                c_silhouette_vals = silhouette_vals[
                    y_km == c]  # valores de silueta cuando y_km toma el valor c de los posibles n de clúster
                c_silhouette_vals.sort()  # se ordenan de menor a mayor los valores de silueta
                y_ax_upper += len(c_silhouette_vals)  # número de valores de silueta
                color = cm.jet(float(i) / km.n_clusters)  # definir el color
                plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                         edgecolor='none', color=color)  # visualización de los valores de silueta para k
                yticks.append((y_ax_lower + y_ax_upper) / 2.)
                y_ax_lower += len(c_silhouette_vals)
            silhouette_avg = np.mean(silhouette_vals)  # media de los valores de silueta
            print("Punto 04")
            plt.axvline(silhouette_avg, color="red",
                        linestyle="--")  # mostrar una línea con los valores medios de silueta
            print("Punto 051")
            plt.yticks(yticks, cluster_labels + 1)
            plt.ylabel('Cluster')
            plt.xlabel('Coeficiente de Silueta')
            plt.title("Silouette para k= " + str(km.n_clusters) + "\n" + "Coeficiente de Silueta= " + str(
                round((silhouette_avg), 2)))
            print("Punto 06")
            plt.tight_layout()

            # Mostramos en el streamlit
            print("Punto 07")
            st.pyplot(fig)
            print("Punto 05")

    except Exception as e:
        st.error(f"Error adding layer: {e}")


def distorcion(X,n_cluster_list,init,n_init,max_iter,tol,semilla):
    try:
        distorsiones = []  # objeto que almacena los valores de inercia o distorsiones
        for i in n_cluster_list:
            km = KMeans(n_clusters=i,
                        init=init,  # elija k observaciones (filas) para los centroides iniciales
                        n_init=n_init,  # número de veces que el algoritmo se ejecutará
                        max_iter=max_iter,  # número máximo de iteraciones para una ejecución
                        tol=tol,  # tolerancia para declarar convergencia
                        random_state=semilla)  # semilla
            km.fit(X)
            distorsiones.append(km.inertia_)
        fig2 = plt.figure()
        plt.rcParams['figure.figsize'] = (14, 6)
        plt.plot(n_cluster_list,distorsiones,marker='o')
        plt.title("Evaluación de la Inercia")
        plt.xlabel('K')
        plt.ylabel('Inercia')
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error adding layer: {e}")

#permite subir un geojson
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
        gdf = gpd.read_file(file_path,encoding="utF-8")

    return gdf

#
def geojson_layer(igeojson):

    """
    GeoJsonLayer
    ===========

    Property values in Vancouver, Canada, adapted from the deck.gl example pages. Input data is in a GeoJSON format.
    """

    DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"
    LAND_COVER = [
        [[-123.0, 49.196], [-123.0, 49.324], [-123.306, 49.324], [-123.306, 49.196]]
    ]

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=49.254, longitude=-123.13, zoom=11, max_zoom=16, pitch=45, bearing=0
    )

    polygon = pdk.Layer(
        "PolygonLayer",
        LAND_COVER,
        stroked=False,
        # processes the data as a flat longitude-latitude pair
        get_polygon="-",
        get_fill_color=[0, 0, 0, 20],
    )

    #print(igeojson.to_json())
    geojson = pdk.Layer(
        "GeoJsonLayer",
        data=igeojson.to_json(),
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="25",
        get_fill_color="[255, 255,  89]",
        get_line_color=[255, 255, 89],
    )
    #print(igeojson)
    print(igeojson['geometry'].head())
    r = pdk.Deck(layers=[ geojson, pdk.Layer("GeoJsonLayer", data=igeojson, get_fill_color=[255, 255, 89],)], initial_view_state=INITIAL_VIEW_STATE)
    return r

def geojson_layer2(geojson):

            # Load in the JSON data
            DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"
            #print(pd.read_json(DATA_URL))
            #print(pd.read_json(geojson))
            djson =pd.read_json(geojson) #pd.read_json(DATA_URL)
            #print(json.loads(geojson))
            df = pd.DataFrame()

            # Custom color scale
            COLOR_RANGE = [
                [65, 182, 196],
                [127, 205, 187],
                [199, 233, 180],
                [237, 248, 177],
                [255, 255, 204],
                [255, 237, 160],
                [254, 217, 118],
                [254, 178, 76],
                [253, 141, 60],
                [252, 78, 42],
                [227, 26, 28],
                [189, 0, 38],
                [128, 0, 38],
            ]

            BREAKS = [1, 2, 3, 4, 5, 6,7,8,9,10]

            def color_scale(val):
                print(val)
                #for i, b in enumerate(BREAKS):
                #    if val <= b:
                #        return COLOR_RANGE[i]
                #return COLOR_RANGE[i]


            # Parse the geometry out in Pandas
            df["coordinates"] = djson["features"].apply(lambda row: row["geometry"]["coordinates"])
            df["COD_PREDIO"] = djson["features"].apply(lambda row: row["properties"]["COD_PREDIO"])
            df["class_km"] = djson["features"].apply(lambda row: row["properties"]["class_km"])
            #"VALOR_TERR_X", "VALORCOMER_X", "VAL_ADQUIS_X"
            df["VALOR_TERR_X"] = djson["features"].apply(lambda row: row["properties"]["VALOR_TERR_x"])
            df["VALORCOMER_X"] = djson["features"].apply(lambda row: row["properties"]["VALORCOMER_x"])
            df["VAL_ADQUIS_X"] = djson["features"].apply(lambda row: row["properties"]["VAL_ADQUIS_x"])
            df["growth"] = djson["features"].apply(lambda row: row["properties"]["class_km"])
            #df["elevation"] = json["features"].apply(lambda row: calculate_elevation(row["properties"]["class_km"]))
            df["elevation"] = djson["features"].apply(lambda row: 1)
            #df["fill_color"] = json["features"].apply(lambda row: color_scale(int(row["properties"]["class_km"])))
            df["fill_color"] = djson["features"].apply(lambda row: COLOR_RANGE [random.randint(1,10)])

            # Add sunlight shadow to the polygons
            sunlight = {
                "@@type": "_SunLight",
                "timestamp": 1564696800000,  # Date.UTC(2019, 7, 1, 22),
                "color": [255, 255, 255],
                "intensity": 1.0,
                "_shadow": True,
            }

            ambient_light = {"@@type": "AmbientLight", "color": [255, 255, 255], "intensity": 1.0}

            lighting_effect = {
                "@@type": "LightingEffect",
                "shadowColor": [0, 0, 0, 0.5],
                "ambientLight": ambient_light,
                "directionalLights": [sunlight],
            }

            view_state = pdk.ViewState(
                **{"latitude": -7.186057119699531, "longitude":-78.52958679199219, "zoom": 11, "maxZoom": 16, "pitch": 45, "bearing": 0}
            )

            LAND_COVER = [
          [
            [
              -78.52958679199219,
              -7.186057119699531
            ],
            [
              -78.46538543701172,
              -7.186057119699531
            ],
            [
              -78.46538543701172,
              -7.1342790232488245
            ],
            [
              -78.52958679199219,
              -7.1342790232488245
            ],
            [
              -78.52958679199219,
              -7.186057119699531
            ]
          ]
        ]



            polygon_layer = pdk.Layer(
                "PolygonLayer",
                df,
                id="geojson",
                opacity=0.6,
                stroked=False,
                get_polygon="coordinates",
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation="elevation",
                get_fill_color="fill_color",
                get_line_color=[255, 255, 255],
                auto_highlight=True,
                pickable=True,
            )
            # "VALOR_TERR_X", "VALORCOMER_X", "VAL_ADQUIS_X"
            tooltip = {"html": "<b>Cod. Predio:</b> {COD_PREDIO} <br /><b>Kmeans:</b> {class_km} <br /><b>Val. Terreno:</b> {VALOR_TERR_X}<br /><b>Val. Adquisicion:</b> {VAL_ADQUIS_X}<br /><b>Val. Comercial:</b> {VALORCOMER_X} "}

            r = pdk.Deck(
                polygon_layer,
                initial_view_state=view_state,
                effects=[lighting_effect],
                map_provider="mapbox",
                map_style=pdk.map_styles.SATELLITE,
                tooltip=tooltip,
            )
            return r




def app():

    st.title("Aplicacion de uso de Kmeans")

    col1, col2 = st.columns([2, 1])
    row2 = st.columns([1])
    with col2:

        filecsv = st.file_uploader("DataSet CSV", type=["csv"])
        encodng = st.text_input('Encoding', 'UTF-8')
        separador = st.text_input('Separador', ';')

        if filecsv and encodng and  separador:
            data = pd.read_csv(filecsv, encoding=encodng, sep=separador)
            #muestra los campor de la tabla a mostrar
            selected_columns = st.multiselect("Seleccione los campos de la tabla", data.columns)
            #print(selected_columns)
            #Se tiene para elegir el numero de clusters
            n_cluster_list = st.multiselect("Defina el numero de clusters", [3, 4, 5, 6, 7, 8, 9, 10, 11])

            if selected_columns:
                #n_cluster_list =[3, 4, 5, 6, 7, 8, 9, 10, 11]
                datos = data[selected_columns]
                sc = StandardScaler()
                X_std = sc.fit_transform(datos)
                #print(X_std)
                # n_cluster_list =
                init = 'k-means++'  # elija k observaciones (filas) para los centroides iniciales
                n_init = 10  # número de veces que el algoritmo se ejecutará
                max_iter = 300  # número máximo de iteraciones para una ejecución
                tol = 1e-04  # tolerancia para declarar convergencia
                semilla = 6

                with col1:
                    Grafico_de_silueta(X_std, n_cluster_list, init, n_init, max_iter, tol, semilla)

                with st.container():
                    with col1:
                        st.title("Número óptimo de clúster según método del codo aplicando kmeans")
                        distorcion(X_std, n_cluster_list, init, n_init, max_iter, tol, semilla)

                with st.container():
                    with col2:

                        st.title("Agrupamiento de datos con kmeans")
                        # Creamos una instancia de KMeans y creamos 5 cluster de acuerdo al criterio de la silueta
                        km = KMeans(n_clusters=10,  # número de clúster a formar
                                    init="k-means++",  # centroides iniciales aleatorios
                                    n_init=10,  # número de veces que se ejecuta el algoritmo
                                    max_iter=300,  # número máximo de iteraciones para una ejecución
                                    tol=1e-04,  # tolerancia para declarar convergencia
                                    random_state=2021)  # semilla
                        y_km = km.fit_predict(X_std)
                        #print(y_km)
                        data['class_km'] = y_km
                        #print(data.head())

                        fileshp = st.file_uploader('Ruta Archivo ShapeFile', type=["geojson","zip"])
                        if fileshp:
                            mapdatos = uploaded_file_to_gdf(fileshp)
                            #mapdatos = uploaded_file_to_gdf(fileshp)
                            #mapdatos.to_crs(epsg=4326)
                            #print(mapdatos.head(2))
                            leftcampo_union = st.selectbox("Seleccione columna del Geojson", mapdatos.columns)
                            #print(leftcampo_union)
                            rigthcampo_union = st.selectbox("Seleccione columna del CSV", data.columns)
                            #print(rigthcampo_union)

                            mapdatos[leftcampo_union]=mapdatos[leftcampo_union].astype(str)
                            data[rigthcampo_union]=data[rigthcampo_union].astype(str)
                            Union = mapdatos.merge(data, left_on=str(leftcampo_union), right_on=str(rigthcampo_union))
                            #Union = pd.merge(mapdatos, data, how="inner", on=[leftcampo_union, rigthcampo_union])
                            Union = Union.to_crs(epsg=4326)
                            #print(Union.crs)
                            #print(type(Union))
                            #print(Union.head())
                            #Union.plot(figsize=(6, 6))
                            fig3 = plt.figure()
                            #st.pyplot(fig3)
                            #plt.show()
                            with st.container():
                                with col1:
                                    st.header("Visor de mapa del resultado")
                                    #print(Union.to_json())
                                    #st.pydeck_chart(geojson_layer(Union.to_json()))
                                    print(Union.head())
                                    #st.pydeck_chart(geojson_layer2(Union[["geometry","COD_PREDIO","VALOR_TERR_x","VALORCOMER_x","VAL_ADQUIS_x","class_km"]].to_json()))
                                    st.pydeck_chart(geojson_layer2(Union.to_json()))
