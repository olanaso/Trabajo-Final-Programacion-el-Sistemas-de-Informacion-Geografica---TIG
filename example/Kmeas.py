import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans #Para utilizar el método de KMeans
import numpy as np
import matplotlib.pyplot as plt
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
    cont=0
    for i in n_cluster_list:
        cont += 1
        plt.subplot(3, 4, cont)
        km = KMeans(n_clusters=i,
                        init=init,  #elija k observaciones (filas) para los centroides iniciales
                        n_init=n_init, #número de veces que el algoritmo se ejecutará
                        max_iter=max_iter, #número máximo de iteraciones para una ejecución
                        tol=tol, #tolerancia para declarar convergencia
                        random_state=semilla) #semilla
        y_km = km.fit_predict(X)
        cluster_labels = np.unique(y_km) #valores de clúster
        n_clusters = cluster_labels.shape[0] #núnero de clústers
        silhouette_vals = silhouette_samples(X, y_km, metric='euclidean') #valores de silueta teniendo en cuenta la distancia euclideana

        y_ax_lower, y_ax_upper = 0, 0
        yticks = [] #objeto tipo lista vacío
        fig = plt.figure()
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c] #valores de silueta cuando y_km toma el valor c de los posibles n de clúster
            c_silhouette_vals.sort() #se ordenan de menor a mayor los valores de silueta
            y_ax_upper += len(c_silhouette_vals) #número de valores de silueta
            color = cm.jet(float(i) / km.n_clusters) # definir el color
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                     edgecolor='none', color=color) #visualización de los valores de silueta para k
            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)
        silhouette_avg = np.mean(silhouette_vals)#media de los valores de silueta

        plt.axvline(silhouette_avg, color="red", linestyle="--") # mostrar una línea con los valores medios de silueta
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Coeficiente de Silueta')
        plt.title("Silouette para k= " + str(km.n_clusters) + "\n" + "Coeficiente de Silueta= "+str(round((silhouette_avg),2)))
        plt.tight_layout()

        plt.show()
        #Mostramos en el streamlit
        st.pyplot(fig)




def app():

    st.title("Aplicacion de uso de Kmeans")
    col1, col2 = st.columns([2, 1])


    filecsv=st.file_uploader("Subir el Archivo Dataset", type=["csv"])
    #filecsv = "../data/datos.csv"
    data = pd.read_csv(filecsv, encoding="latin_1", sep=";")



    selected_columns = st.multiselect("Seleccione los campos de la tabla", data.columns)
    n_cluster_list = st.multiselect("Defina el numero de clusters", [3, 4, 5, 6, 7, 8, 9, 10, 11])

    datos = data[selected_columns]
    sc = StandardScaler()
    X_std = sc.fit_transform(datos)
    #n_cluster_list =
    init = 'k-means++'  # elija k observaciones (filas) para los centroides iniciales
    n_init = 10  # número de veces que el algoritmo se ejecutará
    max_iter = 300  # número máximo de iteraciones para una ejecución
    tol = 1e-04  # tolerancia para declarar convergencia
    semilla = 2021
    Grafico_de_silueta(X_std, n_cluster_list, init, n_init, max_iter, tol, semilla)

    fileshp = st.file_uploader("Subir el Archivo Dataset", type=["zip"])
    df = pd.DataFrame(fileshp)
    st.table(df.style.set_precision(2))