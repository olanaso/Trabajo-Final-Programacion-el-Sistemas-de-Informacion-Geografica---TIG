import streamlit as st
from sklearn.cluster import KMeans #Para utilizar el método de KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
#import geopandas as gpd

from sklearn.preprocessing import StandardScaler #Utilizado para la estandarización.
from sklearn.metrics import silhouette_samples



#Definimos la funcion que generar el grafico
def Grafico_de_silueta(X,n_cluster_list,init,n_init,max_iter,tol,semilla):
    print("x",X)
    print("n_cluster_list",n_cluster_list)
    print("init",init)
    print("n_init",n_init)
    print("max_iter",max_iter)
    print("tol",tol)
    print("semilla",semilla)
    cont = 0
    #fig = plt.figure()
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
            c_silhouette_vals = silhouette_vals[y_km == c]  # valores de silueta cuando y_km toma el valor c de los posibles n de clúster
            c_silhouette_vals.sort()  # se ordenan de menor a mayor los valores de silueta
            y_ax_upper += len(c_silhouette_vals)  # número de valores de silueta
            color = cm.jet(float(i) / km.n_clusters)  # definir el color
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                     edgecolor='none', color=color)  # visualización de los valores de silueta para k
            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)
        silhouette_avg = np.mean(silhouette_vals)  # media de los valores de silueta
        print("Punto 04")
        plt.axvline(silhouette_avg, color="red", linestyle="--")  # mostrar una línea con los valores medios de silueta
        print("Punto 051")
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Coeficiente de Silueta')
        plt.title("Silouette para k= " + str(km.n_clusters) + "\n" + "Coeficiente de Silueta= " + str(
            round((silhouette_avg), 2)))
        print("Punto 06")
        plt.tight_layout()

        #Mostramos en el streamlit
        print("Punto 07")
        st.pyplot(fig)
        print("Punto 05")




def app():

    st.title("Aplicacion de uso de Kmeans")
    col1, col2 = st.columns([2, 1])
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
            #n_cluster_list = st.multiselect("Defina el numero de clusters", [3, 4, 5, 6, 7, 8, 9, 10, 11])

            if selected_columns:
                n_cluster_list =[3, 4, 5, 6, 7, 8, 9, 10, 11]
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
                Grafico_de_silueta(X_std, n_cluster_list, init, n_init, max_iter, tol, semilla)

