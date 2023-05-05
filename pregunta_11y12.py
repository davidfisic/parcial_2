# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:24:59 2023

@author: sebas
"""

import os
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Obtener la ruta de la carpeta que contiene los archivos CSV
folder_path = "C:/Users\sebas\OneDrive\Escritorio\parcial_2"

# Obtener una lista de los archivos CSV en la carpeta
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Mostrar al usuario la lista de archivos CSV y pedirle que seleccione uno
print("Archivos CSV disponibles:")
for i, file in enumerate(csv_files):
    print(f"{i+1}. {file}")
selected_file_index = int(input("Seleccione un archivo (ingrese un número): ")) - 1

# Cargar el archivo CSV seleccionado en un DataFrame de pandas
selected_file_path = os.path.join(folder_path, csv_files[selected_file_index])
df = pd.read_csv(selected_file_path)

# Mostrar al usuario las columnas numéricas disponibles y pedirle que seleccione dos
numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
print("Columnas numéricas disponibles:")
for i, col in enumerate(numeric_columns):
    print(f"{i+1}. {col}")
selected_column_indexes = [int(input(f"Seleccione la columna numérica {j+1} (ingrese un número): ")) - 1 for j in range(2)]

# Seleccionar las dos columnas numéricas elegidas y ajustar un modelo de regresión lineal
selected_column_names = [numeric_columns[index] for index in selected_column_indexes]
x = df[selected_column_names[0]].values.reshape((-1,1))
y = df[selected_column_names[1]]
regression_model = linear_model.LinearRegression().fit(x, y)

# Mostrar los coeficientes de la regresión lineal
print("Intercept:", regression_model.intercept_)
print("Coeficiente:", regression_model.coef_)

# Realizar predicciones para algunas entradas
entrada = [[10], [21], [25], [40]]
predicciones = regression_model.predict(entrada)
print("Predicciones:", predicciones)

# Graficar los datos y la línea de regresión
plt.scatter(x, y, color="red")
plt.plot(x, regression_model.predict(x), color="black")
plt.scatter(entrada, predicciones, color="black")
plt.xlabel(selected_column_names[0])
plt.ylabel(selected_column_names[1])
plt.show()
