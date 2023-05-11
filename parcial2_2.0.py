# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:34:55 2023

@author: sebas
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import time
import numpy as np
import sympy as sp

tab_default = {'id': [1, 2, 3, 4, 5],
        'nombre': ['Juan', 'Pedro', 'María', 'Luisa', 'Ana'],
        'edad': [25, 30, 27, 23, 28]}

exten = '.csv'
abrir = "Ingrese el archivo que desea abrir: "
mens_caja = "Generando gráfico de caja y bigotes..."
type_dat = 'float', 'int'
selec_arch = "Seleccione un archivo (ingrese un número): "

class DatAnalis:
    """
    la clase genera una base de datos con datos por defecto 
    si no se le pasan argumentos,  si se le pasan rutas lee los archivos csv 
    de la ruta que se le pase como argumento, y si no se le pasan rutas pero 
    sí archivos, lee los archivos csv que se le hayan pasado como argumentos.
    """
    def __init__(self,*archivos,**rutas):
        """
        Argumentos:
            *archivos -- uno o varios nombres de archivo (cadenas de 
             caracteres) a cargar.
            **rutas -- uno o varios pares clave-valor donde la clave es un 
              identificador de ruta (cadena de caracteres)
              y el valor es la ruta (cadena de caracteres) que se debe buscar. 
        """
        if archivos == () and rutas =={}:
            data = tab_default
            print(pd.DataFrame(data))
            # return pd.DataFrame(data)
        elif rutas =={}:
            for i,j in enumerate(archivos):
                data = pd.read_csv(j)
                print(data)
        else:
            for key, value in rutas.items():
                archivos = os.listdir(value)
                
                csv_archivos = [f for f in archivos if f.endswith(exten)]
                print(csv_archivos)
                arch = str(input(abrir))
                file_path = os.path.join(value,arch)
                df = pd.read_csv(file_path)
                print(df)
                
    def calculate_time(func):
        """
        Función decoradora que calcula el tiempo de ejecución de la función que
        recibe como argumento.

        Argumentos:
            func -- la función a la que se le calculará el tiempo de ejecución.

        Retorno:
            Retorna una función que envuelve a la función original y calcula su
            tiempo de ejecución. Imprime en consola el tiempo de ejecución y 
            retorna el resultado de la función original.
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times = f"El tiempo de ejecución de {func.__name__} fue de \
                {end_time - start_time} segundos."
            print(times)
            return result
        return wrapper    
        
    @calculate_time  
    def new_table(self):
        """
        Crea una nueva tabla a partir de dos archivos CSV, permitiendo al 
        usuario seleccionar las columnas de cada archivo
        que desea incluir en la tabla resultante.
        """
        df1 = pd.read_csv('Cancer_Data.csv')
        df2 = pd.read_csv('Heart_disease_cleveland_new.csv')
        
        print("Columnas del primer archivo:")
        print(list(df1.columns))
        print("Columnas del segundo archivo:")
        print(list(df2.columns))
        # Pedir al usuario que seleccione las columnas de la primera base de datos
        print("Seleccione las columnas del primer archivo separadas por comas:")
        selected_cols1 = str(input()).split(",")

        # Pedir al usuario que seleccione las columnas de la segunda base de datos
        print("Seleccione las columnas del segundo archivo separadas por comas:")
        selected_cols2 = str(input()).split(",")

        # Unir las columnas seleccionadas de ambas bases de datos en una nueva tabla
        new_df = pd.concat([df1[selected_cols1], df2[selected_cols2]], axis=1)

        # Mostrar la nueva tabla
        print("Nueva tabla:")
        print(new_df)
        
    def mensaje(func):
        def wrapper(*args, **kwargs):
            print(mens_caja)
            return func(*args, **kwargs)
        return wrapper
    
    @mensaje
    def grafico_caja(self,archivo_csv):
        """
        Crea un gráfico de caja y bigotes (boxplot) para cada una de las 
        variables numéricas especificadas.

        Parámetros:
        - vars_num: una lista de los nombres de las variables numéricas a 
                    graficar.
        """
        # Seleccionar las variables numéricas del archivo CSV
        df = pd.read_csv(archivo_csv)
        vars_numericas = df.select_dtypes(include=['float', 'int']).columns.tolist()

        # Crear el gráfico de caja y bigotes
        plt.figure(figsize=(8, 6))
        plt.boxplot(df[vars_numericas].values)
        plt.title('Gráfico de caja y bigotes para las variables numéricas')
        plt.xlabel('Variables')
        plt.ylabel('Valor')
        plt.xticks(range(1, len(vars_numericas) + 1), vars_numericas)
        plt.show()
        
        datos_numericos = df[vars_numericas].values
        Q1 = np.percentile(datos_numericos, 25, axis=0)
        Q3 = np.percentile(datos_numericos, 75, axis=0)
        IQR = Q3 - Q1
        
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        
        outliers_inf = datos_numericos[datos_numericos < lim_inf].flatten()
        outliers_sup = datos_numericos[datos_numericos > lim_sup].flatten()
        
        print("Valores atípicos inferiores: ", outliers_inf)
        print("Valores atípicos superiores: ", outliers_sup)




    @calculate_time    
    def correlation(self):
        """
        Este metodo hace la correlación entre las columnas de datos numericas 
        de las tablas que existen en una carpeta predeterminada
        """
        # Obtener la ruta de la carpeta que contiene los archivos CSV
        folder_path = os.getcwd()

        # Obtener una lista de los archivos CSV en la carpeta
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')
                     or f.endswith('.xlsx')]

        # Mostrar al usuario la lista de archivos CSV y pedirle que seleccione uno
        print("Archivos CSV y XLSX disponibles:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
        selected_file_index = int(input(selec_arch)) - 1

        # Cargar el archivo CSV seleccionado en un DataFrame de pandas
        selected_file_path = os.path.join(folder_path, files[selected_file_index])
        
        if selected_file_path.endswith(exten):
            
            df = pd.read_csv(selected_file_path)
        else:
            df = pd.read_excel(selected_file_path)
            print(df)
        # Seleccionar las columnas numéricas y calcular la matriz de correlación
        numeric_columns = [col for col in df.columns if 
                           pd.api.types.is_numeric_dtype(df[col])]
        corr_matrix = df[numeric_columns].corr(method='pearson')

        # Mostrar la matriz de correlación
        print("Matriz de correlación:")
        print(corr_matrix)
        
    @calculate_time          
    def regresion(self):
        """
        El metodo hace la regresión lineal de la tabla test
        """

        # Obtener la ruta de la carpeta que contiene los archivos CSV
        folder_path = os.getcwd()

        # Obtener una lista de los archivos CSV en la carpeta
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')
                     or f.endswith('.xlsx')]

        # Mostrar al usuario la lista de archivos CSV y pedirle que seleccione uno
        print("Archivos CSV disponibles:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        selected_file_index = int(input(selec_arch)) - 1

        # Cargar el archivo CSV seleccionado en un DataFrame de pandas
        selected_file_path = os.path.join(folder_path, csv_files
                                          [selected_file_index])
        
        df = pd.read_csv(selected_file_path)

        # Mostrar al usuario las columnas numéricas disponibles y pedirle que 
        #seleccione dos
        numeric_columns = [col for col in df.columns 
                           if pd.api.types.is_numeric_dtype(df[col])]
        print("Columnas numéricas disponibles:")
        for i, col in enumerate(numeric_columns):
            print(f"{i+1}. {col}")
        selected_column_indexes = [int(input(f"Seleccione la columna numérica\
                                             {j+1} (ingrese un número): "))
                                             - 1 for j in range(2)]

        # Seleccionar las dos columnas numéricas elegidas y ajustar un modelo
        #de regresión lineal
        
        selected_column_names = [numeric_columns[index] for index 
                                 in selected_column_indexes]
        x = df[selected_column_names[0]].values.reshape((-1,1))
        y = df[selected_column_names[1]]
        regression_model = linear_model.LinearRegression().fit(x, y)

        # Mostrar los coeficientes de la regresión lineal
        print("Intercept:", regression_model.intercept_)
        print("Pendiente:", regression_model.coef_)

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
        print(" coeficiente de correlación de Pearson = ",
              regression_model.score(x,y))

def analizar_graficar_funcion(func,a,b):
    """
    La función toma tres parámetros:

    func: es la función que se desea analizar y graficar. Debe ser una función 
    de una sola variable (es decir, que tenga un solo parámetro).

    a y b: son los extremos del intervalo sobre el cual se desea analizar 
    la función. La función se evalúa en 10000 puntos equidistantes en este 
    intervalo.

    La función grafica la función original, su integral y su derivada en tres
    subgráficos diferentes. En el subgráfico de la integral, se muestra la 
    función original y el área bajo la curva de la función, así como la 
    antiderivada de la función. En el subgráfico de la derivada, se muestra 
    la función original y su derivada.
    """
    # Crear un rango de valores x para evaluar la función
    x = np.linspace(a, b, 100)

    # Evaluar la función en el rango de valores x
    y = func(x)

    # Calcular la integral de la función, estudiar esta linea
    integral = np.zeros_like(x)
    for i in range(len(x)):
        integral[i] = np.trapz(y[:i+1], x[:i+1])
    
    # Calcular la derivada de la función
    derivada = np.gradient(y, x)
    
    
    
    # Graficar la función original
    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.grid(True)
    plt.title('Función Original')

    # Graficar la integral de la función
    plt.subplot(2, 2, 3)
    plt.plot(x, y, label='Área bajo la curva')
    plt.plot(x, integral, label='Antiderivada')
    
    # Rellenar el área bajo la curva de la función
    plt.fill_between(x, y, alpha=0.3)  
    plt.legend()
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.title('Integral')
    plt.grid(True)
   

    # Graficar la derivada de la función
    plt.subplot(2,2,4)
    plt.plot(x,y, label="Función")
    plt.plot(x, derivada, color="deeppink", label="Derivada")
    plt.legend(loc = "best")
    plt.grid(True)
    # plt.title('Derivada')

    # Mostrar las gráficas
    #plt.tight_layout()
    plt.show()
    
    

    


# seb = DatAnalis(Heart_disease_cleveland_new="C:/Users\sebas\OneDrive\Escritorio\Programación\parciales\Parcial_2")

# cam = DatAnalis("Heart_disease_cleveland_new.csv")

# dai = DatAnalis()

# dai.new_table()

# dai.grafico_caja("Heart_disease_cleveland_new.csv")

# Así se usa la función de derivadas e integrales
funcion = lambda x: x**3 + 2*x # Definición de la 

# Definir la variable simbólica
x = sp.symbols('x')

# Definir la expresión matemática
expr = x**3 + 2*x

# Calcular la derivada
derivada = sp.diff(expr, x)

# Calcular la integral
integral = sp.integrate(expr, x)

# Imprimir los resultados
print("Derivada:", derivada)
print("Integral:", integral)



a = -10  # Límite inferior de integración
b = 10 # Límite superior de integración

analizar_graficar_funcion(funcion,a,b)    
