# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 10:04:21 2023

@author: sebas

"""

import os.path

import pandas as pd

import glob

class AnalisisDatos:
    """
    Esta clase permite
    
    """
    
    
    def __init__(self,*args,**kwargs):
        self.files = args
        
        self.data = []
        
    def load_data(self):
        
        for i,j in enumerate(self.files):
            if os.path.isfile(j) == True:
                base = pd.read_csv(s)
                
                print(base)
                print("está")


s = "C:/Users\sebas\OneDrive\Escritorio\Programación\parciales\Parcial_2\Cancer_Data.csv"

d = "C:/Users\sebas\OneDrive\Escritorio\Programación\parciales\Parcial_2\csv-1.csv"

tabla = AnalisisDatos(d)

print(tabla.load_data())

# print(os.path.isfile(s))

# print(os.path.exists(d))



# # Read the CSV file

# airbnb_data = pd.read_csv(s)

# print(airbnb_data)

cwd = type(os.getcwd())


print(cwd)

if cwd == s:
    print("chimba")


""" esto me permite leer todos los archivos de la ruta raiz"""

# csv_files = glob.glob('*.csv')

# print(csv_files)

# list_data = []
  
# # Escribimos un loop que irá a través de cada uno de los nombres de archivo a través de globbing y el resultado final será la lista dataframes
# for filename in csv_files:
#     data = pd.read_csv(filename)
#     list_data.append(data)
# #Para chequear que todo está bien, mostramos la list_data por consola
# print(list_data)