# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:37:57 2023

@author: sebas
"""

import numpy as np
import matplotlib.pyplot as plt

def analizar_graficar_funcion(func,a,b):
    # Crear un rango de valores x para evaluar la función
    x = np.linspace(a, b, 10000)

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
    plt.fill_between(x, y, alpha=0.3)  # Rellenar el área bajo la curva de la función
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
    plt.title('Derivada')

    # Mostrar las gráficas
    #plt.tight_layout()
    plt.show()