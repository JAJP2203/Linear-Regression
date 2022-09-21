#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:03:02 2021

@author: Jose A Juarez_A00572186

Este script crea un algoritmo de regresión lineal para predecir una variable
de salida (y) con base a las variables de entrada (x) de un dataset en específico.

En este caso este script va a predecir la calidad de un vino (variable de salida) 
dependiendo de varios factores (variables de entrada), pero esto es así porque es la base 
de datos que se está utilizando para este ejercicio de regresión lineal.

Así mismo, se utilizan 4 librerías para poder leer la base de datos, hacer la
regresión lineal con herramientas matemáticas y poder crear un gráfico con los resultados.

Finalmente, la base de datos que se va a utilizar en este código es sobre la calidad de un vino 
y la puedes encontrar aquí:
    https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009 
"""
import pandas as pd #Importamos la librería pandas para poder importar y leer la base de datos aquí en python. Para posteriormente hacer uso para la regresión lineal. Así mismo, le asignamos la variable "pd". 
import numpy as np #Importamos la librebría numpy para hacer la Raíz de la desviación media al cuadrado del dataset. Así mismo, le asignamos la variable "np".

#Importar el dataset a Python
dataset = pd.read_csv("winequality-red.csv") #Aquí se importa el dataset como una variable.

print("\nPequeña prueba de los datos en el dataset(primeras 5 filas): ")#Imprimimos los datos del datset, pero solo las primeras 5 filas.
print(dataset.head())#Aquí se muestra una pequeña prueba de los datos, solo se muestran las primeras 5 filas.

datos_vino = dataset.iloc[:,3:12] #Aquí se crea un nuevo database con los datos que queremos para el análisis porque no todos las variables de entrada (x) nos sirven para la variable de salida (y). En este caso, dejamos todas las filas y eliminamos las primeras 3 columnas.

print("\nPequeña prueba de los datos en el nuevo dataset(primeras 5 filas): ")#Imprimimos los datos del nuevo datset, pero solo las primeras 5 filas.
print(datos_vino.head())#Aquí se muestra una pequeña prueba de la nueva database, solo se muestran las primeras 5 filas.

#Realizar la estadística descriptiva de nuestros datos
estadistica_descriptiva = datos_vino.describe() #Aquí se calculan las estadísticas descriptivas del database nuevo, el que ya tiene todas las variables necesarias.

print("\nEstadística descriptiva del nuevo dataset: ")#Imprimimos un aviso para introducir las estadísticas descriptivas de los valores en el nuevo dataset.
print(estadistica_descriptiva) #Aquí se imprime en la consola todos los datos.

#Limpiar nuestros datos
datos_vino.isnull().any()#Vamos a buscar datos nulos (que no tengan valor)

dataset = datos_vino.fillna(method="ffill")#Eliminamos estos datos nulos.

#Asignar nuestras variables X y Y
X = dataset[["volatile acidity","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].values #Asignamos las variables independientes 

Y = dataset["quality"].values #Asignamos las variables dependientes.

#Importar librería de SKLearn para dividir los datos (80 y 20)

from sklearn.model_selection import train_test_split #Importamos la herramienta de train_test_split de la librería SKlearning. Esta nos va a ayudar a dividir los datos para que la inteligencia artificial haga sus pruebas y prediga la calidad del vino.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #Dejamos 80% de los datos  ("X" y "Y")para hacer entrenamiento y el 20% para hacer el examen.

#Importar función de la librería SKLearn para hacer la regresión lineal
from sklearn.linear_model import LinearRegression #Importamos la herramienta de LinearRegression de la librería SKlearning. Esta va a hacer la regresión lineal

modelo_regresion = LinearRegression() #Creamos la variable del modelo de regresión

modelo_regresion.fit(X_train, Y_train) #Ajustamos la herramienta de LinearRegression a nuestra base de datos para que pueda hacer el análisis.

#Para este punto el algoritmo ya aprendió los valores de "X" y de "Y", ya nada más escribimos el código de abajo para desplegar los coeficientes de cada uno.
X_columns = ["volatile acidity","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
coeff_df = pd.DataFrame(modelo_regresion.coef_, X_columns, columns=['Coeficientes'])
print("\nCoeficientes variables de entrada (x): ")#Imprimimos un mensaje introductorio a los coeficientes
print(coeff_df)

#Probar el modelo
Y_pred = modelo_regresion.predict(X_test)#Probar nuestro modelo con los valores de prueba

#Comparar datos 
validacion = pd.DataFrame({'Actual': Y_test, 'Predicción': Y_pred})#Comparamos los valores de predecir y los valores verdaderos

muestra_validacion = validacion.head(25)#Muestra una muestra de 25 valores
print("\nValidación: ")#Imprimos un aviso para introducir la validación de los valores
print(muestra_validacion)#Imprimimos esta muestra

#Comprobar precisión del modelo

from sklearn import metrics #Importamos la herramienta metrics de la librería Sklearning. Esta nos va a ayudar a sacar la Raíz de la desviación media al cuadrado. La cuál nos va a decir que tan preciso es nuestro algoritmo.

print("\nRaíz de la desviación media al cuadrado:", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))) #Aquí se imprime el resultado

#Crear un gráfico de los resultados
import matplotlib.pyplot as plt #Importamos la librería matplotlib.pyplot como la variable "plt".

muestra_validacion.plot.bar(rot=0) #Creamos un gráfico de barras con el dataframe que contiene nuestros datos actuales y de predicción

print("\nGráfico de barras comparando calidad de vino: ")#Imprimimos el mensaje introductorio al gráfico
plt.title("Comparación de calidad de vino actuales y de predicción") #Se imprime el título del gráfico
plt.xlabel("Muestra de vinos") #Se le pone la etiqueta del eje "X"
plt.ylabel("Calidad de vino (Escala del 0(malo) al 10(bueno))") #Se le pone la etiqueta del eje "Y"
plt.savefig("plot.png") #Se guarda el gráfico como una foto en formato png
