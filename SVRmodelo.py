
"""
Algoritmo de vectores de soporte de regresion
Author: Anibal Alas
"""

#importa la libreria a utilizar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
############### PREPARAR LA DATA ###################
# importamos los datos de la misma libreria de scikit-learn
diabetes=datasets.load_diabetes()
print(diabetes)
print()

############ ENTENDIMIENTO DE LA DATA ##############
#VERIFICO INFORMACION CONTENIDA EN EL DATASET
print("informacion en el dataset")
print(diabetes.keys())
print()

#Verifico caracteristicas del dataset
print("Caracteristicas del dataset")
print(diabetes.DESCR)

#Verifico la cantidad de datos que hay en los dataset
print("Cantidad de datos:")
print(diabetes.data.shape)
print()

#verifico la informacion de las columnas
print("Nombres columnas:")
print(diabetes.feature_names)

############ PREPARAR LA DATA VECTORES DE SOPORTE DE REGRESION################
#Seleccionamos solamente la columna 6 del dataset
x_svr=diabetes.data[:,np.newaxis,5]

#Define los datos correspondientes a la etiquetas
y_svr=diabetes.target

#Graficamos los datos correspondientes
plt.scatter(x_svr,y_svr)
plt.show()

############### IMPLEMENTACION DE VECTORES DE SOPORTE REGRESION ##################
#Separo los datos de trian en entrenamientos y prueba para probar los algoritmos
x_train,x_test,y_train,y_test=train_test_split(x_svr,y_svr,test_size=0.2)

#Defino el algoritmo a utilizar
svr=SVR(kernel="linear", C=1.0, epsilon=0.2)

#Entreno del modelo
svr.fit(x_train,y_train)

#Realizo una prediccion
y_pred=svr.predict(x_test)

#Graficamos los datos junto con el modelo
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color="red", linewidth=3)
plt.show()

print()
print("DATOS DEL MODELO DE VECTORES DE SOPORTE REGRESION")
print()

print("Precision del modelo:")
print(svr.score(x_train,y_train))