"""
Algoritmo de Arboles de decision regresion
Author: Anibal Alas
"""

#importa la libreria a utilizar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

################ PREPARAR LA DATA ARBOLES DE DECISION REGRESION #########################
#Seleccionamos solamente la columna 6 del dataset
x_adr=diabetes.data[:,np.newaxis,5]

#Defino los datos correspondientes a las etiquetas
y_adr=diabetes.target

#Graficamos los datos correspondientes
plt.scatter(x_adr,y_adr)
plt.show()

######### IMPLEMENTACION DE ARBOLES DE DECISION REGRESION ############
#Separo los datos de trian en entrenamiento y prueba para probar los algoritmos
xtrain,xtest,ytrain,ytest=train_test_split(x_adr,y_adr,test_size=0.2)

#Defino el algoritmo a utilizar
adr=DecisionTreeRegressor(max_depth=5)

#Entreno del modelo
adr.fit(xtrain,ytrain)

#Realizo una prediccion
y_pred=adr.predict(xtest)

#Graficamos los datos de prueba con la prediccion
xgrid=np.arange(min(xtest),max(xtest),0.1)
xgrid=xgrid.reshape((len(xgrid),1))
plt.scatter(xtest,ytest)
plt.plot(xgrid,adr.predict(xgrid),color="red", linewidth=3)
plt.show()

print("DATOS DEL MODELO ARBOLES DE DECISION REGRESION")
print()

print("Precision del modelo")
print(adr.score(xtrain,ytrain))