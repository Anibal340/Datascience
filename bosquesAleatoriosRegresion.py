"""
Algoritmo de Arboles de decision regresion
Author: Anibal Alas
"""

#importa la libreria a utilizar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


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

################ PREPARAR LA DATA BOSQUES ALEATORIOS REGRESION ###########################
#  Seleccionamos solamente la columna 6 del dataset
x_bar=diabetes.data[:,np.newaxis, 5]

#Defino los datos correspondientes a las etiquetas
y_bar=diabetes.target

#Graficamos los datos correspondientes
plt.scatter(x_bar,y_bar)
plt.show()

######### IMPLEMENTACION DE BOSQUES ALEATORIOS REGRESION ######################

#Sepera los datos de trian en entrenamiento y prueba para probar los algoritmos
xtrain,xtest,ytrian,ytest=train_test_split(x_bar,y_bar,test_size=0.2)

#Defino el algoritmo a utilizar
bar=RandomForestRegressor(n_estimators=300,max_depth=8)

#Entreno del modelo
bar.fit(xtrain,ytrian)

# Realizo de la prediccion
y_pred=bar.predict(xtest)

#Graficamos los datos de prueba junto con la prediccion
xgrid=np.arange(min(xtest),max(xtest), 0.1)
xgrid=xgrid.reshape((len(xgrid),1))
plt.scatter(xtest,ytest)
plt.plot(xgrid, bar.predict(xgrid),color="red", linewidth=3)
plt.show()

print("DATOS DEL MODELO DE BOSQUES ALEATORIOS DE REGRESION")
print()

print("precision del modelo:")
print(bar.score(xtrain,ytrian))
