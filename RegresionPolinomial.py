"""
Regresion polinomial

@author Anibal Alas
"""
##Librerias a utilizar
##se importaran las librerias
import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

###PREPARAR LA DATA

##Importamos los datos de la misma libreria de scikit-learn
diabetes = datasets.load_diabetes()
print(diabetes)
print()

################### Entendimiento de la data ####################
#verifico la informacion contenida en el dataset
print("Informacion en el dataset:")
print(diabetes.keys())

#Verifico caracteristicas del dataset
print("Caracteristicas del dataset")
print(diabetes.DESCR)

#Verifico la cantidad de datos que hay en los dataset
print("Cantidad de datos:")
print(diabetes.data.shape)
print()

################PREPARAR LA DATA REGRESION POLINOMIAL###############################
#Seleccionamos solamente la columna 6 del dataset
x_p=diabetes.data[:,np.newaxis,5]

#Definimos los datos correspondientes a las etiquetas
y_p=diabetes.target

#Graficamos los datos correspondientes
plt.scatter(x_p,y_p)
plt.show()

################## IMPLEMENTACION DE REGRESION POLINOMIAL ##########################
#Separo los datos de trian en entrenamiento y prueba para probar los algoritmos
x_train_p,x_test_p, y_train_p,y_test_p=train_test_split(x_p,y_p,test_size=0.2)

#Se define el grado del polinomio
polig_reg=PolynomialFeatures(degree=2)

#Se transforma las caracteristicas existentes en caracteristicas de mayor grado
xtrain_poli=polig_reg.fit_transform(x_train_p)
x_test_poli=polig_reg.fit_transform(x_test_p)

#Defino el algoritmo a utilizar
pr=linear_model.LinearRegression()

#Entreno el modelo
pr.fit(xtrain_poli,y_train_p)

#Realizo una prediccion
y_pred_pr = pr.predict(x_test_poli)

#Graficamos datos junto con el modelo
plt.scatter(x_test_p,y_test_p)
plt.plot(x_test_p,y_pred_pr,color="red", linewidth=3)
plt.show()

print()
print("Datos del modelo Regresion polinomial")
print()

print("Valor de la pendiente o coeficiente a:")
print(pr.coef_)

print()
print("valor de la interseccion o coeficiente b:")
print(pr.intercept_)

print()
print("Precision del modelo:")
print(pr.score(xtrain_poli,y_train_p))