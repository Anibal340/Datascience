import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
boston =datasets.load_digits()
print(boston)
print()

print("Informacion del datasets")
print(boston.keys())
print()

print("Caracteristicas del datasets")
print(boston.DESCR)
print()

print("Cantidad de datos")
print(boston.data.shape)
print()

print("Nombre columnas:")
print(boston.feature_names)
print()

x=boston.data[:,np.newaxis,5]
y=boston.target

plt.scatter(x,y)
plt.xlabel("numero habitaciones")
plt.ylabel("Valor medio")
plt.show()


#Implementacion de regresion lineal simple
from sklearn.model_selection import train_test_split
from sklearn import linear_model
#Separa los datos de train en entrenamiento de prueba para probar los algoritmos
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Define el algoritmo a utilizar
lr=linear_model.LinearRegression()
#Entreno el modelo
lr.fit(X_train,y_train)
#datos de prediccion
y_pred=lr.predict(X_test)

#Graficando los datos junto al modelo
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color="red",linewidth=3)
plt.title("Regresion Lineal simple")
plt.xlabel("numero de habitaciones")
plt.ylabel("Valor medio")
plt.show()

print()
print("Datos del modelo de regresion lineal simple")
print()
print("valor del coeficiente a:")
print(lr.coef_)
print("Valor de la interseccion o coeficiente b")
print(lr.intercept_)
print()
print("La ecuacion del modelo es igual a:")
print("y = ",lr.coef_,"x",lr.intercept_)

print()
print("precision del modelo")
print(lr.score(X_train,y_train))


