
"""
Regresion lineal multiple

@autor: Anibal Alas

"""
from matplotlib import pyplot as plt
##################librerias a usar#########################
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
import numpy as np

###PREPARAR LA DATA
#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_diabetes()
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

####PREPARAR LA DATA REGRESION LINEAL MULTIPLE#####
x_multiple=boston.data[:,5:8]
print(x_multiple)

#Definimos los datos correspondientes a las etiquetas
y_multiple=boston.target

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train,x_test,y_train,y_test=train_test_split(x_multiple,y_multiple,test_size=0.2)

#definimos el algoritmo a utilizar
lr_multiple=linear_model.LinearRegression()


#entrenar Modelo
lr_multiple.fit(x_train,y_train)

#Realizo una prediccion
y_pred_multiple=lr_multiple.predict(x_test)

print("Datos del modelo Regresion Lineal multiple")
print(y_pred_multiple)

print("Valor de las pendientes o coeficientes a")
print(lr_multiple.coef_)

print("Valor de la interseccion o coeficiente b:")
print(lr_multiple.intercept_)

print("Precision del algoritmo:")
print(lr_multiple.score(x_train,y_train))

#Graficando los datos junto al modelo

indices = np.arange(len(y_test))  # Índices de los datos de prueba

plt.scatter(indices, y_test, label="Valores reales", color="blue")  # Puntos reales
plt.plot(indices, y_pred_multiple, label="Predicción", color="red", linewidth=2)  # Línea de predicción

plt.title("Regresión Lineal Múltiple")
plt.xlabel("Índice de muestra")
plt.ylabel("Valor medio")
plt.legend()
plt.show()







