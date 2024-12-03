"""
K vecinos mas cercanos
@author: Anibal Alas
"""

###################Librerias a utilizar ####################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
################ PREPARAR LA DATA ##################
dataset=datasets.load_breast_cancer()
print(dataset)

################# ENTENDIMIENTO DE LA DATA #######################
##Verfico la informcion contenida en el dataset
print("Informacion en el dataset:")
print(dataset.keys())

# Verifico caracteristicas del dataset
print("Caracteristicas del dataset:")
print(dataset.DESCR)

# Seleccionamos todas las columnas
x=dataset.data

# Definimos los datos correspondiente a las etiquetas
y=dataset.target

############# IMPLEMENTACION DE REGRESION LOGISTICA ####################

#separo los datos de trian en entrenamiento y prueba para probar los a:
xt,xte,yt,yte=train_test_split(x,y,test_size=0.2)
#Defino algoritmo a clasificar
algortmo=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)

#Entreno del modelo
algortmo.fit(xt,yt)

#Realizo una prediccion
yp=algortmo.predict(xte)

#Verifico la matriz
mtz=confusion_matrix(yte,yp)
print("Matriz de Confusion:")
print(mtz)

#Calculo de precision del modelo
prc=precision_score(yte,yp)
print("Precision del modelo:")
print(prc)
