"""
 Maquinas Vectores de Sporte Clasificacion
 @author: Anibal Alas
"""

###################Librerias a utilizar ####################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

############# IMPLEMENTACION DE MAQUINAS VECTORES DE SOPORTE ####################

#separo los datos de trian en entrenamiento y prueba para probar los a:
xtra,xtes,ytra,ytes=train_test_split(x,y,test_size=0.2)

#Defino algoritmo a clasificar
algoritm=SVC(kernel="linear")

#Entreno del modelo
algoritm.fit(xtra,ytra)

#Realizo una prediccion
y_pd=algoritm.predict(xtes)

#Verifico Matriz de confusion
mt=confusion_matrix(ytes,y_pd)
print("Matriz de Confusion:")
print(mt)

#Calculo de precision del modelo
p=precision_score(ytes,y_pd)
print("Precision del modelo:")
print(p)
