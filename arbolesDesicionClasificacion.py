"""
 Arboles de decision de clasificacion
 @author: Anibal Alas
"""

###################Librerias a utilizar ####################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

############# IMPLEMENTACION DE Arboles de desicion clasificacion ####################
#separo los datos de trian en entrenamiento y prueba para probar los a:
xtran,xtest,ytran,ytest=train_test_split(x,y,test_size=0.2)

#Defino algoritmo a utilizar
algo=DecisionTreeClassifier(criterion="entropy")

#Entreno el modelo
algo.fit(xtran,ytran)

#Prediccion del modelo
ypd=algo.predict(xtest)

#Matriz de confusion
mtz=confusion_matrix(ytest,ypd)
print("Matriz de Confusion:")
print(mtz)

#Precision del modelo
prc=precision_score(ytest,ypd)
print("Precision del modelo:")
print(prc)


