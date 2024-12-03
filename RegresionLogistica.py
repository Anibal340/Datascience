"""
Regresion Logistica
@author: Anibal Alas
"""
###################Librerias a utilizar ####################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, roc_auc_score
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
xtr,xtes,ytr,ytes=train_test_split(x,y,test_size=0.2)

# se escalan todos los datos
escalar=StandardScaler()
xtr=escalar.fit_transform(xtr)
xtes=escalar.transform(xtes)

# Defino algortimo a utilizar
algoritmo=LogisticRegression()

# Entreno el modelo
algoritmo.fit(xtr,ytr)

#Realizo una prediccion
y_prd=algoritmo.predict(xtes)

# Verifico la matriz de confusion
matriz=confusion_matrix(ytes,y_prd)
print("Matriz de Confusion:")
print(matriz)

# Calculo la precision del modelo
precision=precision_score(ytes,y_prd)
print("Precision del modelo:")
print(precision)

#Calculo de la exactitud
exactitud=accuracy_score(ytes,y_prd)
print("Exactitud del modelo:")
print(exactitud)

#Calculo la sensibilidad del modelo
sensibilidad=recall_score(ytes,y_prd)
print("Sensibilidad del modelo:")
print(sensibilidad)

#Calculo el puntaje F1 del modelo
puntajef1=f1_score(ytes,y_prd)
print("Puntaje F1 del modelo:")
print(puntajef1)

#Calculo la curva ROC - AUC del modelo
roc_auc=roc_auc_score(ytes,y_prd)
print("Curva ROC - AUC del Modelo")
print(roc_auc)

print("Precision del modelo:",precision)
print("Exactitud del modelo:",exactitud)
print("Sensibilidad del modelo:",sensibilidad)
print("Puntaje F1 del modelo:",puntajef1)
print("Curva ROC - AUC del modelo:",roc_auc)
