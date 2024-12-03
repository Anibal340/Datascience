

import numpy as np


arrays= np.array([[1,2,3,4],[5,6,7,8]], dtype=np.int64)
print("arrray con tipo de dato definido")
print(arrays)

#Crear una matriz de unos -3 filas 4 columnas
print("matriz de unos")
unos=np.ones((3,4))
print(unos)
print("matriz de ceros")
ceros=np.zeros((3,4))
print(ceros)
print("matriz aleatorios")
aleatorios=np.random.random((2,2))
print(aleatorios)
print("matriz vacia")
vacia=np.empty((3,2))
print(vacia)
print("Matriz con un solo valor")
full=np.full((2,2),8)
print(full)
print("Matriz Identidad")
identidad=np.eye(4,4)
print(identidad)
identidad1=np.identity(4)
print(identidad1)