import pandas as pd
import numpy as np
series = pd.Series({"Argentina":"Buenos Aires", "Chile":"Santiago de Chile", "Colombia":"Bogotá", "Perú":"Lima"})
print('Series:')
print(series)

print("=============================================")

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
print('DataFrame:')
print(df)

print("=============================================")
datos = [ [3, 2, 0, 1], [0, 3, 7, 2] ]
df = pd.DataFrame(datos)
df.shape
print(df)
print("Primera columna del dataframe")
print(df[0])
print("=============================================")
datos = [ [3, 2, 0, 1], [0, 3, 7, 2] ]
df = pd.DataFrame(datos)
print(df.describe())


datos = [ [3, 2, 0, 1], [0, 3, 7, 2] ]
df = pd.DataFrame(datos)
print(df[1])

datos = [ [3, 2, 0, 1], [0, 3, 7, 2] ]
df = pd.DataFrame(datos)
print(df[1])
