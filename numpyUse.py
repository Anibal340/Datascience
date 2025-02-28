import sys
import numpy as np
import time
S = range(1000)
print("Resultado lista de Python")
print(sys.getsizeof(5)*len(S))
print()
D=np.arange(1000)
print("Resultado Numpy array:")
print(D.size*D.itemsize)

print("==============================================")
a=np.array([1,2,3])
print("1D array:")
print(a)
print()
b=np.array([(1,2,3),(4,5,6)])
print("2D array")
print(b)
print("=============================================")


SIZE=1000000
L1=range(SIZE)
L2=range(SIZE)
A1=np.arange(SIZE)
A2=np.arange(SIZE)

start=time.time()
result=[(x,y) for x,y in zip(L1,L2)]
print("Resultado lista de Python")
print((time.time()-start)*1000)
print()
start=time.time()
result=A1+A2
print("resultado Numpy array:")
print((time.time()-start)*1000)
