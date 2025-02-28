import matplotlib.pyplot as plt
#Definir los datos
x1 = [3, 4, 5, 6]
y1 = [5, 6, 3, 4]
x2 = [2, 5, 8]
y2 = [3, 4, 3]

#Configurar las características del gráfico
plt.plot(x1, y1, label = 'Línea 1', linewidth = 4, color = 'blue')
plt.plot(x2, y2, label = 'Línea 2', linewidth = 4, color = 'green')

#Definir título y nombres de ejes
plt.title('Diagrama de Líneas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda, cuadrícula y figura
plt.legend()
plt.grid()
plt.show()