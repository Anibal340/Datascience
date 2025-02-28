import matplotlib.pyplot as plt

a = [1, 2, 3, 4]
b = [11, 22, 33, 44]

plt.plot(a, b, color='blue', linewidth=3, label='línea')
plt.legend()
plt.show()