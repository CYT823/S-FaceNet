import matplotlib.pyplot as plt

x = [38.29459953,73.53179932,56.02519989,41.54930115,70.72990036]
y = [51.69630051,51.50139999,71.73660278,92.36550140,92.20410156]

plt.scatter(x, y, s=30)
plt.axis([0, 112, 0, 112])
plt.show()
