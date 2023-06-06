import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
labels = np.random.randint(0, 2, 100) 
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter (x, y, c=colors)
plt.show()

# random black and white image
image = np.random.rand(10, 10)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()