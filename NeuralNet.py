import matplotlib.pyplot as plt
import numpy as np

# Plot a sequence of (x, y) points connected as a line
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.show()

# Visualize labeled points as a scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
labels = np.random.randint(0, 2, 100)  # Random labels (0 or 1)
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter(x, y, c=colors)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.show()

# Create a simple image
image = np.random.rand(10, 10)  # Random 10x10 image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Simple Image')
plt.show()