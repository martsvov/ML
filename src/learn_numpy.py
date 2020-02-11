import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set()


rand = np.random.RandomState(42)
X = rand.rand(10, 2)

plt.scatter(X[:, 0], X[:, 1], s=100)
plt.show()
