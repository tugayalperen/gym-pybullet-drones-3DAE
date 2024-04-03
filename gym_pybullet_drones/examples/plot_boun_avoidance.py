import numpy as np
import matplotlib.pyplot as plt

distance_to_bound = np.linspace(0.1, 0.5, 300)
krep = 0.3
l0 = 0.1
force_mag = krep * (((1 / distance_to_bound) - (1 / l0)) * (1 / np.power(distance_to_bound, 3))) * 1
plt.plot(distance_to_bound, force_mag)
plt.show()
print("a")