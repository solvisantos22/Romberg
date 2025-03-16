import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nauðsynlegt til 3D myndunar

test_functions = {
    "Flókið sveiflukennt fall": lambda x, y: np.sin(10*x) * np.cos(10*y) + np.exp(-5*(x**2 + y**2)),
    "Gaussískt fall": lambda x, y: np.exp(-10*((x - 0.3)**2 + (y - 0.7)**2)),
    "Margliðu–veldisvísis fall": lambda x, y: (x**3 + y**4) * np.exp(-x - y),
    "Sveiflukennt ripple": lambda x, y: np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y),
    "Samsett fall með háum sveiflum": lambda x, y: (np.sin(10*x) * np.cos(10*y) +
                                 np.exp(-5*((x - 0.5)**2 + (y - 0.5)**2)) +
                                 0.5 * np.exp(-10*((x - 0.2)**2 + (y - 0.8)**2)))
}

x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15,10), subplot_kw={'projection': '3d'})

axs = axs.flatten()

for ax, (name, func) in zip(axs, test_functions.items()):
    Z = func(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

for ax in axs[len(test_functions):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()