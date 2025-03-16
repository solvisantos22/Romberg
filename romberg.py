import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def exact_integral(f):
    """
    Reiknar nákvæmt heildi
    """
    result, _ = dblquad(f, 0, 1, lambda y: 0, lambda y: 1)
    return result

def trapezoidal_2d(f, a, b, c, d, n, ax=None, plot_3d=False):
    """
    Reiknar trapisu reglu nálgun fyrir heildi f yfir [a,b]x[c,d]. Plottar einnig.
    n: fjöldi hlutbila, þarf að vera veldi af 2
    """
    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, n+1)
    hx = (b - a) / n
    hy = (d - c) / n

    integral = 0
    grid_x, grid_y, grid_z = [], [], []
    for i in range(n+1):
        for j in range(n+1):
            value = f(x[i], y[j])
            integral += value * hx * hy
            grid_x.append(x[i])
            grid_y.append(y[j])
            grid_z.append(value)

            # Plot trapezoidal regions if axis is provided
            if ax is not None:
                rect = patches.Rectangle((x[i], y[j]), hx, hy, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    if plot_3d:
        fig = plt.figure(figsize=(8, 6))
        ax3d = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        ax3d.scatter(grid_x, grid_y, grid_z, color='red', s=10)
        ax3d.set_title(f'3D framsetning fyrir n={n}')
        plt.savefig(os.path.join(output_dir, f'trap_3d_n{n}.png'))
        plt.close()

    return integral

def plot_trapezoids(a, b, c, d, n, iter_num):
    """
    Plottar fall í 2d
    """
    fig, ax = plt.subplots(figsize=(6, 6))


    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, n+1)
    hx = (b - a) / n
    hy = (d - c) / n

    for i in range(n):
        for j in range(n):
            color = 'blue' if (i + j) % 2 == 0 else 'red'  # Alternate colors
            rect = plt.Rectangle((x[i], y[j]), hx, hy, edgecolor=color, fill=False, linewidth=1)
            ax.add_patch(rect)

    ax.set_xlim(a - 0.1, b + 0.1)
    ax.set_ylim(c - 0.1, d + 0.1)
    ax.set_title(f"Skipting í ítrun {iter_num}")
    ax.set_aspect('equal')

    plt.savefig(os.path.join(output_dir, f'trapezoidal_division_iter_{iter_num}.png'))
    plt.close()

def romberg_2d(f, a, b, c, d, max_iter=5):
    """
    Framkvæmir Romberg heildun í 2d með trapisureglu
    """
    R = np.zeros((max_iter, max_iter))
    exact_value = exact_integral(f)

    print(f"Rétt gildi: {exact_value:.6f}")
    print("Ítrun     | Nálgun        | Skekkja")
    print("------------------------------------------")

    for i in range(max_iter):
        n = 2**i
        R[i, 0] = trapezoidal_2d(f, a, b, c, d, n, plot_3d=True)

        plot_trapezoids(a, b, c, d, n, iter_num=i)

        error = abs(R[i, 0] - exact_value)
        print(f"{i+1:<9} | {R[i, 0]:<13.6f} | {error:.6e}")

        for j in range(1, i + 1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    plt.savefig(os.path.join(output_dir, 'romberg_trapezoids.png'))
    plt.close()
    return R[max_iter-1, max_iter-1], R

a, b = 0, 1
c, d = 0, 1

test_function = lambda x, y: np.sin(10*x) * np.cos(10*y) + np.exp(-5 * ((x-0.5)**2 + (y-0.5)**2)) + 0.5 * np.exp(-10 * ((x-0.2)**2 + (y-0.8)**2))

result, R_table = romberg_2d(test_function, a, b, c, d, max_iter=5)
print(f"Loka niðurstaða Romberg Heildunar: {result}")
