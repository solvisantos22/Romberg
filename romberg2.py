import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import dblquad
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def exact_integral(f):
    """
    Reiknar nákvæmt heildi falls
    """
    result, _ = dblquad(f, 0, 1, lambda y: 0, lambda y: 1)
    return result

def midpoint_triangle(f, tri):
    """
    Nálgar heildið yfir þríhyrning með því að nota miðpunktsreglu
    """
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]

    xm = (x1 + x2 + x3) / 3
    ym = (y1 + y2 + y3) / 3

    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    return f(xm, ym) * area, (xm, ym)

def subdivide_triangle(tri):
    """
    Skiptir þríhyrningi í 4 minni
    """
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]

    m12 = ((x1 + x2) / 2, (y1 + y2) / 2)
    m23 = ((x2 + x3) / 2, (y2 + y3) / 2)
    m31 = ((x3 + x1) / 2, (y3 + y1) / 2)

    return [
        [tri[0], m12, m31],
        [m12, tri[1], m23],
        [m31, m23, tri[2]],
        [m12, m23, m31]
    ]

def plot_triangles(triangles1, triangles2, iter_num):
    """
    Plottar í 2d
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for tri in triangles1:
        polygon = plt.Polygon(tri, edgecolor='blue', fill=False, linewidth=1)
        ax.add_patch(polygon)

    for tri in triangles2:
        polygon = plt.Polygon(tri, edgecolor='red', fill=False, linewidth=1)
        ax.add_patch(polygon)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Skipting í ítrun {iter_num}")
    ax.set_aspect('equal')

    plt.savefig(os.path.join(output_dir, f'triangle_division_iter_{iter_num}.png'))
    plt.close()

def plot_3d_triangles(f, triangles1, triangles2, iter_num):
    """
    Plottar fallið
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for color, triangles in zip(['blue', 'red'], [triangles1, triangles2]):
        for tri in triangles:
            x = [p[0] for p in tri] + [tri[0][0]]
            y = [p[1] for p in tri] + [tri[0][1]]
            z = [f(p[0], p[1]) for p in tri] + [f(tri[0][0], tri[0][1])]

            verts = [list(zip(x, y, z))]
            poly = Poly3DCollection(verts, alpha=0.5, edgecolor=color, linewidth=0.5)
            ax.add_collection3d(poly)

    ax.set_title(f"3D framsetning fyrir n={iter_num}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("f(x,y)")

    plt.savefig(os.path.join(output_dir, f'triangle_3d_iter_{iter_num}.png'))
    plt.close()

def romberg_triangles(f, max_iter=5):
    """
    Framkvæmir Romberg heildun yfir [0,1]x[0,1]
    """
    tri1 = [(0, 0), (1, 0), (0, 1)]
    tri2 = [(1, 0), (1, 1), (0, 1)]

    R1 = np.zeros((max_iter, max_iter))
    R2 = np.zeros((max_iter, max_iter))

    exact_value = exact_integral(f)

    print(f"Rétt gildi: {exact_value:.6f}")
    print("Ítrun     | Nálgun        | Skekkja")
    print("------------------------------------------")
    triangles1 = [tri1]
    triangles2 = [tri2]

    for i in range(max_iter):
        R1[i, 0] = sum(midpoint_triangle(f, t)[0] for t in triangles1)
        R2[i, 0] = sum(midpoint_triangle(f, t)[0] for t in triangles2)

        plot_triangles(triangles1, triangles2, iter_num=i)
        plot_3d_triangles(f, triangles1, triangles2, iter_num=i)

        triangles1 = [sub for t in triangles1 for sub in subdivide_triangle(t)]
        triangles2 = [sub for t in triangles2 for sub in subdivide_triangle(t)]

        for j in range(1, i + 1):
            R1[i, j] = (4**j * R1[i, j-1] - R1[i-1, j-1]) / (4**j - 1)
            R2[i, j] = (4**j * R2[i, j-1] - R2[i-1, j-1]) / (4**j - 1)

        best_R1 = R1[i, i] if i > 0 else R1[i, 0]
        best_R2 = R2[i, i] if i > 0 else R2[i, 0]
        total_approx = best_R1 + best_R2
        error = abs(total_approx - exact_value)
        print(f"{i+1:<9} | {total_approx:<13.6f} | {error:.6e}")

    final_result = R1[max_iter-1, max_iter-1] + R2[max_iter-1, max_iter-1]

    print("\nLoka niðurstaða Romberg Heildunar (þríhyrninga): {:.6f}".format(final_result))
    return final_result
test_function = lambda x, y: x**2+y**2
romberg_triangles(test_function, max_iter=5)
