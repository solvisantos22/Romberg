import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# Föll sem við ætlum að prófa
test_functions = {
    "sin_cos_exp": lambda x, y: np.sin(10*x) * np.cos(10*y) + np.exp(-5 * (x**2 + y**2)),
    "gaussian_peak": lambda x, y: np.exp(-10 * ((x - 0.3)**2 + (y - 0.7)**2)),
    "polynomial_exp": lambda x, y: (x**3 + y**4) * np.exp(-x - y),
    "sin_ripple": lambda x, y: np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y),
    "fractal_mix": lambda x, y: np.sin(10*x) * np.cos(10*y) + np.exp(-5 * ((x-0.5)**2 + (y-0.5)**2)) + 0.5 * np.exp(-10 * ((x-0.2)**2 + (y-0.8)**2)),
}

# Reikna nákvæmt gildi fyrir hvert fall
exact_integrals = {name: dblquad(f, 0, 1, lambda y: 0, lambda y: 1)[0] for name, f in test_functions.items()}

# Trapisuregla í 2D
def trapezoidal_2d(f, a, b, c, d, n):
    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, n+1)
    hx = (b - a) / n
    hy = (d - c) / n

    integral = 0
    for i in range(n):
        for j in range(n):
            integral += f(x[i], y[j]) * hx * hy

    return integral

# Romberg fyrir trapisu
def romberg_trapezoidal_2d(f, a, b, c, d, max_iter=5):
    R = np.zeros((max_iter, max_iter))
    for i in range(max_iter):
        n = 2**i  # Fjöldi skiptinga
        R[i, 0] = trapezoidal_2d(f, a, b, c, d, n)

        for j in range(1, i + 1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[max_iter-1, max_iter-1]

# Miðpunktsregla fyrir þríhyrninga
def midpoint_triangle(f, tri):
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]
    xm = (x1 + x2 + x3) / 3
    ym = (y1 + y2 + y3) / 3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return f(xm, ym) * area

# Skipta þríhyrningi í fjóra minni þríhyrninga
def subdivide_triangle(tri):
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]

    # Miðpunktar
    m12 = ((x1 + x2) / 2, (y1 + y2) / 2)
    m23 = ((x2 + x3) / 2, (y2 + y3) / 2)
    m31 = ((x3 + x1) / 2, (y3 + y1) / 2)

    # Skilgreina fjóra minni þríhyrninga
    return [
        [tri[0], m12, m31],
        [m12, tri[1], m23],
        [m31, m23, tri[2]],
        [m12, m23, m31]
    ]

# Romberg fyrir þríhyrninga
def romberg_triangles(f, max_iter=5):
    tri1 = [(0, 0), (1, 0), (0, 1)]
    tri2 = [(1, 0), (1, 1), (0, 1)]

    R1 = np.zeros((max_iter, max_iter))
    R2 = np.zeros((max_iter, max_iter))

    triangles1 = [tri1]
    triangles2 = [tri2]

    for i in range(max_iter):
        R1[i, 0] = sum(midpoint_triangle(f, t) for t in triangles1)
        R2[i, 0] = sum(midpoint_triangle(f, t) for t in triangles2)

        # Skipta þríhyrningum frekar niður
        triangles1 = [sub for t in triangles1 for sub in subdivide_triangle(t)]
        triangles2 = [sub for t in triangles2 for sub in subdivide_triangle(t)]

        for j in range(1, i + 1):
            R1[i, j] = (4**j * R1[i, j-1] - R1[i-1, j-1]) / (4**j - 1)
            R2[i, j] = (4**j * R2[i, j-1] - R2[i-1, j-1]) / (4**j - 1)

    return R1[max_iter-1, max_iter-1] + R2[max_iter-1, max_iter-1]

# Niðurstöður
results = []

for name, func in test_functions.items():
    exact = exact_integrals[name]

    # Mæla tíma og skekkjur
    start_time = time.time()
    trap_result = romberg_trapezoidal_2d(func, 0, 1, 0, 1, max_iter=5)
    trap_time = time.time() - start_time
    trap_error = abs(trap_result - exact)

    start_time = time.time()
    tri_result = romberg_triangles(func, max_iter=5)
    tri_time = time.time() - start_time
    tri_error = abs(tri_result - exact)

    results.append([name, exact, trap_result, trap_error, trap_time, tri_result, tri_error, tri_time])

# Búa til DataFrame
df_results = pd.DataFrame(results, columns=["Fall", "Nákvæmt heildi", "Trapisuregla", "Trapisu skekkja",
                                            "Trapisu keyrslutími", "Þríhyrningsaðferð", "Þríhyrnings skekkja", "Þríhyrnings keyrslutími"])

# Prenta niðurstöður í fallegu töfluformi
print("\nSamanburður á Romberg Heildun\n")
print(df_results.to_string(index=False))

# Plotta samanburð á skekkju
plt.figure(figsize=(8,5))
plt.bar(df_results["Fall"], df_results["Trapisu skekkja"], label="Trapisu skekkja", alpha=0.7)
plt.bar(df_results["Fall"], df_results["Þríhyrnings skekkja"], label="Þríhyrnings skekkja", alpha=0.7)
plt.xticks(rotation=45)
plt.yscale("log")  # Nota log-skala til að sjá betur mismuninn
plt.ylabel("Error")
plt.legend()
plt.title("Samanburður á skekkju")
plt.show()

# Plotta samanburð á reiknitíma
plt.figure(figsize=(8,5))
plt.bar(df_results["Fall"], df_results["Þríhyrnings keyrslutími"], label="Þríhyrnings keyrslutími", alpha=0.7)
plt.bar(df_results["Fall"], df_results["Trapisu keyrslutími"], label="Trapisu keyrslutími", alpha=0.7)
plt.xticks(rotation=45)
plt.ylabel("Time (seconds)")
plt.legend()
plt.title("Samanburður á reiknitíma")
plt.show()
