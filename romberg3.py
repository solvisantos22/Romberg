import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import dblquad

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Exact integral for the same region [0,1] x [0,1]
def exact_integral():
    result, _ = dblquad(lambda x, y: x**2 + y**2, 0, 1, lambda y: 0, lambda y: 1)
    return result

#####################
### TRAPEZOIDAL RULE
#####################
def trapezoidal_2d(f, a, b, c, d, n):
    x = np.linspace(a, b, n+1)
    y = np.linspace(c, d, n+1)
    hx = (b - a) / n
    hy = (d - c) / n

    integral = 0
    for i in range(n):
        for j in range(n):
            integral += 0.25 * (f(x[i], y[j]) + f(x[i+1], y[j]) + f(x[i], y[j+1]) + f(x[i+1], y[j+1])) * hx * hy
    return integral

def romberg_2d(f, a, b, c, d, max_iter=5):
    R = np.zeros((max_iter, max_iter))
    exact_value = exact_integral()
    errors_trapezoidal = []

    for i in range(max_iter):
        n = 2**i
        R[i, 0] = trapezoidal_2d(f, a, b, c, d, n)
        errors_trapezoidal.append(abs(R[i, 0] - exact_value))

        for j in range(1, i + 1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[max_iter-1, max_iter-1], errors_trapezoidal

#####################
### TRIANGLE METHOD
#####################
def midpoint_triangle(f, tri):
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]
    xm = (x1 + x2 + x3) / 3
    ym = (y1 + y2 + y3) / 3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return f(xm, ym) * area

def subdivide_triangle(tri):
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

def romberg_triangles(f, max_iter=5):
    tri1 = [(0, 0), (1, 0), (0, 1)]
    tri2 = [(1, 0), (1, 1), (0, 1)]

    R1 = np.zeros((max_iter, max_iter))
    R2 = np.zeros((max_iter, max_iter))
    exact_value = exact_integral()
    errors_triangles = []

    triangles1 = [tri1]
    triangles2 = [tri2]

    for i in range(max_iter):
        R1[i, 0] = sum(midpoint_triangle(f, t) for t in triangles1)
        R2[i, 0] = sum(midpoint_triangle(f, t) for t in triangles2)

        total_approx = R1[i, 0] + R2[i, 0]
        errors_triangles.append(abs(total_approx - exact_value))

        triangles1 = [sub for t in triangles1 for sub in subdivide_triangle(t)]
        triangles2 = [sub for t in triangles2 for sub in subdivide_triangle(t)]

        for j in range(1, i + 1):
            R1[i, j] = (4**j * R1[i, j-1] - R1[i-1, j-1]) / (4**j - 1)
            R2[i, j] = (4**j * R2[i, j-1] - R2[i-1, j-1]) / (4**j - 1)

    return R1[max_iter-1, max_iter-1] + R2[max_iter-1, max_iter-1], errors_triangles

##########################
### FUNCTION TO INTEGRATE
##########################
def test_function(x, y):
    return x**2 + y**2

################################
### RUN INTEGRATION & PLOT ERROR
################################
a, b = 0, 1
c, d = 0, 1
max_iter = 10

result_trapezoidal, errors_trapezoidal = romberg_2d(test_function, a, b, c, d, max_iter)
result_triangles, errors_triangles = romberg_triangles(test_function, max_iter)

# Plot error convergence
plt.figure(figsize=(8, 6))
iterations = np.arange(1, max_iter+1)
plt.plot(iterations, errors_trapezoidal, marker='o', linestyle='-', label="Trapezoidal Rule", color="blue")
plt.plot(iterations, errors_triangles, marker='s', linestyle='--', label="Triangle-Based", color="red")

plt.yscale("log")  # Log scale to better show convergence
plt.xlabel("Iteration")
plt.ylabel("Absolute Error")
plt.title("Error Convergence of Romberg Integration Methods")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig(os.path.join(output_dir, "error_convergence.png"))
plt.show()

print(f"\nFinal Romberg Integration Result (Trapezoidal Rule): {result_trapezoidal:.6f}")
print(f"Final Romberg Integration Result (Triangles): {result_triangles:.6f}")
