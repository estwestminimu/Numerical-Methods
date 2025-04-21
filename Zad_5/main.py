import numpy as np
import matplotlib.pyplot as plt


def jacobi(N, d, quit, dif, vec, exact):
    g = [i + 1 for i in range(N)]
    h0 = 0.5 / d
    h1 = 0.1 / d
    x = [vec] * N
    x_new = x[:]
    stops = []

    for _ in range(quit):
        for i in range(N):
            sum1 = 0
            if i - 1 >= 0:
                sum1 += h0 * x[i - 1]
            if i + 1 < N:
                sum1 += h0 * x[i + 1]
            if i - 2 >= 0:
                sum1 += h1 * x[i - 2]
            if i + 2 < N:
                sum1 += h1 * x[i + 2]

            x_new[i] = g[i] - sum1

        stop = max(abs(x_new[i] - exact[i]) for i in range(N))
        stops.append(stop)

        x = x_new[:]

        if stop < dif:
            break

    return x_new, stops


def gauss(N, d, quit, dif, vec, exact):
    g = [i + 1 for i in range(N)]
    h0 = 0.5 / d
    h1 = 0.1 / d
    x = [vec] * N
    stops = []

    for _ in range(quit):
        x_old = x[:]
        for i in range(N):
            sum1 = 0
            sum2 = 0

            if i - 1 >= 0:
                sum1 = h0 * x[i - 1]
            if i - 2 >= 0:
                sum1 += h1 * x[i - 2]
            if i + 1 < N:
                sum2 = h0 * x_old[i + 1]
            if i + 2 < N:
                sum2 += h1 * x_old[i + 2]

            x[i] = g[i] - sum1 - sum2

        stop = max(abs(x[i] - exact[i]) for i in range(N))
        stops.append(stop)

        if stop < dif:
            break

    return x, stops


def numpy_f(N, d):
    h0 = 0.5 / d
    h1 = 0.1 / d
    A = np.zeros((N, N))
    b = np.array([i + 1 for i in range(N)])


    for i in range(N):
        if i - 1 >= 0:
            A[i, i - 1] = h0
        if i + 1 < N:
            A[i, i + 1] = h0
        if i - 2 >= 0:
            A[i, i - 2] = h1
        if i + 2 < N:
            A[i, i + 2] = h1
        A[i, i] = 1

    x_exact = np.linalg.solve(A, b)
    return x_exact


N = 200
d_arr = [1.1, 1.2,1.5, 2,3]
vec = 0

plt.figure(figsize=(12, 8))

for d in d_arr:
    exact = numpy_f(N, d)

    tmp_jacobi, jacobi_stops = jacobi(N, d, 100, 1e-10, vec, exact)
    tmp_gauss, gs_stops = gauss(N, d, 100, 1e-10, vec, exact)

    print(f"d={d}:")
    print(f"Numpy: {exact[:5]}")
    print(f"Jacobi : {tmp_jacobi[:5]}")
    print(f"Gauss-Seidel: {tmp_gauss[:5]}")
    plt.semilogy(range(len(jacobi_stops)), jacobi_stops, label=f'd={d}, Jacobi', linestyle='-')
    plt.semilogy(range(len(gs_stops)), gs_stops, label=f'd={d}, Gauss-Seidel', linestyle=':')

plt.xlabel('Numer iteracji')
plt.ylabel('Różnica wektora danej iteracji i wartości dokładnej')
plt.legend()
plt.grid(True)
plt.show()
