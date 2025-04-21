import time
import matplotlib.pyplot as plt
import numpy as np

def solve(N):
    matrixA = []
    for i in range(N):
        row = [0, 1.01, 0, 0]
        if i > 0:
            row[0] = 0.3
            row[2] = 0.2 / i
        if i > 1:
            row[3] = 0.15 / (i ** 3)

        matrixA.append(row)


    for i in range(1, N):
        matrixA[i][0] = matrixA[i][0] / matrixA[i - 1][1]  
        matrixA[i][1] = matrixA[i][1] - matrixA[i][0] * matrixA[i - 1][2]
        if i < N - 1:
            matrixA[i][2] = matrixA[i][2] - matrixA[i][0] * matrixA[i - 1][3]


    x = [0] * N
    for i in range(N):
        x[i] = (i + 1 - (matrixA[i][0] * x[i - 1] if i > 0 else 0)) / matrixA[i][1]


    determinant = 1
    for i in range(N):
        determinant *= matrixA[i][1]

    return x, determinant

def solve_numpy(N):
    matrixA = np.zeros((N, N))
    for i in range(N):
        matrixA[i][i] = 1.01
        if i > 0:
            matrixA[i][i - 1] = 0.3
        if i > 1:
            matrixA[i][i - 2] = 0.2 / i
        if i > 2:
            matrixA[i][i - 3] = 0.15 / (i ** 3)

    determinant = np.linalg.det(matrixA)
    inverse_matrix = np.linalg.inv(matrixA)

    return inverse_matrix, determinant

def measure_time(func, N):
    total_time = 0
    for _ in range(10):
        start = time.time()
        func(N)
        stop = time.time()
        total_time += (stop - start)
    return total_time / 10


times_original = []
times_numpy = []
N_values = list(range(10, 1000, 50))

for N in N_values:
    avg_time_original = measure_time(solve, N)
    avg_time_numpy = measure_time(solve_numpy, N)
    times_original.append(avg_time_original * 1000)
    times_numpy.append(avg_time_numpy * 1000)


result, determinant = solve(300)
print("Rozwiązanie wektora:", result)
print("Wyznacznik macierzy:", determinant)




plt.figure(figsize=(10, 6))
plt.scatter(N_values, times_original, color='red', label='Własna funkcja')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas (milisekund)')
plt.title("Własna funkcja")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(N_values, times_numpy, color='blue', label='NumPy')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas (milisekund)')
plt.title("NumPy")
plt.legend()
plt.grid()
plt.show()
