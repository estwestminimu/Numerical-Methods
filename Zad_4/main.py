import numpy as np
import matplotlib.pyplot as plt
import time


def sherman_morrison(matrix, tmp, size):

    diagA, diagB = matrix
    forward = [0] * size
    backward = [0] * size

    backward[-1] = 1 / diagA[-1]
    for i in range(size - 2, -1, -1):
        backward[i] = (1 - diagB[i] * backward[i + 1]) / diagA[i]


    forward[-1] = tmp[-1] / diagA[-1]
    for i in range(size - 2, -1, -1):
        forward[i] = (tmp[i] - diagB[i] * forward[i + 1]) / diagA[i]


    det = sum(forward) / (1 + sum(backward))
    solution = [forward[i] - backward[i] * det for i in range(size)]

    return solution

def generate(N):
    diagA = [5] * N
    b = [2] * N
    diagB = [3] * (N - 1)
    matrix = [diagA, diagB + [0]]

    return matrix, b

def numpy_solve(N):
    b = np.array([2] * N)
    A = np.diag([5] * N) + np.diag([3] * (N - 1), 1) + np.ones((N, N))
    return np.linalg.solve(A, b)

def average(func, *args, mult=5):
    times = []
    for _ in range(mult):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    return sum(times) / len(times)

def plota():
    numpy = []
    sherman = []
    N_values = range(10, 5000, 100)


    for N in N_values:
        matrix, b = generate(N)

        time_sherman = average(sherman_morrison, matrix, b, N)
        sherman.append(time_sherman)

        time_numpy = average(numpy_solve, N)
        numpy.append(time_numpy)

    plt.plot(N_values, numpy, label="NumPy", linestyle='--', marker='x')
    plt.plot(N_values, sherman, label="Sherman-Morrison", linestyle='-', marker='o')
    plt.ylabel("Czas oblcizen (s)")
    plt.xlabel("Rozmiar macierzy")
    plt.legend()
    plt.grid()
    plt.show()

def plotb():
    sherman = []
    N_values = range(10, 5000, 100)

    for N in N_values:
        matrix, b = generate(N)

        time_sherman = average(sherman_morrison, matrix, b, N)
        sherman.append(time_sherman)

    plt.plot(N_values, sherman, label="Sherman-Morrison", color='blue', linestyle='-', marker='o')
    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas oblcizen (s)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    N = 120
    matrix, b = generate(N)
    result_numpy = numpy_solve(N)
    result_sherman = sherman_morrison(matrix, b, N)

    print("NumPy:")
    print(result_numpy)

    print("Sherman-Morrison:")
    print(result_sherman)

    plota()
    plotb()
