import math
import matplotlib.pyplot as plt
import numpy as np

M = [
    [9, 2, 0, 0],
    [2, 4, 1, 0],
    [0, 1, 3, 1],
    [0, 0, 1, 2]
]


# norma euklidesowa
def norm(vector):
    return np.linalg.norm(vector)


# zmniejszamy wartość wektora tak, aby miał długość 1
def skalar(vector, scalar):
    return vector / scalar


def power(matrix, max_iter=100, tol=1e-10):
    n = len(matrix)
    y = np.ones(n)
    y = skalar(y, norm(y))
    tmp = []

    for _ in range(max_iter):
        z = np.dot(matrix, y)
        app_tmp = norm(z)
        tmp.append(app_tmp)
        y_new = skalar(z, app_tmp)

        if norm(y_new - y) < tol:
            break
        y = y_new

    return app_tmp, y, tmp



largest, max, tmp = power(M)
lambda_diff = [abs(l - largest) for l in tmp]

plt.figure(figsize=(8, 6))
plt.plot(range(len(lambda_diff)), [math.log10(d + 1e-15) for d in lambda_diff])
plt.xlabel("Iteracja")
plt.ylabel("Różnica wartości własnej danej iteracji")
plt.title("Metoda potęgowa")
plt.legend()
plt.grid()
plt.show()

print("Wartość własna:", largest)
print("Wektor własny:", max)
