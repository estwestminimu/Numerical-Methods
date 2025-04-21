import numpy as np
import matplotlib.pyplot as plt

M = np.array([
    [9, 2, 0, 0],
    [2, 4, 1, 0],
    [0, 1, 3, 1],
    [0, 0, 1, 2]
])


def qr(M, iter=100, tolerancja=1e-13):
    A = np.array(M, dtype=float)
    tmp = []

    for _ in range(iter):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

        current = [A[i][i] for i in range(len(A))]
        tmp.append(current)

        if len(tmp) > 1:
            prev_diag = tmp[-2]
            diff = [abs(current[i] - prev_diag[i]) for i in range(len(A))]
            if all(d < tolerancja for d in diff):
                break

    return A, tmp

final_A, tmp = qr(M)

errors = []
for i in range(1, len(tmp)):
    diff = [abs(tmp[i][j] - tmp[i - 1][j]) for j in range(4)]
    errors.append([(d) for d in diff])

plt.figure(figsize=(10, 6))

for i in range(4):
    y_values = [errors[j][i] for j in range(len(errors))]
    x_values = [j + 1 for j in range(len(errors)) if y_values[j] is not None]
    y_values = [y for y in y_values if y is not None]
    plt.plot(x_values, y_values,  label=f"({i + 1}) Wartość własna")

plt.xlabel("Iteracja")
plt.ylabel("Różnica wartości własnej danej iteracji")
plt.title("Metoda QR")
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()

values = [final_A[i][i] for i in range(len(final_A))]
print("Wartości własne: ", end="")
print(", ".join(f"{value}" for value in values))
