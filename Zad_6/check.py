import numpy as np

M = np.array([
    [9, 2, 0, 0],
    [2, 4, 1, 0],
    [0, 1, 3, 1],
    [0, 0, 1, 2]
])

values, vec = np.linalg.eig(M)

print("Wartości własne macierzy M:")
for i, j in enumerate(values, start=1):
    print(f"Wartość {i}: {j}")
