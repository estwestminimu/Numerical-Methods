import numpy as np

A1 = np.array([
    [5.8267103432, 1.0419816676, 0.4517861296, -0.2246976350, 0.7150286064],
    [1.0419816676, 5.8150823499, -0.8642832971, 0.6610711416, -0.3874139415],
    [0.4517861296, -0.8642832971, 1.5136472691, -0.8512078774, 0.6771688230],
    [-0.2246976350, 0.6610711416, -0.8512078774, 5.3014166511, 0.5228116055],
    [0.7150286064, -0.3874139415, 0.6771688230, 0.5228116055, 3.5431433879]
])

A2 = np.array([
    [5.4763986379, 1.6846933459, 0.3136661779, -1.0597154562, 0.0083249547],
    [1.6846933459, 4.6359087874, -0.6108766748, 2.1930659258, 0.9091647433],
    [0.3136661779, -0.6108766748, 1.4591897081, -1.1804364456, 0.3985316185],
    [-1.0597154562, 2.1930659258, -1.1804364456, 3.3110327980, -1.1617171573],
    [0.0083249547, 0.9091647433, 0.3985316185, -1.1617171573, 2.1174700695]
])
cond_A1 = np.linalg.cond(A1)
cond_A2 = np.linalg.cond(A2)
print(f"A1: {cond_A1}")
print(f"A2: {cond_A2}")

b = np.array([-2.8634904630, -4.8216733374, -4.2958468309, -0.0877703331, -2.0223464006])

y1 = np.linalg.solve(A1, b)
y2 = np.linalg.solve(A2, b)

print("A1 * y = b:")
print(y1)

print("\nA2 * y = b:")
print(y2)


np.random.seed(112)
n_vector_length = 5

#10^-6
target_norm = 1e-6

#Wygeneruj losowy wektor o normie 1
vector = np.random.randn(n_vector_length)
#skalujemy wektor do 1
vector /= np.linalg.norm(vector)

#Skalujemy wketor
vector *= target_norm


#Rozwiązanie równań z zaburzonym wektorem b
y1_unsettled = np.linalg.solve(A1, vector)
y2_unsettled = np.linalg.solve(A2, vector)

#Różnica
delta_y1 = np.linalg.norm(y1 - y1_unsettled)
delta_y2 = np.linalg.norm(y2 - y2_unsettled)

print("\nA1 * y = (b + Δb):")
print(y1_unsettled)
print(f"Różnica między rozwiązaniami A1: {delta_y1}")

print("\nA2 * y = (b + Δb):")
print(y2_unsettled)
print(f"Różnica między rozwiązaniami A2: {delta_y2}")
