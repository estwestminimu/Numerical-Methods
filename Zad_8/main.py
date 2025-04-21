import numpy as np
import matplotlib.pyplot as plt




def dup_fuc(x, coefficients, func):
    return sum(a * phi(x) for a, phi in zip(coefficients, func))


def dane(n_points, coefficients, func, sigma):
    x = np.linspace(-1, 1, n_points)
    y = np.array([dup_fuc(x, coefficients, func) for x in x])
    n = np.random.normal(0, sigma, size=n_points)
    v = y + n
    return x, v, y


def approx(x, y, func):
    n_points = len(x)
    n_basis = len(func)

    A = np.zeros((n_points, n_basis))
    for i, x in enumerate(x):
        for j, phi in enumerate(func):
            A[i, j] = phi(x)

    AT = A.T
    ATA = AT @ A
    ATy = AT @ y
    coefficients = np.linalg.solve(ATA, ATy)
    return coefficients


def plot_results(x, y, y_exact, y_approx):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Zaburzone punkty", color="red", s=10)
    plt.plot(x, y_exact, label="Funkcja dokładna", color="blue")
    plt.plot(x, y_approx, label="Aproksymacja", color="green", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()



def phi_1(x):
    return np.exp(-5 * x) * np.sin(40 * x)

def phi_2(x):
    return np.log(x + 1.51)

def phi_3(x):
    return np.sin(np.cos(3 * x))

def phi_4(x):
    return np.tanh(2 * x)

def phi_5(x):
    return np.cos(np.exp(x) + 1)

def function():
    return [phi_1, phi_2, phi_3, phi_4, phi_5]



if __name__ == "__main__":
    coefficients_exact = [-0.55, 1.5, 2.0, -1.4, 1.0]
    funct = function()
    n_points = 200
    sigma = 7

    x_values, y_values, y_exact = dane(n_points, coefficients_exact, funct, sigma)

    coef_aprox = approx(x_values, y_values, funct)
    y_approx = np.array([dup_fuc(x, coef_aprox, funct) for x in x_values])

    print("Dokłade:", coefficients_exact)
    print("Aproksymacja:", coef_aprox)

    plot_results(x_values, y_values, y_exact, y_approx )
