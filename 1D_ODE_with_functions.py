import numpy as np
import matplotlib.pyplot as plt

def solve_differential_equation(N, domain_length, T1, is_backward=True):
    X = np.linspace(0, domain_length, N)
    dx = domain_length / (N - 1)

    coefficient_matrix = np.zeros((N, N))

    for i in range(1, N - 1):
        coefficient_matrix[i, i - 1:i + 2] = [-1, 0, 1]

    coefficient_matrix[0, 0] = 1

    B = np.zeros(N)
    B[0] = T1
    B[1:N - 1] = 2 * dx * X[1:N - 1]

    if is_backward:
        # Second order backward, N should be at least 4
        coefficient_matrix[N - 1, N - 3:] = [1, -4, 3]
        B[N - 1] = 2 * dx * X[N - 1]
    else:
        coefficient_matrix[N - 1, -2:] = [-1, 1]
        B[N - 1] = dx * X[N - 1]

    T = np.linalg.solve(coefficient_matrix, B)
    return X, T

def plot_results(X, T_numerical, T_analytical):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, T_numerical, label='Numerical Solution', color='blue', marker='o')
    plt.plot(X, T_analytical, label='Analytical Solution', color='red', linestyle='--')
    plt.title('Comparison of Numerical and Analytical Solutions')
    plt.xlabel('X')
    plt.ylabel('T')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage
N = 10
domain_length = 10
T1 = 0

X, T_numerical = solve_differential_equation(N, domain_length, T1)
T_analytical = X**2 / 2

plot_results(X, T_numerical, T_analytical)
