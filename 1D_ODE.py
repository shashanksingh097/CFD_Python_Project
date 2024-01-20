"""
Solve a differental equation with dT/dX = X  with boundary condition T(X=0) = 0  in the domain X=0 to X=10.

Here FDM is used to solve the ODE and the result is compared with the analytical solution.
"""
import numpy as np
import matplotlib.pyplot as plt

# Variable initialization.
N = 5 # number of Nodes
domain_length = 10
T1 = 0

X = np.linspace(0, domain_length, N)
dx = domain_length / (N - 1)

T = np.zeros(N)
coefficient_matrix = np.zeros((N, N))

for i in range(1, N - 1):
    coefficient_matrix[i, i - 1:i + 2] = [-1, 0, 1]

coefficient_matrix[0, 0] = 1
# For AX=B
B = np.zeros(N)
B[0] = T1
B[1:N - 1] = 2 * dx * X[1:N - 1]

# This can be changed for the last point
is_backward_second = 1

if is_backward_second:

    # second order backward, N should be at least 4
    coefficient_matrix[N - 1, N - 3:] = [1, -4, 3]
    B[N - 1] = 2 * dx * X[N - 1]
else:
    coefficient_matrix[N - 1, -2:] = [-1,1]
    B[N - 1] =  dx * X[N - 1]

T = np.linalg.solve(coefficient_matrix, B)

# Plotting
plt.figure(figsize=(8, 6))

# Plot the numerical solution
plt.scatter(X, T, label='Numerical Solution', color='blue', marker='o')

# Plot the analytical solution
plt.plot(X, X**2 / 2, label='Analytical Solution', color='red', linestyle='--')

plt.title('Comparison of Numerical and Analytical Solutions')
plt.xlabel('X')
plt.ylabel('T')
plt.legend()
plt.grid(True)
plt.show()
