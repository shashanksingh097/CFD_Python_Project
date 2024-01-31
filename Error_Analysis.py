import numpy as np
import matplotlib.pyplot as plt


def diffex(func,dfunc,x,n):
    dftrue = dfunc(x)
    h = 1
    H = np.zeros(n)
    D = np.zeros(n)
    E = np.zeros(n)
    H[0] = h
    D[0] = (func(x+h)- func(x-h))/(2*h)
    E[0] = abs(dftrue - D[0])
    for i in range(1,n):
        h = h/10
        H[i] = h
        D[i] = (func(x+h) - func(x-h))/(2*h)
        E[i] = abs(dftrue - D[i])
    return H,D,E

ff = lambda x: -0.1*x**4 - 0.15*x**3 - 0.5*x**2 - 0.25*x + 1.2
df = lambda x: -0.4*x**3 - 0.45*x**2 - x - 0.25

H,D,E = diffex(ff,df,0.5,11)
print(' step size finite difference true error')
for i in range(11):
   print('{0:14.10f} {1:16.14f} {2:16.13f}'.format(H[i],D[i],E[i]))

# Plotting code
plt.figure(figsize=(15,10))
plt.loglog(H, E, marker='o', markerfacecolor='red', color='blue', linestyle='dotted', linewidth=2) # log-log plot
plt.xlabel('Step size (H)')
plt.ylabel('Error (E)')
plt.title('Error (E) vs Step size (H)')

plt.xticks(H)

# Annotate each point with its H and E values

for (h, e) in zip(H, E):
    plt.annotate(f'H={h:.2e}, E={e:.2e}', (h, e))

plt.grid(True)
plt.show()