from EulerMethods import EulerMethods
from IntegralSolver import IntegralSolver
import numpy as np
import matplotlib.pyplot as plt

def my_function(x):
    return np.sin(x)

def system_ode(t, y):
    # System parameters directly included
    a, b, c, d = 0.1, 0.2, 0.3, 0.1
    dx_dt = a * y[0] - b * y[1]
    dy_dt = c * y[0] + d * y[1]
    return np.array([dx_dt, dy_dt])

if __name__ == "__main__":
    print('Integration Methods')
    
    integrator = IntegralSolver(my_function)
    print(integrator.trapezoidal_rule(0, np.pi, 100))
    print(integrator.simpsons_rule(0, np.pi, 100))
    print(integrator.gaussian_quadrature(0, np.pi, 10))


    print("Euler method:")
    solver = EulerMethods(system_ode, [10, 5], 0)
    dt = 0.01
    n_steps = 1000
    times, values = solver.euler(dt=dt, n_steps=n_steps)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(times, values[:, 0], label="x(t)")
    plt.plot(times, values[:, 1], label="y(t)")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Populations Over Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(values[:, 0], values[:, 1], label="Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Plot: y vs. x")
    plt.legend()

    plt.tight_layout()
    plt.show()
