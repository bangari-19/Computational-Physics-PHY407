import numpy as np
from gaussxw import gaussxwab

class IntegralSolver:

    def __init__(self, integrand):
        self.integrand = integrand

    def trapezoidal_rule(self, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.integrand(x)

        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    def simpsons_rule(self, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = self.integrand(x)

        # np.sum(y[1:-1:2]) is even sums
        # np.sum(y[2:-2:2]) is odd sums
        
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

    def gaussian_quadrature(self, a, b, n):
        x, w = gaussxwab(n, a, b) # w is the weights
        return np.sum(w * self.integrand(x))
