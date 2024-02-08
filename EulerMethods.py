import numpy as np

class EulerMethods:

    def __init__(self, func, y0, t0):
        self.func = func
        self.y0 = y0
        self.t0 = t0

    def euler(self, dt, n_steps):
        times = np.linspace(self.t0, self.t0 + dt * n_steps, n_steps + 1)
        # Input of np.empty combines a tuple for time steps (rows) and number of variables (cols) 
        values = np.empty((n_steps + 1,) + np.shape(self.y0))
        values[0] = self.y0  # Store initial conditions

        y_temp = self.y0
        for i in range(1, n_steps + 1):
            # y_{n+1} = y_n + dt * f(t_n, y_n)
            y_temp = y_temp + dt * self.func(times[i-1], y_temp)
            
            # Update values with y_temp
            values[i] = y_temp

        return times, values
        
