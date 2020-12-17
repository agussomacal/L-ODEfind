from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.integrate import odeint


class DifferentialModels:
    def __init__(self, var_names):
        self.var_names = var_names
        self.d_to_coeff = defaultdict(lambda: defaultdict(lambda: 0))

    def odespy_func(self):
        def f(u, t):
            return self.get_dt(u)

        return f

    def get_dt(self, X):
        # TODO: this can be generalized
        return X

    def coeff(self, der: str, term: str) -> float:
        return self.d_to_coeff[der][term]


# ============================================================= #
#                       oscilating 2d
# ============================================================= #
class Oscilator(DifferentialModels):
    def __init__(self, a, b, c, d):
        DifferentialModels.__init__(self, var_names=["x", "y"])

        self.d_to_coeff['dx']['x'] = a
        self.d_to_coeff['dx']['y'] = b
        self.d_to_coeff['dy']['x'] = c
        self.d_to_coeff['dy']['y'] = d

    def get_dt(self, X):
        return np.matmul(
            np.array([[self.d_to_coeff['dx']['x'], self.d_to_coeff['dx']['y']],
                      [self.d_to_coeff['dy']['x'], self.d_to_coeff['dy']['y']]]), X)


# ============================================================= #
#                       Lorenz attractor
# ============================================================= #
class LorenzAttractor(DifferentialModels):
    def __init__(self, sigma=10, rho=28, beta=8.0 / 3):
        DifferentialModels.__init__(self, var_names=["x", "y", "z"])

        self.d_to_coeff['dx']['x'] = -sigma
        self.d_to_coeff['dx']['y'] = sigma

        self.d_to_coeff['dy']['x'] = rho
        self.d_to_coeff['dy']['xz'] = -1
        self.d_to_coeff['dy']['y'] = -1

        self.d_to_coeff['dz']['xy'] = 1
        self.d_to_coeff['dz']['z'] = -beta

    def get_dt(self, X):
        # return np.array((self.sigma * (X[1] - X[0]),
        #                  X[0] * (self.rho - X[2]) - X[1],
        #                  X[0] * X[1] - self.beta * X[2]))
        return np.array([
            self.d_to_coeff['dx']['x'] * X[0] + self.d_to_coeff['dx']['y'] * X[1],
            self.d_to_coeff['dy']['x'] * X[0] + self.d_to_coeff['dy']['xz'] * X[0] * X[2] +
            self.d_to_coeff['dy']['y'] * X[1],
            self.d_to_coeff['dz']['xy'] * X[0] * X[1] + self.d_to_coeff['dz']['z'] * X[2]
        ])


# ============================================================= #
#                       Roseler attractor
# ============================================================= #
class RoselerAttractor(DifferentialModels):
    def __init__(self, a=0.52, b=2.0, c=4.0):
        DifferentialModels.__init__(self, var_names=["x", "y", "z"])
        self.d_to_coeff['dx']['y'] = -1
        self.d_to_coeff['dx']['z'] = -1

        self.d_to_coeff['dy']['x'] = 1
        self.d_to_coeff['dy']['y'] = a

        self.d_to_coeff['dz']['cte'] = b  # TODO: correct name of constant terms
        self.d_to_coeff['dz']['xz'] = 1
        self.d_to_coeff['dz']['z'] = -c

    def get_dt(self, X):
        return np.array((
            self.d_to_coeff['dx']['y'] * X[1] + self.d_to_coeff['dx']['z'] * X[2],
            self.d_to_coeff['dy']['x'] * X[0] + self.d_to_coeff['dy']['y'] * X[1],
            self.d_to_coeff['dz']['cte'] + self.d_to_coeff['dz']['xz'] * X[2] * X[0] + self.d_to_coeff['dz']['z'] * X[2]
        ))


# ============================================================= #
#                       Van der Pol attractor
# ============================================================= #

class VanDerPolAttractor(DifferentialModels):
    def __init__(self, mu=0.01):
        DifferentialModels.__init__(self, var_names=["x", "y"])
        self.mu = mu
        self.d_to_coeff['dx']['y'] = 1

        self.d_to_coeff['dy']['x'] = -1
        self.d_to_coeff['dy']['y'] = mu
        self.d_to_coeff['dy']['xxy'] = -mu

    def get_dt(self, X):
        return np.array((
            self.d_to_coeff['dx']['y'] * X[1],
            self.d_to_coeff['dy']['y'] * X[1] + self.d_to_coeff['dy']['xxy'] * X[0] ** 2 * X[1] +
            self.d_to_coeff['dy']['x'] * X[0]
        ))


# ============================================================= #
#                       Lorenz attractor X
# ============================================================= #
class LorenzXAttractor(DifferentialModels):
    def __init__(self, sigma=10, rho=28, beta=8.0 / 3):
        DifferentialModels.__init__(self, var_names=["X(t)", "dX(t)"])
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def get_dt(self, X):
        return np.array((
            X[1],
            self.sigma * X[0] / 2 * (self.rho + self.sigma * self.rho - (1 + self.sigma)) - X[1] / 2 * (
                    self.sigma + 1) ** 2
        ))


# ============================================================= #
#                       Pendulus
# ============================================================= #
class Pendulus(DifferentialModels):
    def __init__(self, c=10):
        DifferentialModels.__init__(self, var_names=["Omega", "Theta"])
        self.c = c

    def get_dt(self, X):
        theta, omega = X
        return [omega, -self.c * np.sin(theta)]


# ============================================================= #
#                       Oscillator
# ============================================================= #
class StressedString(DifferentialModels):
    def __init__(self, L, k, m, A, g=10):
        DifferentialModels.__init__(self, var_names=['u'])
        self.L = L
        self.k = k
        self.m = m
        self.A = A
        self.g = g

    def get_dt(self, X):
        return


# ============================================================= #
#            Eq_diff Integrator for simulations
# ============================================================= #
class Integrator:
    """
    Integrador de ecuaciones diferenciales
    """

    def __init__(self, model):
        self.model = model

    def integrate_solver(self, Xinit, time_steps, integration_dt):
        def ode_func(x, t):
            return self.model.get_dt(x)

        time = np.round(np.linspace(0, (time_steps - 1) * integration_dt, time_steps),
                        int(np.ceil(-np.log10(integration_dt))))
        u = odeint(ode_func, Xinit, t=time)
        sol = pd.DataFrame(u, columns=self.model.var_names, index=time)
        sol.index.name = 't'
        return sol
