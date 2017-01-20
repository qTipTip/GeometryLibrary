from .BasisFunctions import BernsteinPolynomial

import numpy as np
import matplotlib.pyplot as plt

from .Bezier import BezierCurve


class LeastSquaresApproximation(object):

    def __init__(self, x_values, z_values, n, basis='bernstein'):

        self.x = np.array(x_values, dtype=float)
        self.z = np.array(z_values, dtype=float)
        self.m = len(x_values)
        self.n = n

        self.a = x_values[0]
        self.b = x_values[-1]

        self._set_basis_functions(n, basis)
        self._compute_coefficients()

    def _set_basis_functions(self, n, basis):

        if basis == 'bernstein':
            d = n - 1 # degree one lower than number of control points
            self.basis = [BernsteinPolynomial(i=i, degree=d) for i in range(n)]
        else:
            raise NotImplementedError('Basis: %s not supported' % basis)

    def _compute_coefficients(self):
        m = self.m
        n = self.n
        x = self.x
        z = self.z

        B = np.ndarray((m, n))
        for i, row in enumerate(B):
            for j, b in enumerate(self.basis):
                B[i, j] = b(x[i])

        A = np.dot(B.transpose(), B)
        v = np.dot(B.transpose(), z)

        c = np.linalg.solve(A, v)
        self.c = c

    def __call__(self, x):

        terms = [self.basis[i](x)*self.c[i] for i in range(self.n)]

        return sum(terms)

    def evaluate(self, n=100):

        t_values = np.linspace(self.a, self.b, n)
        y_values = np.array([self(t) for t in t_values])

        return t_values, y_values

    def plot(self, n=100):
        t, y = self.evaluate(n=n)
        plt.plot(self.x, self.z)
        plt.plot(t, y)
        plt.show()

if __name__ == '__main__':
    def f(x):
        return np.cos(x)*x

    def lq_demo():

        x_values = np.array([2, 5, 8, 12, 13, 35, 50])
        z_values = f(x_values)

        for n in [1, 3, 5, 6, 8]:
            test = LeastSquaresApproximation(x_values=x_values, z_values=z_values, n=n)
            test.plot()
    lq_demo()
