import numpy as np
import matplotlib.pyplot as plt
from .Bezier import CompositeBezierCurve, BezierCurve


class HermiteInterpolant(CompositeBezierCurve):
    def __init__(self, data_values, t_values=None, order_of_continuity=2, mu=1):
        """
        :param data_values: data values to interpolate
        :param t_values: parameter values, if None, computed
        :param order_of_continuity: the analytical continuity of the interpolation
        """
        self.x = np.array(data_values, dtype=float)
        self.n = len(data_values)
        self.C = order_of_continuity

        if t_values is None:
            self._compute_t_values(mu=mu)
        else:
            self.t = t_values

        self._compute_derivatives()
        self._compute_composite_curve()

    def _compute_t_values(self, mu=1):
        """
        implements three forms:
        mu = 0: uniform
        mu = 0.5: centripetal parametrization
        mu = 1: chordal parametrization
        """
        x = self.x
        t = [0]
        for i in range(len(x) - 1):
            t.append(t[-1] + np.linalg.norm(x[i+1] - x[i])**mu)

        self.t = t

    def _compute_composite_curve(self):
        n = self.n
        x = self.x
        t = self.t
        m = self.m
        curves = []
        knots = [t[0]]
        for i in range(n-1):
            h = t[i+1] - t[i]
            c0 = x[i]
            c1 = x[i] + h * m[i] / 3
            c2 = x[i+1] - h * m[i+1] / 3
            c3 = x[i+1]
            curve = BezierCurve([c0, c1, c2, c3], t[i], t[i+1])
            curves.append(curve)
            knots.append(t[i+1])
        self.c = CompositeBezierCurve(curves, knots)

    def _compute_derivatives(self):
        k = self.C
        x = self.x
        t = self.t
        n = self.n

        m = np.zeros_like(x)

        # boundaries
        m[0] = (x[1] - x[0]) / float(t[1] - t[0])
        m[-1] = (x[-1] - x[-2]) / float(t[-1] - t[-2])

        # fill h
        h = np.zeros(n) # difference in time
        for i in range(n-1):
            h[i+1] = t[i+1] - t[i]

        # inner points
        if k == 1:
            for i in range(1, n-1):
                m[i] = (x[i+1] - x[i]) / h[i]


        elif k == 2:

            beta = [2*(h[i-1] + h[i]) for i in range(1, n-1)]
            gamma = [h[i-1] for i in range(1, n-2)]
            alpha = [h[i] for i in range(2, n-1)]

            # construct tridiagonal matrix and rhs vector b
            A = np.diag(gamma, 1) + np.diag(beta) + np.diag(alpha, -1)
            b = np.array([3 * ((h[i] / h[i-1]) * (x[i] - x[i-1]) + (h[i-1] / h[i]) * (x[i+1] - x[i])) for i in range(1, n-1)])
            b[0] -= h[1]*m[0]
            b[-1] -= h[-2]*m[-1]
            m = np.concatenate((m[:1], np.linalg.solve(A, b), m[-1:]))

        else:
            raise NotImplementedError('Higher continuity than C2 not implemented')

        self.m = m

    def plot(self, n=100, data_points=True, control_polygon=True, display=True, fig=None):
        fig = self.c.plot(n = n, display=False, control_polygon=control_polygon)
        ax = fig.gca()

        if data_points:
            ax.scatter(*zip(*self.x))

        if display:
            plt.show()

        return fig

if __name__ == '__main__':
    def hermite_demo():
        interpol_points = [(2, 3, 0), (5, 3, 1), (0, 0, 1), (3, 2, 2), (2, 1, 3)]
        f = HermiteInterpolant(data_values=interpol_points, order_of_continuity=1)
        f.plot(n=1000, control_polygon=True)

    hermite_demo()
