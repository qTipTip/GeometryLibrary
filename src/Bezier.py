import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from src.BasisFunctions import BernsteinPolynomial


class BezierCurve(object):
    """
    contains methods and fields needed for representing, computing
    and visualizing Bezier curves
    """

    def __init__(self, control_points, a=0, b=1):
        """
        :param control_points: an array containing the control points of the curve in any number of dimensions
        :param a: the lower parameter limit
        :param b: the upper parameter limit
        """

        self.cp = np.array(control_points)
        self.a = a
        self.b = b

        self._set_degree_and_dimension()
        self._set_basis_functions()

    def _set_degree_and_dimension(self):
        """
        extracts the degree of the bezier curve as well
        as the dimension of the ambient space from the control points
        """
        try:
            self.d, self.D = self.cp.shape
            self.d -= 1
        except ValueError:
            self.d, self.D = len(self.cp) - 1, 1

    def _set_basis_functions(self):
        """
        initializes and stores the d+1 basis functions
        in a list
        """
        d = self.d
        self.basis = [BernsteinPolynomial(i=i, degree=d) for i in range(d + 1)]

    def __call__(self, t):
        """
        computes a point on the curve using DeCasteljau

        :param t: the parameter value in an arbitrary interval [a, b]
        :return: the value of the curve at point t
        """

        u = (t - self.a) / float(self.b - self.a)  # local parameter variable
        d = self.d
        dim = self.D

        previous_step = self.cp
        for r in range(1, d+1):
            temporary_step = np.zeros((d+1 - r, dim))
            for i in range(len(previous_step) - 1):
                temporary_step[i] = (1 - u) * previous_step[i] + u * previous_step[i + 1]
            previous_step = temporary_step

        return previous_step[0]

    def evaluate(self, n=100):
        """
        computes *n* points on the curve

        :param n: number of sampling points
        :return: t, y
        """
        a, b = self.a, self.b
        t_values = np.linspace(a, b, num=n)
        y_values = np.array([self(t) for t in t_values])

        return t_values, y_values

    def plot(self, n=100, control_polygon=True, display=True, fig=None):
        """
        :param n: resolution
        :param control_polygon: whether to plot the control polygon or not
        :param display: whether to call plt.show
        :fig: a figure handle
        :return: a figure handle
        """

        dim = self.D
        if dim > 3:
            raise NotImplementedError('Cannot plot curves if dimension higher than 3')

        # if no figure handle was given, create a new one
        # otherwise, get current axis
        if fig == None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        t, y = self.evaluate(n=n)

        if dim != 1:
            ax.plot(*zip(*y))
        else:
            ax.plot(t, y)

        if control_polygon and dim != 1:
            ax.scatter(*zip(*self.cp))
            ax.plot(*zip(*self.cp), alpha=0.3, color='grey')
        if display:
           plt.show()

        return fig

class CompositeBezierCurve(BezierCurve):

    def __init__(self, curve_segments, knots):
        self.cs = curve_segments
        self.k = knots

        self._set_control_points()
        self._set_degree_and_dimension()

    def _set_control_points(self):
        self.cp = np.concatenate([c.cp for c in self.cs])

    def _set_degree_and_dimension(self):
        try:
            self.D = self.cp.shape[1]
        except:
            self.D = 1

    def __call__(self, t):
        k = self.k

        for i in range(len(k)-1):
            if k[i] <= t < k[i+1]:
                break

        return self.cs[i](t)

    def evaluate(self, n=100):

        a, b = self.k[0], self.k[-1]
        t_values = np.linspace(a, b, num=n)
        y_values = [self(t) for t in t_values]

        return t_values, y_values

class BezierSurface(BezierCurve):
    """
    contains methods and fields needed for representing, computing
    and visualizing Bezier surfaces
    """

    def __init__(self, control_points, a=[0, 1], b=[0, 1]):
        super(BezierSurface, self).__init__(control_points, a, b)

    def _set_degree_and_dimension(self):
        """
        extracts the degree of the surface, i.e, how many basis functions
        in each direction, and the dimension of the ambient space
        """

        m, n, dim = self.cp.shape
        self.d = (m-1, n-1)
        self.D = dim

    def _set_basis_functions(self):
        """
        initializes and stores the basis functions
        in a list
        """
        m, n = self.d
        self.basis = [BernsteinPolynomial(i=i, degree=m) for i in range(m+1)], \
                     [BernsteinPolynomial(i=j, degree=n) for j in range(n+1)]


    def __call__(self, s, t):
        """
        computes the value of the Bezier surface at the point (s, t) using
        deCasteljau

        :param s: first parameter
        :param t: second parameter
        :return:  the value of the surface at point (s, t)
        """

        # local parameter values
        u = (s - self.a[0]) / float(self.a[1] - self.a[0])
        v = (t - self.b[0]) / float(self.b[1] - self.b[0])

        m, n = self.d
        m = m + 1
        n = n + 1
        dim = self.D
        previous_step = self.cp
        while m != 1:
            tmp = np.ndarray((m - 1, n, self.D))
            for j in range(n):
                for i in range(m - 1):
                    tmp[i, j] = u * previous_step[i, j] + (1 - u) * previous_step[i + 1, j]
            m = m - 1
            previous_step = tmp
        previous_step = previous_step[0, :]

        while n != 1:
            tmp = np.ndarray((n - 1, self.D))
            for i in range(n - 1):
                tmp[i] = v * previous_step[i] + (1 - v) * previous_step[i + 1]
            n = n - 1
            previous_step = tmp

        return previous_step[0]


    def evaluate(self, n=(100, 100)):
        """
        evaluates points on the surface
        for a resolution of (100, 100) by default
        :param n: resolution
        :return: x-values, y-values z-values
        """
        s_res, t_res = n
        s_values = np.linspace(self.a[0], self.a[1], num=s_res)
        t_values = np.linspace(self.b[0], self.b[1], num=t_res)

        y_values = np.zeros(shape=(s_res, t_res, self.D))
        for i, s in enumerate(s_values):
            for j, t in enumerate(t_values):
                y_values[i, j] = self(s, t)

        return y_values[:, :, 0], y_values[:, :, 1], y_values[:, :, 2]

    def _get_control_grid(self):
        """
        :return: two arrays compatible with mpl3d surfaceplot for plotting lines
        and a flat array of control points
        """
        rows = []
        m, n, d = self.cp.shape
        for row in self.cp:
            for i in range(len(row) - 1):
                x = [row[i][0], row[i + 1][0]]
                y = [row[i][1], row[i + 1][1]]
                z = [row[i][2], row[i + 1][2]]
                rows.append([x, y, z])

        cols = []
        for j in range(n):
            col = self.cp[:, j]
            for i in range(len(col)-1):

                x = [col[i][0], col[i + 1][0]]
                y = [col[i][1], col[i + 1][1]]
                z = [col[i][2], col[i + 1][2]]
                cols.append([x, y, z])

        points = []
        for i in self.cp:
            for j in i:
                points.append(j)

        return np.array(rows), np.array(cols), np.array(points)

    def _plot_control_grid(self, ax):
        rows, cols, points = self._get_control_grid()
        for r in rows:
            ax.plot(*r, color='grey', alpha=0.4)

        for c in cols:
            ax.plot(*c, color='grey', alpha=0.4)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    def plot(self, n=(100, 100), control_polygon=True, display=True, fig=None):
        """
        :param n: resolution
        :param control_polygon: whether to plot the control polygon or not
        :param display: whether to call plt.show
        :fig: a figure handle
        :return: a figure handle
        """

        dim = self.D
        if dim > 3:
            raise NotImplementedError('Cannot plot curves if dimension higher than 3')

        # if no figure handle was given, create a new one
        # otherwise, get current axis
        if fig == None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        x, y, z = self.evaluate(n=n)

        if dim != 1:
            ax.plot_surface(x, y, z)

        if control_polygon and dim != 1:
            self._plot_control_grid(ax)
        if display:
            plt.show()

        return fig

if __name__ == '__main__':

    def curve_demo():
        control_points = np.array([[1, 2, 3], [3, 1, 2], [3, 2, 1], [1, 0, 0]])
        curve = BezierCurve(control_points=control_points)

        curve.plot(n=50)

    def comp_curve_demo():
        control_points_one = np.array([[2, 3, 1], [4, 2, 3], [3, 2, 1]])
        control_points_two = np.array([[3, 2, 1], [5, 2, 3], [2, 1, 1]])

        p = BezierCurve(control_points_one, a=0, b=1)
        q = BezierCurve(control_points_two, a = 1, b =2)

        comp = CompositeBezierCurve([p, q], [0, 1, 2])
        comp.plot()

    def surface_demo():
        control_points = np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                                   [[0.5, 0.9, 1], [0.5, 2, 0.5], [0.5, 1, 2]],
                                   [[1, 0, 0], [1, 3, 0], [1, 1, 1]]])
        test = BezierSurface(control_points)
        test.plot(n=(50, 50))
    comp_curve_demo()
    surface_demo()
    curve_demo()
