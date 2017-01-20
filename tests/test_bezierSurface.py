from unittest import TestCase
import numpy as np

from src.Bezier import BezierSurface


class TestBezierSurface(TestCase):
    control_points = np.array([[[0, 0, 0], [2, 0, 0], [4, 0, 0]],
                              [[0, 2, 0], [2, 2, 0], [4, 2, 2]],
                               [[0, 4, 0], [2, 4, 4], [4, 4, 4]]])

    def test__set_degree_and_dimension(self):
        surface = BezierSurface(control_points=self.control_points)

        expected_degree = (3-1, 3-1) # number of control points minus 1 in each direction
        expected_dimension = 3

        degree = surface.d
        dimension = surface.D

        self.assertEquals(expected_degree, degree)
        self.assertEquals(expected_dimension, dimension)

    def test__set_basis_functions(self):
        surface = BezierSurface(control_points=self.control_points)
        for i, b in enumerate(surface.basis[0]):
            self.assertEquals(i, b.i)
        for j, b in enumerate(surface.basis[1]):
            self.assertEquals(j, b.i)

    def test___call__(self):
        surface = BezierSurface(control_points=self.control_points)
        expected = np.array([2, 2, 1])
        computed = surface(0.5, 0.5)

        self.assertEquals(computed.shape, (3,))
        self.assertTrue(np.allclose(expected, computed), msg='%s != %s' % (expected, computed))

    def test_endpoint_property(self):
        test = BezierSurface(self.control_points)
        m, n = test.d
        x, y, z = test.evaluate()
        self.assertTrue(np.allclose(test.cp[0, 0], [x[-1, -1], y[-1, -1], z[-1, -1]]))
        self.assertTrue(np.allclose(test.cp[-1, -1], [x[0, 0], y[0, 0], z[0, 0]]))
