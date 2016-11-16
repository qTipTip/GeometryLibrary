from unittest import TestCase

from src.Bezier import BezierCurve
import numpy as np

class TestBezierCurve(TestCase):
    control_points = np.array([(1, 2, 3), (3, 4, 5), (5, 6, 7)], dtype=float)

    def test__set_degree_and_dimension(self):
        curve = BezierCurve(control_points=self.control_points)
        expected_degree = len(self.control_points) - 1
        expected_dimension = len(self.control_points)
        degree, dimension = curve.d, curve.D

        self.assertEquals(degree, expected_degree)
        self.assertEquals(dimension, expected_dimension)

    def test__set_basis_functions(self):
        curve = BezierCurve(control_points=self.control_points)

        for i, b in enumerate(curve.basis):
            self.assertEquals(i, b.i)


    def test___call__endpoint_property(self):

        curve = BezierCurve(control_points=self.control_points)
        start_point = list(self.control_points[0])
        end_point = list(self.control_points[2])
        computed_start = list(curve(0))
        computed_end = list(curve(1))
        self.assertTrue(np.allclose(start_point, computed_start), msg='%s != %s ' % (start_point, computed_start))
        self.assertTrue(np.allclose(end_point, computed_end), msg='%s != %s ' % (end_point, computed_end))

    def test_evaluate(self):
        curve = BezierCurve(control_points=self.control_points)
        t, y = curve.evaluate(n=3)

        self.assertTrue(np.allclose(y, self.control_points), msg='%s != %s' % (y, self.control_points))