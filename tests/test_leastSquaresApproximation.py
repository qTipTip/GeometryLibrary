from unittest import TestCase

from src.Approximation import LeastSquaresApproximation
import numpy as np

class TestLeastSquaresApproximation(TestCase):
    z_values = [2, 0, 2, 4]
    x_values = np.linspace(0, 1, num=len(z_values))

    def test___call__(self):

        approx = LeastSquaresApproximation(x_values=self.x_values, z_values=self.z_values, n=len(self.z_values))
        for x, z in zip(self.x_values, self.z_values):
            self.assertAlmostEquals(approx(x), z, places=4)

