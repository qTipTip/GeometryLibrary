from unittest import TestCase

from src.BasisFunctions import BernsteinPolynomial


class TestBernsteinPolynomial(TestCase):
    def test__call__zero_degree(self):
        bernstein = BernsteinPolynomial(i=0, degree=0)

        self.assertAlmostEquals(bernstein(0), 1, places=8, msg='zeroth degree polynomial is not correct')
        self.assertAlmostEquals(bernstein(1), 1, places=8, msg='zeroth degree polynomial is not correct')

    def test__call__first_degree(self):
        bernstein_one = BernsteinPolynomial(i=0, degree=1)
        bernstein_two = BernsteinPolynomial(i=1, degree=1)

        self.assertAlmostEquals(bernstein_one(0.5), 0.5, places=8, msg='first degree polynomials are not correct')
        self.assertAlmostEquals(bernstein_two(0.5), 0.5, places=8, msg='first degree polynomials are not correct')

    def test_partition_of_unity(self):
        d = 3
        total = 0
        for i in range(d + 1):
            bernstein_i = BernsteinPolynomial(i=i, degree=d)
            total += bernstein_i(0.5)

        self.assertAlmostEquals(total, 1, places=8, msg='BernsteinPolynomial does not partition unity')
