from math import factorial


class BasisFunction(object):
    """
    The most basic basis-function
    keeping track of it's degree as well as it's number.
    """

    def __init__(self, i, degree):
        """
        :param i: the i'th basis function
        :param degree: the degree of the basis function
        """
        self.i = i
        self.d = degree

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ not implemented for class BasisFunction')


class BernsteinPolynomial(BasisFunction):
    """
    The i'th basis Bernstein polynomial of degree d.
    """

    def __call__(self, x):
        """
        :param x: point of evaluation
        :return: the value of the i'th bernstein polynomial of degree d at x
        """
        d, i = self.d, self.i

        binomial_coeff = factorial(d) / float(factorial(i) * factorial(d - i))

        return binomial_coeff * x ** i * (1 - x) ** (d - i)
