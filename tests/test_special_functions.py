import unittest
import scipy.special
import torch
import numpy as np
from tkan.nn.chebyshev import chebyshev_t_polynomials
from tkan.nn.hermite import hermite_polynomials
from tkan.nn.legendre import legendre_polynomials

class Test_TestSpecialFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.degree = 7
    
    @torch.inference_mode()
    def test_chebyshev_t_polynomials(self):
        x = torch.linspace(-1, 1, 256, dtype=torch.float64)

        y_scipy = scipy.special.eval_chebyt(np.arange(self.degree+1), x[:, None])
        y_torch = chebyshev_t_polynomials(x, self.degree)

        self.assertEqual(y_scipy.shape, y_torch.shape)
        self.assertEqual(y_scipy.dtype, y_torch.dtype)

        for n in range(y_scipy.size(1)):
            with self.subTest(n=n):
                self.assertTrue(torch.allclose(y_scipy[:, n], y_torch[:, n]))

    @torch.inference_mode()
    def test_hermite_polynomials(self):
        x = torch.linspace(-1, 1, 256, dtype=torch.float64)
        y_scipy = scipy.special.eval_hermitenorm(np.arange(self.degree+1), x[:, None])
        y_torch = hermite_polynomials(x, self.degree)

        self.assertEqual(y_scipy.shape, y_torch.shape)
        self.assertEqual(y_scipy.dtype, y_torch.dtype)

        for n in range(y_scipy.size(1)):
            with self.subTest(n=n):
                self.assertTrue(torch.allclose(y_scipy[:, n], y_torch[:, n]))


    @torch.inference_mode()
    def test_legendre_polynomials(self):
        x = torch.linspace(-1, 1, 256, dtype=torch.float64)

        y_scipy = scipy.special.eval_legendre(np.arange(self.degree+1), x[:, None])
        y_torch = legendre_polynomials(x, self.degree)

        self.assertEqual(y_scipy.shape, y_torch.shape)
        self.assertEqual(y_scipy.dtype, y_torch.dtype)

        for n in range(y_scipy.size(1)):
            with self.subTest(n=n):
                self.assertTrue(torch.allclose(y_scipy[:, n], y_torch[:, n]))
                