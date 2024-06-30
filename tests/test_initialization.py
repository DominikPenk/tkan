import unittest

import torch

from tkan.nn import ChebyshevKanLinear, HermiteKanLinear, LegendreKanLinear
from tkan.nn.base import KanLinearBase, PolynomialKanLinear
from tkan.nn.init import init_with_non_linearity


class Test_TestInitWithNonLinearity(unittest.TestCase):
    def test_init_custom_layer(self):
        class TestLayer(KanLinearBase):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                self.scales = torch.nn.Parameter(torch.ones(out_features, in_features))

            def compute_activations(self, x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(-2) ** 2 * self.scales

        layer = TestLayer(3, 4)
        init_with_non_linearity(layer, lambda x: 0.5 * x ** 2, (-1, 1))

        self.assertTrue(torch.allclose(
            layer.scales, 
            torch.full_like(layer.scales, 0.5),
            atol=1e-3
        ))

    def test_init_polynomial_layer(self):
        layer_classes = [
            ChebyshevKanLinear,
            HermiteKanLinear,
            LegendreKanLinear
        ]

        non_linearity = lambda x: x
        domain = (-1, 1)

        for layer_class in layer_classes:
            with self.subTest(layer_class=layer_class.__name__):
                layer:PolynomialKanLinear = layer_class(3, 4, order=4)
                mse = init_with_non_linearity(layer, non_linearity, domain)
                self.assertLess(mse, 1e-3)

                self.assertTrue(torch.allclose(layer.control_points[..., 0], torch.ones_like(layer.control_points[..., 0])))
                self.assertTrue(torch.allclose(layer.control_points[..., 1:], torch.zeros_like(layer.control_points[..., 1:]), atol=1e-6))

