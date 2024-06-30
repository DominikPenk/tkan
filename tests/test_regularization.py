import unittest

import torch

from tkan.nn import (ChebyshevKanLinear, FixedNodesLagrangeKanLayer,
                         HermiteKanLinear, LagrangeKanLayer, LegendreKanLinear)
from tkan.nn.base import KanLinearBase
from tkan.training.losses import KanLayerRegularizationLoss


class Test_TestRegularization(unittest.TestCase):
    def test_regularization_is_0d_tensor(self):
        layer_types = [
            LagrangeKanLayer,
            FixedNodesLagrangeKanLayer,
            LegendreKanLinear,
            ChebyshevKanLinear,
            HermiteKanLinear,
        ]

        for layer_type in layer_types:
            with self.subTest(layer_type=layer_type.__name__):
                layer:KanLinearBase = layer_type(3, 4)
                self.assertTrue(isinstance(layer.regularization_loss(), torch.Tensor))
                self.assertEqual(layer.regularization_loss().ndim, 0)

    def test_regularization_custom_layer(self):
        class CustomKanLinear(KanLinearBase):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                self.parameter = torch.nn.Parameter(torch.ones(out_features, in_features))

        layer = CustomKanLinear(3, 4)
        self.assertTrue(isinstance(layer.regularization_loss(), torch.Tensor))
        self.assertEqual(layer.regularization_loss().ndim, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_regularization_result_device(self):
        layer_types = [
            LagrangeKanLayer,
            LegendreKanLinear,
            ChebyshevKanLinear,
            HermiteKanLinear,
        ]

        devices = [
            torch.device('cpu'),
            torch.device('cuda:0')
        ]

        for layer_type in layer_types:
                with self.subTest(layer_type=layer_type.__name__):
                    layer:KanLinearBase = layer_type(3, 4)
                    for device in devices:
                        with self.subTest(device=device):
                            layer = layer.to(device)
                            self.assertEqual(layer.regularization_loss().device, device)

class Test_KanOrderRegularizationLoss(unittest.TestCase):
    def test_kan_order_regularization_loss(self):
        simple_model = torch.nn.Sequential(
            LagrangeKanLayer(3, 4),
            LagrangeKanLayer(4, 5),
            LagrangeKanLayer(5, 1),
            HermiteKanLinear(1, 1)
        )

        reg_loss = KanLayerRegularizationLoss(simple_model)
        self.assertEqual(len(reg_loss._layers), 4)
        self.assertEqual(reg_loss.compute().ndim, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_kan_order_regularization_loss_cuda(self):
        simple_model = torch.nn.Sequential(
            LagrangeKanLayer(3, 4),
            LagrangeKanLayer(4, 5),
            LagrangeKanLayer(5, 1),
            HermiteKanLinear(1, 1)
        )

        reg_loss = KanLayerRegularizationLoss(simple_model)
        self.assertEqual(reg_loss.compute().device, torch.device('cpu'))

        cuda_device = torch.device("cuda:0")
        simple_model = simple_model.to(cuda_device)
        self.assertEqual(reg_loss.compute().device, cuda_device)