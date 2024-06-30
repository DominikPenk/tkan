import unittest

import torch

from tkan.nn import (ChebyshevKan, FixedNodesLagrangeKan,
                         HermiteKan, LagrangeKan, LegendreKan)
from tkan.nn.base import KanLayerBase
from tkan.training.losses import KanLayerRegularizationLoss


class Test_TestRegularization(unittest.TestCase):
    def test_regularization_is_0d_tensor(self):
        layer_types = [
            LagrangeKan,
            FixedNodesLagrangeKan,
            LegendreKan,
            ChebyshevKan,
            HermiteKan,
        ]

        for layer_type in layer_types:
            with self.subTest(layer_type=layer_type.__name__):
                layer:KanLayerBase = layer_type(3, 4)
                self.assertTrue(isinstance(layer.regularization_loss(), torch.Tensor))
                self.assertEqual(layer.regularization_loss().ndim, 0)

    def test_regularization_custom_layer(self):
        class CustomKanLayer(KanLayerBase):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                self.parameter = torch.nn.Parameter(torch.ones(out_features, in_features))

        layer = CustomKanLayer(3, 4)
        self.assertTrue(isinstance(layer.regularization_loss(), torch.Tensor))
        self.assertEqual(layer.regularization_loss().ndim, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_regularization_result_device(self):
        layer_types = [
            LagrangeKan,
            LegendreKan,
            ChebyshevKan,
            HermiteKan,
        ]

        devices = [
            torch.device('cpu'),
            torch.device('cuda:0')
        ]

        for layer_type in layer_types:
                with self.subTest(layer_type=layer_type.__name__):
                    layer:KanLayerBase = layer_type(3, 4)
                    for device in devices:
                        with self.subTest(device=device):
                            layer = layer.to(device)
                            self.assertEqual(layer.regularization_loss().device, device)

class Test_KanOrderRegularizationLoss(unittest.TestCase):
    def test_kan_order_regularization_loss(self):
        simple_model = torch.nn.Sequential(
            LagrangeKan(3, 4),
            LagrangeKan(4, 5),
            LagrangeKan(5, 1),
            HermiteKan(1, 1)
        )

        reg_loss = KanLayerRegularizationLoss(simple_model)
        self.assertEqual(len(reg_loss._layers), 4)
        self.assertEqual(reg_loss.compute().ndim, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_kan_order_regularization_loss_cuda(self):
        simple_model = torch.nn.Sequential(
            LagrangeKan(3, 4),
            LagrangeKan(4, 5),
            LagrangeKan(5, 1),
            HermiteKan(1, 1)
        )

        reg_loss = KanLayerRegularizationLoss(simple_model)
        self.assertEqual(reg_loss.compute().device, torch.device('cpu'))

        cuda_device = torch.device("cuda:0")
        simple_model = simple_model.to(cuda_device)
        self.assertEqual(reg_loss.compute().device, cuda_device)