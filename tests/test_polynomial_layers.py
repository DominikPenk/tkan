import unittest

import torch

from tkan.nn.base import KanLinearBase
from tkan.nn import ChebyshevKanLinear, HermiteKanLinear, LegendreKanLinear


class BasePolynomialLayerTest:
    class BaseTest(unittest.TestCase):
        @torch.inference_mode()
        def test_output_shape(self) -> None:
            batch_sizes = [1, 32, 64, 128, 256]
            for bs in batch_sizes:
                with self.subTest(batch_size=bs):
                    x = torch.empty(bs, self.layer.in_features)
                    y = self.layer(x)
                    self.assertEqual(y.shape, (bs, self.layer.out_features))

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        @torch.inference_mode()
        def test_cuda(self) -> None:
            layer = self.layer.cuda()
            batch_sizes = [1, 32, 64, 128, 256]
            for bs in batch_sizes:
                with self.subTest(batch_size=bs):
                    x = torch.empty(bs, self.layer.in_features, device='cuda')
                    y = layer(x)
                    self.assertEqual(y.shape, (bs, self.layer.out_features))
                    self.assertEqual(y.device, x.device)

        @torch.inference_mode()
        def test_multiple_batch_dimensions(self) -> None:
            x = torch.empty(128, 128, self.layer.in_features)
            y = self.layer(x)
            self.assertEqual(y.shape, (128, 128, self.layer.out_features))

        @torch.inference_mode()
        def test_no_batch_dimension(self) -> None: 
            x = torch.empty(self.layer.in_features)
            y = self.layer(x)
            self.assertEqual(y.shape, (self.layer.out_features,))

        @torch.inference_mode()
        def test_invalid_feature_size(self) -> None:
            with self.assertRaises(ValueError):
                x = torch.empty(self.layer.in_features + 1)
                self.layer(x)

        def test_train_eval_are_same(self) -> None:
            x = torch.rand((128, self.layer.in_features))

            self.layer.train()
            y_train = self.layer(x)

            self.layer.eval()
            y_eval = self.layer(x)

            self.assertTrue(torch.allclose(y_train, y_eval))


class Test_TestChebyshevLayer(BasePolynomialLayerTest.BaseTest):
    def setUp(self) -> None:
        self.layer = ChebyshevKanLinear(in_features=3, out_features=4, order=5).eval()

class Test_TestHermiteLayer(BasePolynomialLayerTest.BaseTest):
    def setUp(self) -> None:
        self.layer = HermiteKanLinear(in_features=3, out_features=4, order=5).eval()


class Test_TestLegendreLayer(BasePolynomialLayerTest.BaseTest):
    def setUp(self) -> None:
        self.layer = LegendreKanLinear(in_features=3, out_features=4, order=5).eval()
        