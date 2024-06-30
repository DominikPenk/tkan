import unittest

import numpy as np
import torch
import torch.nn as nn

from tkan.nn import LagrangeKan
from tkan.training import KanNodeSparsityLoss, kan_node_sparsity_loss


class Test_TestNodeSparsityLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.net = nn.Sequential(
            LagrangeKan(2, 3),
            LagrangeKan(3, 1)
        )

    def test_functional_loss_output_shapes(self):
        batch_shapes = ((128, ), (64, 32))

        for batch_shape in batch_shapes:
            with self.subTest(batch_shape=batch_shape):
                activations = torch.zeros((*batch_shape, 2))
                norm_loss, entropy_loss = kan_node_sparsity_loss(activations)

                self.assertEqual(norm_loss.shape, ())
                self.assertEqual(entropy_loss.shape, ())

                self.assertGreater(norm_loss, 0.0)
                self.assertGreater(entropy_loss, -np.spacing(1.0))

    def test_node_sparsity_class(self):
        reg_loss_1 = KanNodeSparsityLoss(self.net)
        reg_loss_2 = KanNodeSparsityLoss(self.net, lambda_norm=0.1)

        batch_shapes = ((128, ), (64, 32))

        for batch_shape in batch_shapes:
            with self.subTest(batch_shape=batch_shape):
                x = torch.rand((*batch_shape, 2))
                self.net(x)

                loss_1 = reg_loss_1.compute()
                loss_2 = reg_loss_2.compute()

                self.assertEqual(loss_1.shape, ())
                self.assertGreater(loss_1, 0.0)
                self.assertLessEqual(loss_1, loss_2)
