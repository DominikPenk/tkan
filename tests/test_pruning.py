import unittest

import torch

from tkan.nn import (ChebyshevKan, FixedNodesLagrangeKan,
                         HermiteKan, LagrangeKan, LegendreKan)

from tkan.training.pruning import prune_from_data, prune_from_activations


class Test_TestLayerPruning(unittest.TestCase):
    def test_LagrangeKan(self):
        layer = LagrangeKan(3, 4)
        in_features = [0, 2]
        out_features = [1, 3, 0]
        pruned = layer.get_pruned(in_features, out_features)
        self.assertEqual(pruned.in_features, len(in_features))
        self.assertEqual(pruned.out_features, len(out_features))
        
        x = torch.rand((128, len(in_features)))
        y = pruned(x)
        self.assertEqual(y.shape, (128, len(out_features)))

    def test_FixedNodesLagrangeKan(self):
        layer = FixedNodesLagrangeKan(3, 4)
        in_features = [0, 2]
        out_features = [1, 3, 0]
        pruned = layer.get_pruned(in_features, out_features)
        self.assertEqual(pruned.in_features, len(in_features))
        self.assertEqual(pruned.out_features, len(out_features))
        
        x = torch.rand((128, len(in_features)))
        y = pruned(x)
        self.assertEqual(y.shape, (128, len(out_features)))

    def test_ChebyshevKan(self):
        layer = ChebyshevKan(3, 4)
        in_features = [0, 2]
        out_features = [1, 3, 0]
        pruned = layer.get_pruned(in_features, out_features)
        self.assertEqual(pruned.in_features, len(in_features))
        self.assertEqual(pruned.out_features, len(out_features))
        
        x = torch.rand((128, len(in_features)))
        y = pruned(x)
        self.assertEqual(y.shape, (128, len(out_features)))

    def test_LegendreKan(self):
        layer = LegendreKan(3, 4)
        in_features = [0, 2]
        out_features = [1, 3, 0]
        pruned = layer.get_pruned(in_features, out_features)
        self.assertEqual(pruned.in_features, len(in_features))
        self.assertEqual(pruned.out_features, len(out_features))
        
        x = torch.rand((128, len(in_features)))
        y = pruned(x)
        self.assertEqual(y.shape, (128, len(out_features)))

    def test_HermiteKan(self):
        layer = HermiteKan(3, 4)
        in_features = [0, 2]
        out_features = [1, 3, 0]
        pruned = layer.get_pruned(in_features, out_features)
        self.assertEqual(pruned.in_features, len(in_features))
        self.assertEqual(pruned.out_features, len(out_features))
        
        x = torch.rand((128, len(in_features)))


class Test_TestPruningMethods(unittest.TestCase):
    def test_raise_error_on_non_sequential(self):
        class CustomNet:
            def __init__(self):
                self.l1 = LagrangeKan(3, 4)
                self.l2 = LagrangeKan(4, 5)
                self.l3 = LagrangeKan(5, 6)
                self.l4 = LagrangeKan(6, 7)
            
            def forward(self, x):
                x = self.l1(x)

        with self.assertRaises(ValueError):
            prune_from_data(
                CustomNet(),
                torch.rand((128, 3))
            )

    def test_raises_error_on_non_kan_layers(self):
        net = torch.nn.Sequential(
            LagrangeKan(3, 4),
            torch.nn.Linear(4, 2),
            LagrangeKan(2, 1)
        )
        with self.assertRaises(ValueError):
            prune_from_data(net, torch.rand((128, 3)))

    def test_output_is_new_model(self):
        net = torch.nn.Sequential(
            LagrangeKan(3, 4),
            LagrangeKan(4, 1),
            LagrangeKan(1, 1)
        )
        pruned = prune_from_data(net, torch.rand((128, 3)))
        self.assertNotEqual(net, pruned)

    def test_input_is_same_or_smaller(self):
        net = torch.nn.Sequential(
            LagrangeKan(3, 4),
            LagrangeKan(4, 1),
            LagrangeKan(1, 1)
        )
        pruned = prune_from_data(net, torch.rand((128, 3)))

        for original_layer, pruned_layer in zip(net, pruned):
            self.assertLessEqual(pruned_layer.in_features, original_layer.in_features)
            self.assertLessEqual(pruned_layer.out_features, original_layer.out_features)

        self.assertEqual(pruned[0].in_features, net[0].in_features)
        self.assertEqual(pruned[-1].out_features, net[-1].out_features)        

    def test_pruning_by_activations(self):
        net = torch.nn.Sequential(
            LagrangeKan(2, 3),
            LagrangeKan(3, 1)
        )

        activations_layer_1 = torch.cat([
            torch.zeros((128, 1, 2)),
            torch.ones((128, 2, 2))
        ], dim=1)
        activations_layer_2= torch.cat([
            torch.zeros((128, 1, 1)),
            torch.ones((128, 1, 2))
        ], dim=2)

        pruned = prune_from_activations(
            net, 
            [
                activations_layer_1,
                activations_layer_2
            ]
        )

        self.assertEqual((pruned[0].out_features, pruned[0].in_features), (2, 2))
        self.assertEqual((pruned[1].out_features, pruned[1].in_features), (1, 2))

    def test_multiple_batch_dimensions(self):
        net = torch.nn.Sequential(
            LagrangeKan(2, 3),
            LagrangeKan(3, 1)
        )

        activations_layer_1 = torch.cat([
            torch.zeros((128, 64, 1, 2)),
            torch.ones((128, 64, 2, 2))
        ], dim=2)
        activations_layer_2= torch.cat([
            torch.zeros((128, 64, 1, 1)),
            torch.ones((128, 64, 1, 2))
        ], dim=3)

        pruned = prune_from_activations(
            net, 
            [
                activations_layer_1,
                activations_layer_2
            ]
        )

        self.assertEqual((pruned[0].out_features, pruned[0].in_features), (2, 2))
        self.assertEqual((pruned[1].out_features, pruned[1].in_features), (1, 2))