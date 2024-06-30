import unittest

import torch

from tkan.nn import LagrangeKanLayer, FixedNodesLagrangeKanLayer


class Test_TestLagrangeKanLayer(unittest.TestCase):
    def test_output_shape(self):
        layer = LagrangeKanLayer(in_features=3, out_features=4, nodes=5)
        input = torch.randn(32, 3)
        output = layer(input)
        self.assertEqual(output.shape, (32, 4))

    @torch.inference_mode()
    def test_multiple_batch_dimensioins(self):
        layer = LagrangeKanLayer(in_features=7, out_features=1, nodes=4)
        input = torch.randn(128, 128, 7)
        output = layer(input)
        self.assertEqual(output.shape, (128, 128, 1))

    def test_node_positions_linear(self):
        start = -12
        end = 42
        layer = LagrangeKanLayer(
            in_features=3, 
            out_features=4, 
            nodes=5, 
            node_positions='linear',
            domain=(start, end)
        )
        nodes = torch.linspace(-12, 42, 5).view(1, 1, -1).repeat(4, 3, 1)
        self.assertTrue(torch.allclose(layer.nodes, nodes))

    def test_interpolation_property(self):
        layer = LagrangeKanLayer(in_features=1, out_features=1, nodes=5)
        input = layer.nodes[0].transpose(1, 0)
        output:torch.Tensor = layer(input)
        all_close = torch.allclose(
            output.flatten(),
            layer.control_points.flatten(),
            atol=1e-6
        )
        self.assertTrue(all_close)

class Test_TestFixedNodesLayer(unittest.TestCase):
    def test_output_shape(self):
        layer = FixedNodesLagrangeKanLayer(in_features=3, out_features=4, nodes=5)
        input = torch.randn(32, 3)
        output = layer(input)
        self.assertEqual(output.shape, (32, 4))

    @torch.inference_mode()
    def test_multiple_batch_dimensioins(self):
        layer = LagrangeKanLayer(in_features=7, out_features=1, nodes=4)
        input = torch.randn(128, 128, 7)
        output = layer(input)
        self.assertEqual(output.shape, (128, 128, 1))

    def test_node_positions_linear(self):
        start = -12
        end = 42
        layer = FixedNodesLagrangeKanLayer(
            in_features=3, 
            out_features=4, 
            nodes=5, 
            node_positions='linear',
            domain=(start, end)
        )
        nodes = torch.linspace(-12, 42, 5).view(1, 1, -1).repeat(4, 3, 1)
        self.assertTrue(torch.allclose(layer.nodes, nodes))

    def test_interpolation_property(self):
        layer = FixedNodesLagrangeKanLayer(in_features=1, out_features=1, nodes=5)
        input = layer.nodes[0].transpose(1, 0)
        output:torch.Tensor = layer(input)
        all_close = torch.allclose(
            output.flatten(),
            layer.control_points.flatten(),
            atol=1e-6
        )
        self.assertTrue(all_close)

if __name__ == '__main__':
    unittest.main()