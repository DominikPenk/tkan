import unittest

import torch
import torch.nn as nn

from tkan.nn import HermiteKanLinear, LagrangeKanLayer
from tkan.training import ActivationsTracker

class Test_TestActivationsTracker(unittest.TestCase):
    def setUp(self) -> None:
        inner = nn.Sequential(
            HermiteKanLinear(5, 3),
            nn.Linear(3, 2),
            LagrangeKanLayer(2, 3)
        )
        self.net = nn.Sequential(
            HermiteKanLinear(2, 4),
            nn.Linear(4, 5),
            nn.ReLU(),
            inner
        )

    def test_all_layers_found(self):
        tracker = ActivationsTracker(self.net)
        self.assertEqual(tracker.num_tracked_layers, 3)

    def test_context_manager(self):
        self.net.train()
        tracker = ActivationsTracker(self.net)
        self.assertEqual(len(tracker._hook_handles), 0)
        x = torch.rand((128, 2))

        with tracker:
            self.assertEqual(len(tracker._hook_handles), tracker.num_tracked_layers)
            y = self.net(x)
            self.assertEqual(len(tracker._hook_handles), tracker.num_tracked_layers)
        
        self.assertEqual(len(tracker._hook_handles), 0)

    def test_manual_hook_handling(self):
        self.net.train()
        tracker = ActivationsTracker(self.net)

        self.assertEqual(len(tracker._hook_handles), 0)
        tracker.register_hooks()
        self.assertEqual(len(tracker._hook_handles), tracker.num_tracked_layers)
        tracker.unregister_hooks()
        self.assertEqual(len(tracker._hook_handles), 0)

    def test_captured_activity_shapes(self):
        self.net.train()
        tracker = ActivationsTracker(self.net)

        batch_shapes = [(128,), (64, 128), (4, 5, 23)]

        for batch_shape in batch_shapes:
            with self.subTest(batch_shape=batch_shape):
                with tracker:
                    x = torch.rand((*batch_shape, 2))
                    self.net(x)

                for i, layer in enumerate(tracker._tracked_layers):
                    with self.subTest(layerid=i):
                        in_features  = layer.in_features
                        out_features = layer.out_features
                        act = tracker.get_activation(layer)
                        self.assertEqual(act.shape, (*batch_shape, out_features, in_features))

        tracker.reset()
        self.assertEqual(len(tracker._activations), 0)

    def test_no_tracking_in_eval(self):
        self.net.eval()
        tracker = ActivationsTracker(self.net)

        self.assertFalse(tracker.track_always)

        with tracker:
            self.net(torch.rand((128, 2)))

        self.assertEqual(len(tracker._activations), 0)

    def test_untracked_layer_raises_error(self):
        tracker = ActivationsTracker(self.net)
        untracked_layer = nn.Linear(23, 4)

        with tracker:
            self.net(torch.rand((128, 2)))

        with self.assertRaises(KeyError):
            tracker.get_activation(untracked_layer)

    def test_force_tracking(self):
        self.net.eval()
        tracker = ActivationsTracker(self.net)
        x = torch.rand((128, 2))

        with tracker:
            self.assertFalse(tracker.track_always)
            self.net(x)
            self.assertAlmostEqual(len(tracker._activations), 0)

        with tracker.force_tracking():
            self.assertTrue(tracker.track_always)
            self.net(x)
            
            self.assertEqual(len(tracker._activations), tracker.num_tracked_layers)
            for i, layer in enumerate(tracker._tracked_layers):
                with self.subTest(layerid=i):
                    in_features  = layer.in_features
                    out_features = layer.out_features
                    act = tracker.get_activation(layer)
                    self.assertEqual(act.shape, (128, out_features, in_features))

        tracker.reset()