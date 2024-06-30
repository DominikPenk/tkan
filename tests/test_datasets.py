import unittest

import torch

from tkan.datasets import get_feymann_equations

class Test_TestDatasets(unittest.TestCase):
    @torch.inference_mode()
    def test_feymann_datasets(self) -> None:
        for i, eq in enumerate(get_feymann_equations()):
            with self.subTest(i=i, eq=eq):
                x, y = eq.create_dataset(1000)
                self.assertEqual(x.shape, (1000, eq.dimensions))
                self.assertEqual(y.shape, (1000, 1))
                self.assertFalse(x.isnan().any())
                self.assertFalse(x.isinf().any())