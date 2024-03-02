import unittest
import torch
from src.controlnet.utils import apply_mask


class TestApplyMask(unittest.TestCase):

    def test_apply_mask(self):
        # Creating a simple mask and tensor for testing
        mask = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        mask = mask.unsqueeze(0).repeat(1, 2, 1, 1)
        tensor = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32)

        # Applying the mask
        masked_tensor = apply_mask(mask, tensor)

        # Expected result after applying the mask
        expected_result = torch.tensor([[[[1, 0], [0, 4]], [[5, 0], [0, 8]]]], dtype=torch.float32)

        # Asserting equality
        self.assertTrue(torch.allclose(masked_tensor, expected_result))


if __name__ == '__main__':
    unittest.main()
