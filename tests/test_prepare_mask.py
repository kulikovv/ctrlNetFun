import unittest
import torch
import numpy as np
from PIL import Image
from src.controlnet.utils import prepare_mask


class TestPrepareMask(unittest.TestCase):

    def test_prepare_mask_PILImage(self):
        mask = Image.new('L', (32, 32))  # Creating a PIL Image mask
        image = torch.randn(3, 32, 32)    # Creating a random image tensor
        prepared_mask = prepare_mask(mask, image)
        self.assertIsInstance(prepared_mask, torch.Tensor)

    def test_prepare_mask_npArray(self):
        mask = np.zeros((32, 32), dtype=np.uint8)  # Creating a NumPy array mask
        image = torch.randn(5, 3, 32, 32)             # Creating a random image tensor
        prepared_mask = prepare_mask(mask, image)
        self.assertIsInstance(prepared_mask, torch.Tensor)
        self.assertEqual(image.shape[0], prepared_mask.shape[0])

    def test_prepare_mask_Tensor(self):
        mask = torch.zeros(1, 32, 32)  # Creating a PyTorch tensor mask
        image = torch.randn(1, 3, 64, 64)  # Creating a random image tensor
        prepared_mask = prepare_mask(mask, image)
        self.assertIsInstance(prepared_mask, torch.Tensor)
        self.assertEqual(image.shape[-1], prepared_mask.shape[-1])
        self.assertEqual(image.shape[-2], prepared_mask.shape[-2])

    def test_prepare_mask_ListPILImage(self):
        mask_list = [Image.new('L', (32, 32)) for _ in range(3)]  # Creating a list of PIL Image masks
        image = torch.randn(3, 3, 64, 64)                             # Creating a random image tensor
        prepared_mask = prepare_mask(mask_list, image)
        self.assertIsInstance(prepared_mask, torch.Tensor)

    # Add more tests as needed for different scenarios


if __name__ == '__main__':
    unittest.main()
