import unittest
import torch
from PIL import Image
import numpy as np
from src.controlnet.utils import prepare_mask


class TestPrepareMask(unittest.TestCase):

    def test_prepare_mask_tensor(self):
        # Test prepare_mask with a torch Tensor input
        mask = torch.ones((1, 3, 3))  # Example input tensor
        width = 5
        height = 5
        batch_size = 2
        num_images_per_prompt = 1
        device = 'cpu'
        dtype = torch.float32
        prepared_mask = prepare_mask(mask, width, height, batch_size, num_images_per_prompt, device, dtype)
        self.assertTrue(torch.is_tensor(prepared_mask))
        self.assertEqual(prepared_mask.shape, (batch_size, 1, height, width))

    def test_prepare_mask_numpy(self):
        # Test prepare_mask with a numpy array input
        mask = np.ones((3, 3, 1))  # Example numpy array
        width = 5
        height = 5
        batch_size = 2
        num_images_per_prompt = 1
        device = 'cpu'
        dtype = torch.float16
        prepared_mask = prepare_mask(mask, width, height, batch_size, num_images_per_prompt, device, dtype)
        self.assertTrue(torch.is_tensor(prepared_mask))
        self.assertEqual(prepared_mask.shape, (batch_size, 1, height, width))

    def test_prepare_mask_image(self):
        # Test prepare_mask with a PIL Image input
        mask = Image.new('L', (3, 3), color=255)  # Example PIL Image
        width = 5
        height = 5
        batch_size = 2
        num_images_per_prompt = 1
        device = 'cpu'
        dtype = torch.float32
        prepared_mask = prepare_mask(mask, width, height, batch_size, num_images_per_prompt, device, dtype)
        self.assertTrue(torch.is_tensor(prepared_mask))
        self.assertEqual(prepared_mask.shape, (batch_size, 1, height, width))

    def test_prepare_mask_invalid_type(self):
        # Test prepare_mask with an invalid input type
        mask = "invalid"  # Example invalid input
        width = 5
        height = 5
        batch_size = 2
        num_images_per_prompt = 1
        device = 'cpu'
        dtype = torch.float32
        with self.assertRaises(NotImplementedError):
            prepare_mask(mask, width, height, batch_size, num_images_per_prompt, device, dtype)


if __name__ == '__main__':
    unittest.main()
