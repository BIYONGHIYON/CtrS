import unittest
import torch
from ssamrn.gradient_loss import masked_gradient_l1


class GradientLossTests(unittest.TestCase):
    def test_constant_offset_has_no_gradient_error(self):
        x = torch.randn(2, 3, 4, 4)
        self.assertLess(masked_gradient_l1(x + 2, x).item(), 1e-6)

    def test_invalid_pair_excluded(self):
        prediction = torch.tensor([[[[0., 1., 100.]]]], requires_grad=True)
        mask = torch.tensor([[[[True, True, False]]]])
        loss = masked_gradient_l1(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(prediction.grad[..., -1].item(), 0.)

    def test_empty_mask_is_finite_and_differentiable(self):
        prediction = torch.randn(1, 2, 3, 3, requires_grad=True)
        loss = masked_gradient_l1(prediction, torch.zeros_like(prediction), torch.zeros(1, 1, 3, 3))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
