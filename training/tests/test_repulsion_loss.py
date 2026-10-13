"""Gradient, neighbor search and legacy-config checks for spacing regularization."""
import unittest

import torch

from utils.repulsion_loss import RepulsionLoss, repulsion_settings, repulsion_weight


class RepulsionLossTests(unittest.TestCase):
    def test_gradient_separates_close_points_and_excludes_self(self):
        points = torch.tensor([[[0., 0., 0.], [.005, 0., 0.]]], requires_grad=True)
        criterion = RepulsionLoss(k=1, radius=.02, anchor_points=0, chunk_size=1)
        loss = criterion(points)
        self.assertAlmostEqual(float(loss), .5625, places=5)
        loss.backward()
        moved = points.detach()-.0001*points.grad
        self.assertGreater(float(torch.dist(moved[0, 0], moved[0, 1])), .005)
        self.assertLess(float(criterion(moved)), float(loss))
        far = torch.tensor([[[0., 0., 0.], [.1, 0., 0.]]], requires_grad=True)
        zero = criterion(far)
        self.assertEqual(float(zero), 0.)
        zero.backward()
        self.assertTrue(torch.isfinite(far.grad).all())

    def test_chunked_loss_and_gradient_match_full_neighbor_reference(self):
        torch.manual_seed(4)
        points = (torch.rand(2, 19, 3, dtype=torch.float64)*.03).requires_grad_()
        criterion = RepulsionLoss(k=3, radius=.02, anchor_points=0, chunk_size=4)
        value = criterion(points)
        grad = torch.autograd.grad(value, points)[0]
        distances = torch.cdist(points, points)
        eye = torch.eye(19, dtype=torch.bool).unsqueeze(0)
        nearest = distances.masked_fill(eye, float('inf')).topk(3, largest=False).values
        reference = torch.relu(1-nearest/.02).square().mean()
        expected = torch.autograd.grad(reference, points)[0]
        torch.testing.assert_close(value, reference)
        torch.testing.assert_close(grad, expected)

    def test_original_input_receives_no_repulsion_gradient(self):
        generated = torch.tensor([[[0., 0., 0.], [.005, 0., 0.]]], requires_grad=True)
        observed = torch.zeros(1, 3, 3, requires_grad=True)
        output = torch.cat([generated, observed], dim=1)
        RepulsionLoss(k=1, radius=.02, anchor_points=0)(output[:, :-observed.shape[1]]).backward()
        self.assertTrue(generated.grad.abs().sum() > 0)
        self.assertEqual(float(observed.grad.abs().sum()), 0.)

    def test_warmup_and_missing_config_preserve_legacy_behavior(self):
        old, _ = repulsion_settings({})
        self.assertEqual(repulsion_weight(old, 99), 0.)
        enabled, _ = repulsion_settings(dict(loss=dict(repulsion=dict(enabled=True))))
        self.assertAlmostEqual(repulsion_weight(enabled, 0), .005/20)
        self.assertAlmostEqual(repulsion_weight(enabled, 19), .005)
        with self.assertRaises(ValueError):
            repulsion_settings(dict(loss=dict(repulsion=dict(raduis=.02))))


if __name__ == '__main__':
    unittest.main()
