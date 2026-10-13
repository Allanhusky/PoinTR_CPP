"""Check native-pixel geometry, expected density and depth defects without a GPU."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mojoco_sim'))
from sim import density_grid, sensor_depth


class DepthSensorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'mojoco_sim/config.json').read_text())

    def test_random_sampling_density_and_reproducibility(self):
        c, d = self.cfg['camera'], self.cfg['density']
        d = dict(d, sampling='random_pixels')
        one = density_grid(c, d, np.random.default_rng(4))
        same = density_grid(c, d, np.random.default_rng(4))
        other = density_grid(c, d, np.random.default_rng(5))
        np.testing.assert_array_equal(one, same)
        self.assertFalse(np.array_equal(one, other))
        expected = d['reference_depth_m']**2/(c['fx']*c['fy']*d['spacing_x_m']*d['spacing_y_m'])
        self.assertAlmostEqual(float(one.mean()), expected, delta=.005)
        # Irregular occupancy within each row rather than identical selected columns.
        self.assertFalse(np.array_equal(one[50], one[51]))

    def test_measured_depth_quantization_holes_and_distance_noise(self):
        c = dict(self.cfg['camera'], width=160, height=120)
        mask = np.ones((120, 160), dtype=bool)
        cap = dict(self.cfg['capture'], noise_std_m=0, dropout=0)
        sensor = dict(self.cfg['sensor_model'], enabled=True, edge_dropout=0,
                      hole_fraction=0, outlier_fraction=0)
        errors = []
        for distance in [.5, 1.5]:
            depth = np.full(mask.shape, distance, dtype=np.float32)
            measured, valid = sensor_depth(depth, mask, mask, c, cap, sensor, np.random.default_rng(3))
            self.assertTrue(valid.all())
            self.assertTrue(np.isfinite(measured).all())
            np.testing.assert_allclose(measured/sensor['quantization_m'],
                                       np.rint(measured/sensor['quantization_m']), atol=.001)
            errors.append(float((measured-depth).std()))
        self.assertGreater(errors[1], errors[0]*4)
        sensor['hole_fraction'] = .1
        depth = np.ones(mask.shape, dtype=np.float32)
        measured, valid = sensor_depth(depth, mask, mask, c, cap, sensor, np.random.default_rng(3))
        self.assertLess(int(valid.sum()), int(mask.sum()))
        self.assertTrue((measured[~valid] == 0).all())
        np.testing.assert_array_equal(depth, np.ones(mask.shape))


if __name__ == '__main__':
    unittest.main()
