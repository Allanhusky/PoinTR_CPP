"""Integration cases for paired coordinates, split isolation and invalid views."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_mujoco_dataset import build


class DatasetPreparationTests(unittest.TestCase):
    def fixture(self, root, frame='object'):
        source = root / 'captures'
        source.mkdir()
        cfg = dict(mesh='object.STL', mesh_scale=.001, object_rotation_x_deg=90,
                   density=dict(mode='native_pixels'),
                   camera=dict(width=640, height=480, fx=518., fy=518., cx=320., cy=240.))
        config = root / 'config.json'
        config.write_text(json.dumps(cfg))
        rng = np.random.default_rng(4)
        for i, azimuth in enumerate([0, 5, 120, 240]):
            view = source / f'view_{i}'
            view.mkdir()
            partial = rng.random((256, 3), dtype=np.float32)*.1
            truth = rng.random((512, 3), dtype=np.float32)*.2
            np.save(view / 'partial_object.npy', partial)
            np.save(view / 'complete_object.npy', truth)
            transform = np.eye(4)
            transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            transform[:3, 3] = [.4, .5, .6]
            meta = dict(source_mesh='object.STL', mesh_scale=.001, object_rotation_x_deg=90,
                        camera=cfg['camera'], density=cfg['density'],
                        R_source_object=np.eye(3).tolist(), T_world_camera_optical=transform.tolist(),
                        object_center_world_m=[0, 0, .11], pose=dict(azimuth_deg=azimuth))
            (view / 'metadata.json').write_text(json.dumps(meta))
        bad = source / 'bad'
        bad.mkdir()
        (bad / 'metadata.json').write_text('{broken json')
        return argparse.Namespace(source=source, sim_config=config, frame=frame,
            include_demos=False, all_profiles=False, min_raw_points=128, seed=42,
            input_points=2048, gt_points=8192, sector_deg=30, train_ratio=.7, val_ratio=.15,
            output=root / 'dataset', batch_size=2, epochs=100, lr=.0001)

    def test_shared_normalization_and_sector_isolation(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(Path(folder))
            out = build(args)
            splits = {k: json.loads((out / f'{k}.json').read_text()) for k in ['train', 'val', 'test']}
            all_samples = [s for rows in splits.values() for s in rows]
            self.assertEqual(len(all_samples), 4)
            sectors = [{s['group'] for s in rows} for rows in splits.values()]
            self.assertTrue(all(sectors))
            self.assertTrue(all(sectors[i].isdisjoint(sectors[j]) for i in range(3) for j in range(i)))
            for sample in all_samples:
                n = sample['normalization']
                for filename, source_file in [('partial', 'partial_object.npy'), ('complete', 'complete_object.npy')]:
                    restored = np.load(out / sample[filename])*n['radius_m']+n['centroid']
                    raw = np.load(Path(sample['source']) / source_file)
                    # Upsampling preserves actual measurements, never adds origin padding.
                    distance = np.linalg.norm(restored[:, None]-raw[None], axis=-1).min(axis=1)
                    self.assertLess(float(distance.max()), 1e-7)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(len(report['rejected']), 1)
            with self.assertRaises(FileExistsError):
                build(args)

    def test_camera_truth_and_partial_share_rigid_transform(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            args = self.fixture(Path(folder), frame='camera')
            out = build(args)
            sample = json.loads(next((out / 'metadata').glob('*.json')).read_text())
            meta, n = sample['source_metadata'], sample['normalization']
            t = np.array(meta['T_world_camera_optical'])
            for name, source_file in [('partial', 'partial_object.npy'), ('complete', 'complete_object.npy')]:
                camera = np.load(out / sample[name])*n['radius_m']+n['centroid']
                obj = camera@t[:3, :3].T+t[:3, 3]-meta['object_center_world_m']
                raw = np.load(Path(sample['source']) / source_file)
                error = np.linalg.norm(obj[:, None]-raw[None], axis=-1).min(axis=1)
                self.assertLess(float(error.max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
