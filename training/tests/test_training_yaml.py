"""Check YAML runtime defaults, CLI priority and generated training settings."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import parser as training_parser
from utils.config import cfg_from_yaml_file
import test_prepare_mujoco_dataset as dataset_tests


class TrainingYamlTests(unittest.TestCase):
    def test_separate_dataset_config_and_single_training_address(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = dataset_tests.DatasetPreparationTests().fixture(root)
            config = yaml.safe_load((ROOT / 'cfgs/MuJoCo_models/dataset.yaml').read_text())
            config.update(source=str(fixture.source), sim_config=str(fixture.sim_config), output=str(fixture.output))
            path = root / 'dataset.yaml'
            path.write_text(yaml.safe_dump(config))
            train_path = root / 'train.yaml'
            training = yaml.safe_load((ROOT / 'cfgs/MuJoCo_models/train.yaml').read_text())
            training['dataset_path'] = str(fixture.output)
            train_path.write_text(yaml.safe_dump(training))
            before = train_path.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / 'tools/prepare_mujoco_dataset.py'),
                '--config', str(path)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(train_path.read_bytes(), before)
            self.assertFalse((fixture.output / 'train.yaml').exists())
            self.assertTrue((fixture.output / 'loader.yaml').exists())
            resolved = cfg_from_yaml_file(str(train_path))
            for split in ['train', 'val', 'test']:
                self.assertEqual(resolved.dataset[split]._base_.DATA_PATH, str(fixture.output))
                self.assertEqual(resolved.dataset[split].others.subset, split)
            repeated = subprocess.run([sys.executable, str(ROOT / 'tools/prepare_mujoco_dataset.py'),
                '--config', str(path)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)

    def test_runtime_defaults_and_cli_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'train.yaml'
            config.write_text(yaml.safe_dump(dict(runtime=dict(
                num_workers=0, seed=73, val_freq=5, exp_name='yaml_test', deterministic=True))))
            argv = ['main.py', '--config', str(config), '--num_workers', '3']
            with patch.object(sys, 'argv', argv), patch.object(training_parser, 'create_experiment_dir'):
                args = training_parser.get_args()
            self.assertEqual(args.num_workers, 3)
            self.assertEqual(args.seed, 73)
            self.assertEqual(args.val_freq, 5)
            self.assertTrue(args.deterministic)
            self.assertEqual(args.exp_name, 'yaml_test')
            config.write_text('runtime:\n  num_worker: 2\n')
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    training_parser.get_args()



if __name__ == '__main__':
    unittest.main()
