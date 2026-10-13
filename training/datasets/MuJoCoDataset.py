"""Paired simulation point clouds produced by tools/prepare_mujoco_dataset.py."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class MuJoCo(Dataset):
    def __init__(self, config):
        self.root = Path(config.DATA_PATH)
        self.npoints = int(config.N_POINTS)
        self.input_points = int(config.get('INPUT_POINTS', 2048))
        self.samples = json.loads((self.root / f'{config.subset}.json').read_text())
        if not self.samples:
            raise ValueError(f'Empty MuJoCo split: {config.subset}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        partial = np.load(self.root / sample['partial'], allow_pickle=False)
        complete = np.load(self.root / sample['complete'], allow_pickle=False)
        if partial.shape != (self.input_points, 3) or complete.shape != (self.npoints, 3):
            raise ValueError(f'Invalid point count for {sample["id"]}')
        return sample['taxonomy_id'], sample['id'], (
            torch.from_numpy(partial.astype(np.float32)),
            torch.from_numpy(complete.astype(np.float32)))
