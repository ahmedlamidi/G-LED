"""Arrays written by prepare_data.py, and normalization helpers."""
import json
import os
import sys

import numpy as np

from .config import REPO_ROOT


def sdflow_dicom_dataset():
    """SD-Flow's own dicom_dataset, so sinograms come from exactly its code."""
    data_dir = os.path.join(REPO_ROOT, 'data')
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    from dicom_preprocess import dicom_dataset
    return dicom_dataset


class SplitData:
    """One split's slices: the manifest, the cached sinograms, and memory-mapped
    images (label, fbp_in, rls), all indexed the same way."""

    def __init__(self, setting_dir, split):
        self.dir = os.path.join(setting_dir, split)
        if not os.path.exists(os.path.join(self.dir, 'DONE')):
            raise SystemExit(f'{self.dir} is missing or incomplete; run python -m baselines.prepare_data first')
        with open(os.path.join(self.dir, 'slices.json')) as f:
            self.slices = json.load(f)
        self._arrays = {}

    def __len__(self):
        return len(self.slices)

    @property
    def ids(self):
        return [s['id'] for s in self.slices]

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(os.path.join(self.dir, name + '.npy'), mmap_mode='r')
        return self._arrays[name]

    def image(self, name, i):
        return np.array(self.array(name)[i], dtype=np.float32)

    def sinogram(self, i):
        return np.load(self.slices[i]['sino']).astype(np.float32)


def load_stats(setting_dir):
    with open(os.path.join(setting_dir, 'stats.json')) as f:
        return json.load(f)


def load_offset(setting_dir):
    """K = FBP(1), see common/ct.py."""
    return np.load(os.path.join(setting_dir, 'K.npy'))


class Normalizer:
    """Global affine map between an image's units and roughly [-1, 1]."""

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def to_unit(self, x):
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1

    def from_unit(self, u):
        return (u + 1) / 2 * (self.hi - self.lo) + self.lo


def minmax_unit(img):
    """Per-image min-max to [-1, 1], the way DOLCE normalizes its condition."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return (2 * (img - lo) / (hi - lo) - 1).astype(np.float32)
