import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dvs_gesture.dvs_dataset import PreprocessedNpzDataset


class CIFAR10DVSPreprocessedDataset(Dataset):
    """Frames cached as npz, flattened to (T, C*H*W) per sample.

    NOTE: this dataset has no official split and no subject information, so the
    split is deterministic per class: indices are shuffled with `seed` and
    partitioned by `val_ratio` and `test_ratio` (default 8:1:1).
    """

    # polarity indices in the cached npz frames
    OFF_IDX = 0
    ON_IDX = 1

    def __init__(
        self,
        npz_root: str,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        spatial_bin_size: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        direction_selective=False,
    ):
        assert split in ("train", "val", "test"), split
        assert 0.0 <= val_ratio < 1.0
        assert 0.0 <= test_ratio < 1.0
        assert val_ratio + test_ratio < 1.0, (
            f"val_ratio + test_ratio must be < 1, got {val_ratio} + {test_ratio}"
        )
        assert isinstance(spatial_bin_size, int) and spatial_bin_size >= 1

        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.spatial_bin_size = spatial_bin_size
        self.transform = transform
        self.target_transform = target_transform
        self.direction_selective = direction_selective

        classes = sorted(e.name for e in os.scandir(npz_root) if e.is_dir())
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        rng = np.random.default_rng(seed)
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(npz_root, cls)
            files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".npz"))
            n_total = len(files)
            if n_total == 0:
                continue

            idx = np.arange(n_total)
            rng.shuffle(idx)

            n_val = int(round(val_ratio * n_total))
            n_test = int(round(test_ratio * n_total))
            n_train = n_total - n_val - n_test

            if split == "train":
                keep = idx[:n_train]
            elif split == "val":
                keep = idx[n_train : n_train + n_val]
            else:
                keep = idx[n_train + n_val :]

            label = self.class_to_idx[cls]
            for k in keep:
                self.samples.append((os.path.join(cls_dir, files[k]), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path, allow_pickle=True)["frames"].astype(np.float32)
        frames = torch.from_numpy(frames)  # (T, 2, H, W)

        if self.spatial_bin_size > 1:
            b = self.spatial_bin_size
            T, C, H, W = frames.shape
            # drop the last partial bin if H or W is not divisible by b
            H_keep = (H // b) * b
            W_keep = (W // b) * b
            frames = frames[:, :, :H_keep, :W_keep]
            frames = frames.reshape(T, C, H_keep // b, b, W_keep // b, b).sum(
                dim=(3, 5)
            )

        if self.transform is not None:
            frames = self.transform(frames)

        # reused from the gesture module so both datasets share one correlator
        if self.direction_selective == "ema":
            frames = PreprocessedNpzDataset._direction_channels_ema(
                frames, on_idx=self.ON_IDX, off_idx=self.OFF_IDX, decay=0.6
            )

        T, C, H, W = frames.shape
        frames = frames.reshape(T, C * H * W)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label
