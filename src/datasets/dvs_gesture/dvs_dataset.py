import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# random spatial shift applied identically to all frames of a sample
class RandomTranslateEventFrames:
    """Expects (T, 2, H, W) or (2, H, W). 
    NOTE: motion vectors between consecutive frames are preserved, so this is
    safe to compose with direction-selective channels.
    """

    def __init__(self, max_shift: int):
        self.max_shift = int(max_shift)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")
        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)
        if dx == 0 and dy == 0:
            return x
        H, W = x.shape[-2], x.shape[-1]
        out = torch.zeros_like(x)
        h_src = slice(max(0, -dy), H - max(0, dy))
        h_dst = slice(max(0, dy), H - max(0, -dy))
        w_src = slice(max(0, -dx), W - max(0, dx))
        w_dst = slice(max(0, dx), W - max(0, -dx))
        out[..., h_dst, w_dst] = x[..., h_src, w_src]
        return out


class PreprocessedNpzDataset(Dataset):
    """Frames cached as npz, flattened to (T, C*H*W) per sample."""

    # polarity indices in the cached npz frames
    OFF_IDX = 0
    ON_IDX = 1

    # fixed pixel permutation, built once and shared across train/val/test
    _shared_pixel_permutation = None

    def __init__(
        self,
        npz_root: str,
        split: str = "test",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform=None,
        target_transform=None,
        direction_selective: bool = False,
        shuffle_inputs: bool = False,
        permutation_seed: int = 42,
        input_size: int = 64,
    ):
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.direction_selective = direction_selective
        self.shuffle_inputs = shuffle_inputs
        self.input_size = input_size

        if self.shuffle_inputs:
            if PreprocessedNpzDataset._shared_pixel_permutation is None:
                perm = np.random.default_rng(permutation_seed).permutation(
                    input_size**2
                )
                PreprocessedNpzDataset._shared_pixel_permutation = perm
            self.input_permutation = torch.from_numpy(
                np.asarray(PreprocessedNpzDataset._shared_pixel_permutation)
            ).long()
        else:
            self.input_permutation = None

        # train and val both read from train/, test reads from test/
        folder = "train" if split in ("train", "val") else "test"
        scan_root = os.path.join(npz_root, folder)

        classes = sorted(e.name for e in os.scandir(scan_root) if e.is_dir())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(scan_root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".npz"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        if split in ("train", "val"):
            self._apply_train_val_split()

    # split by recording subject, so no subject appears in both train and val
    def _apply_train_val_split(self):
        subjects = []
        for path, _ in self.samples:
            fname = os.path.basename(path)
            subject_id = int(fname.split("_")[0].replace("user", ""))
            subjects.append(subject_id)

        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_subjects)

        n_val = int(len(unique_subjects) * self.val_ratio)
        val_subjects = set(unique_subjects[:n_val])
        train_subjects = set(unique_subjects[n_val:])

        keep_subjects = train_subjects if self.split == "train" else val_subjects

        self.samples = [
            (path, label)
            for (path, label), subject in zip(self.samples, subjects)
            if subject in keep_subjects
        ]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _direction_channels_ema(
        frames: torch.Tensor,
        on_idx: int,
        off_idx: int,
        decay: float = 0.7,
    ) -> torch.Tensor:
        """Hassenstein-Reichardt correlator: (T, 2, H, W) -> (T, 16, H, W).

        Null-direction opponency on 8 directions (4 cardinal + 4 diagonal) per
        polarity, correlating each frame against an exponential moving average
        of past frames.

        Output channel order (8 directions per polarity, ON then OFF):
            0:  ON+right,       1:  ON+left,       2:  ON+down,       3:  ON+up,
            4:  ON+down_right,  5:  ON+down_left,  6:  ON+up_right,   7:  ON+up_left,
            8:  OFF+right,      9:  OFF+left,     10:  OFF+down,     11:  OFF+up,
           12:  OFF+down_right,13:  OFF+down_left,14:  OFF+up_right, 15:  OFF+up_left.
        """
        T = frames.shape[0]

        M = torch.zeros_like(frames)
        for t in range(1, T):
            M[t] = decay * M[t - 1] + (1.0 - decay) * frames[t - 1]

        # cardinal neighbors
        M_from_left = torch.roll(M, shifts=1, dims=-1)
        M_from_left[..., :, 0] = 0
        M_from_right = torch.roll(M, shifts=-1, dims=-1)
        M_from_right[..., :, -1] = 0
        M_from_above = torch.roll(M, shifts=1, dims=-2)
        M_from_above[..., 0, :] = 0
        M_from_below = torch.roll(M, shifts=-1, dims=-2)
        M_from_below[..., -1, :] = 0

        # diagonal neighbors
        M_from_upleft = torch.roll(M, shifts=(1, 1), dims=(-2, -1))
        M_from_upleft[..., 0, :] = 0
        M_from_upleft[..., :, 0] = 0
        M_from_upright = torch.roll(M, shifts=(1, -1), dims=(-2, -1))
        M_from_upright[..., 0, :] = 0
        M_from_upright[..., :, -1] = 0
        M_from_downleft = torch.roll(M, shifts=(-1, 1), dims=(-2, -1))
        M_from_downleft[..., -1, :] = 0
        M_from_downleft[..., :, 0] = 0
        M_from_downright = torch.roll(M, shifts=(-1, -1), dims=(-2, -1))
        M_from_downright[..., -1, :] = 0
        M_from_downright[..., :, -1] = 0

        # opponency on cardinal axes
        right = torch.relu(frames * (M_from_left - M_from_right))
        left = torch.relu(frames * (M_from_right - M_from_left))
        down = torch.relu(frames * (M_from_above - M_from_below))
        up = torch.relu(frames * (M_from_below - M_from_above))

        # opponency on diagonal axes
        down_right = torch.relu(frames * (M_from_upleft - M_from_downright))
        down_left = torch.relu(frames * (M_from_upright - M_from_downleft))
        up_right = torch.relu(frames * (M_from_downleft - M_from_upright))
        up_left = torch.relu(frames * (M_from_downright - M_from_upleft))

        on = torch.stack(
            [
                right[:, on_idx], left[:, on_idx], down[:, on_idx], up[:, on_idx],
                down_right[:, on_idx], down_left[:, on_idx],
                up_right[:, on_idx], up_left[:, on_idx],
            ],
            dim=1,
        )
        off = torch.stack(
            [
                right[:, off_idx], left[:, off_idx], down[:, off_idx], up[:, off_idx],
                down_right[:, off_idx], down_left[:, off_idx],
                up_right[:, off_idx], up_left[:, off_idx],
            ],
            dim=1,
        )
        return torch.cat([on, off], dim=1)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path, allow_pickle=True)["frames"].astype(np.float32)
        frames = torch.from_numpy(frames)  # (T, 2, H, W)

        if self.transform is not None:
            frames = self.transform(frames)

        # NOTE: the shuffle must happen BEFORE the direction-channel split.
        if self.shuffle_inputs:
            T0, C0, H0, W0 = frames.shape
            assert H0 * W0 == self.input_permutation.numel(), (
                f"pixel permutation size {self.input_permutation.numel()} does not "
                f"match H*W={H0 * W0}"
            )
            frames = frames.reshape(T0, C0, H0 * W0)[:, :, self.input_permutation]
            frames = frames.reshape(T0, C0, H0, W0)

        if self.direction_selective == "ema":
            frames = self._direction_channels_ema(
                frames, on_idx=self.ON_IDX, off_idx=self.OFF_IDX, decay=0.6
            )

        T, C, H, W = frames.shape
        frames = frames.reshape(T, C * H * W)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label
