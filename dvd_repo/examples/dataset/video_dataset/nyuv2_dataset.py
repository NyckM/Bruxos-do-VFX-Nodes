#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

import os

# Requirements: Numpy as PIL/Pillow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class NYUv2Dataset(Dataset):
    """
    Dataset returning full scene sequences as TCHW tensors, reading depth maps.
    """

    def __init__(
        self,
        data_root,
        start=0,
        test=True,
        target_size=(480, 640),
        min_depth=1e-3,
        max_depth=10,
        transform=None,
        **kwargs

    ):
        self.data_root = data_root
        self.metadata = self.get_meta_data()
        self.samples = self.metadata[start:]
        self.transform = transform
        self.target_size = target_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        if not test:
            self.samples = self.samples[:20]

    def get_meta_data(self):
        metadata = []
        root = self.data_root
        rgb_root = "images"
        # depth_root = "depths_npz"
        depth_root = 'raw_depth'

        for idx, filename in enumerate(sorted(os.listdir(os.path.join(root, rgb_root)))):
            scene_final_path = os.path.join(rgb_root, filename)
            scene_depth_path = os.path.join(
                depth_root, filename.replace('.png', '.npz'))

            metadata.append(
                {
                    "id": idx,
                    "image_path": scene_final_path,
                    "depth_path": scene_depth_path,
                }
            )
        return metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        entry = self.samples[idx]
        _id = entry['id']
        img_path = os.path.join(self.data_root, entry["image_path"])
        depth_path = os.path.join(self.data_root, entry["depth_path"])

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)  # HWC
        img_tensor = torch.from_numpy(img_np).permute(
            2, 0, 1).float() / 255.0  # C,H,W
        img_tensor = img_tensor.unsqueeze(0)  # T=1 -> 1,C,H,W
        eval_mask = torch.zeros_like(img_tensor).bool()

        eval_mask[:, :, 45:471, 41:601] = 1

        depth_np = np.load(depth_path)['depth']
        depth_tensor = torch.from_numpy(
            depth_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        # depth_tensor = torch.clamp(
        #     depth_tensor, min=self.min_depth, max=self.max_depth)

        # img_tensor = F.interpolate(
        #     img_tensor, size=self.target_size, mode="bilinear")
        # depth_tensor = F.interpolate(
        #     depth_tensor, size=self.target_size, mode="nearest")
        # eval_mask = F.interpolate(
        #     eval_mask, size=self.target_size, mode='nearest'
        # )

        sample = {
            'sample_idx': torch.tensor(_id),
            "images": img_tensor,
            "disparity": depth_tensor,
            'eval_mask': eval_mask
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return f"SintelDepthDataset(root='{self.data_root}', scenes={len(self.scenes)})"

