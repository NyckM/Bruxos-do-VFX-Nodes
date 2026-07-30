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

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"


def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def depth_write(filename, depth):
    """Write depth to file."""
    height, width = depth.shape[:2]
    f = open(filename, "wb")
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)

    depth.astype(np.float32).tofile(f)
    f.close()


def disparity_write(filename, disparity, bitdepth=16):
    """Write disparity to file.

    bitdepth can be either 16 (default) or 32.

    The maximum disparity is 1024, since the image width in Sintel
    is 1024.
    """
    d = disparity.copy()

    # Clip disparity.
    d[d > 1024] = 1024
    d[d < 0] = 0

    d_r = (d / 4.0).astype("uint8")
    d_g = ((d * (2.0**6)) % 256).astype("uint8")

    out = np.zeros((d.shape[0], d.shape[1], 3), dtype="uint8")
    out[:, :, 0] = d_r
    out[:, :, 1] = d_g

    if bitdepth > 16:
        d_b = (d * (2**14) % 256).astype("uint8")
        out[:, :, 2] = d_b

    Image.fromarray(out, "RGB").save(filename, "PNG")


def disparity_read(filename):
    """Return disparity read from filename."""
    f_in = np.array(Image.open(filename))
    d_r = f_in[:, :, 0].astype("float64")
    d_g = f_in[:, :, 1].astype("float64")
    d_b = f_in[:, :, 2].astype("float64")

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return depth


class SintelDataset(Dataset):
    """
    Dataset returning full scene sequences as TCHW tensors, reading depth maps.
    """

    def __init__(
        self,
        data_root,
        start=0,
        stack_scene_depth=True,
        target_size=(480, 640),
        min_depth=0.1,
        max_depth=70.0,
        transform=None,
        **kwargs
    ):
        self.stack_scene_depth = stack_scene_depth
        self.data_root = data_root
        self.metadata = self.get_meta_data()
        self.transform = transform
        self.target_size = target_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        # with open(metadata_json, "r") as f:
        #     self.metadata = json.load(f)

        self.scenes = list(self.metadata.keys())
        if stack_scene_depth:
            self.scenes = self.scenes[start:]
        # print(f"Meta data loaded, {len(self.scenes)} scenes found. first metadata: {self.metadata[self.scenes[0]]}")
        self.samples = []
        for scene in self.scenes:
            scene_frame_list = self.metadata[scene]
            # print(f"{scene} has {len(scene_frame_list)} frames")
            for entry in scene_frame_list:
                self.samples.append(
                    {"scene": scene, "image_path": entry["image_path"], "depth_path": entry["depth_path"]})
        self.samples = sorted(self.samples, key=lambda x: (
            x["scene"], x["image_path"]))

        if not self.stack_scene_depth:
            self.samples = self.samples[start:]

    def get_meta_data(self):
        metadata = {}
        root = self.data_root
        final_root = os.path.join(root, "final")
        depth_root = os.path.join(root, "depth")  # 改成 depth 文件夹

        for scene in os.listdir(final_root):
            scene_final_path = os.path.join(final_root, scene)
            scene_depth_path = os.path.join(depth_root, scene)

            if os.path.isdir(scene_final_path) and os.path.isdir(scene_depth_path):
                image_files = sorted(os.listdir(scene_final_path))
                depth_files = sorted(os.listdir(scene_depth_path))

                # 保证 image 和 depth 数量一致
                assert len(image_files) == len(
                    depth_files
                )

                entries = []
                for idx, (img_file, depth_file) in enumerate(
                    zip(image_files, depth_files)
                ):
                    entries.append(
                        {
                            "id": idx,
                            "image_path": os.path.join("final", scene, img_file),
                            "depth_path": os.path.join(
                                "depth", scene, depth_file
                            ),  # 改 key
                        }
                    )
                metadata[scene] = entries
        return metadata

    def __len__(self):
        if self.stack_scene_depth:
            return len(self.scenes)
        else:
            return len(self.samples)

    def __getitem__(self, idx):

        if self.stack_scene_depth:
            scene = self.scenes[idx]
            frames = self.metadata[scene]
            imgs = []
            depths = []
            for entry in frames:
                img_path = os.path.join(self.data_root, entry["image_path"])
                depth_path = os.path.join(
                    self.data_root, entry["depth_path"]
                )  # 原 disparity 文件，这里存的是 depth

                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)  # HWC uint8
                imgs.append(img_np)

                depth_np = depth_read(depth_path).astype("float32")  # HW
                depths.append(depth_np)

            # 转成 TCHW / T1HW
            imgs = np.stack(imgs, axis=0)  # T, H, W, C
            imgs = (
                torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous() / 255.0
            )  # T, C, H, W

            depths = np.stack(depths, axis=0)  # T, H, W
            depths = torch.from_numpy(depths).unsqueeze(
                1).contiguous()  # T, 1, H, W

            # imgs = F.interpolate(
            #     imgs.float(), size=self.target_size, mode="bilinear")
            # depths = F.interpolate(
            #     depths.float(), size=self.target_size, mode="bilinear")
            # eval_mask
            # depths = torch.clamp(
            #     depths, min=self.min_depth, max=self.max_depth)

        else:
            entry = self.samples[idx]
            scene = entry["scene"]
            img_path = os.path.join(self.data_root, entry["image_path"])
            depth_path = os.path.join(self.data_root, entry["depth_path"])

            # 读图
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)  # HWC
            img_tensor = torch.from_numpy(img_np).permute(
                2, 0, 1).float() / 255.0  # C,H,W
            imgs = img_tensor.unsqueeze(0)  # T=1 -> 1,C,H,W
            # img_tensor = F.interpolate(
            #     img_tensor, size=self.target_size, mode="bilinear")

            # 读深度
            depth_np = depth_read(depth_path).astype("float32")  # H,W
            depths = torch.from_numpy(
                depth_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            # depth_tensor = F.interpolate(
            #     depth_tensor, size=self.target_size, mode="bilinear")

            # depth_tensor = torch.clamp(
            #     depth_tensor, min=self.min_depth, max=self.max_depth)

        imgs = F.interpolate(imgs, size=self.target_size, mode="bilinear")
        depths = F.interpolate(depths, size=self.target_size, mode="nearest")

        sample = {
            'sample_idx': torch.tensor(idx),
            "images": imgs,
            "depth_raw_linear": depths,
            "scene_name": scene
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return f"SintelDepthDataset(root='{self.data_root}', scenes={len(self.scenes)})"

