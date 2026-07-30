import io
import os
import random
import time
from math import ceil, floor

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor,
    q,
    dim=None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out=None,
):
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        raise ValueError

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0]
            for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


def read_numpy_file(
    numpy_file,
):
    """
    return a numpy array given the file path
    """

    ff = np.load(numpy_file)
    return ff


def read_image_file(
    image_file,
):
    """
    return a uint8 numpy array given the file path
    """
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def depth2vis(depth, maxthresh=50):
    depthvis = np.clip(depth, 0, maxthresh)
    # depthvis = depthvis/maxthresh*255
    depthvis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    # depthvis = depthvis.astype(np.uint8)
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape + (1,)), (1, 1, 3))

    return depthvis


class TartanAirDepthTransform:
    def __init__(
        self,
        size,
        random_flip,
        norm_type,
        truncnorm_min=0.02,
    ) -> None:
        self.size = size
        self.random_flip = random_flip
        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        self.d_max = 50

    def __call__(self, image, depth):
        # resize
        image = transforms.functional.resize(
            image, self.size, interpolation=Image.BILINEAR
        )

        depth = torch.from_numpy(depth).unsqueeze(1)
        depth = torch.clamp(depth, 0, self.d_max)
        depth = F.interpolate(depth, size=self.size, mode="nearest").squeeze(1)

        # random flip
        if self.random_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            depth = torch.flip(depth, [-1])

        torch.quantile = torch_quantile
        # depth
        if self.norm_type == "instnorm":
            dmin = depth.min()
            dmax = depth.max()
            # depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == "truncnorm":
            # refer to Marigold
            dmin = torch.quantile(depth, self.truncnorm_min)
            dmax = torch.quantile(depth, self.truncnorm_max)
            # depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == "perscene_norm":
            pass
            # depth_norm = ((depth / self.d_max) - 0.5) * 2.0
        elif self.norm_type == "disparity":
            disparity = 1 / depth
            disparity_min = disparity.min()
            disparity_max = disparity.max()
            # disparity_norm = ((disparity - disparity_min) /
            #                   (disparity_max - disparity_min + 1e-5))
            depth_norm = disparity
        elif self.norm_type == "trunc_disparity":
            disparity = 1 / depth
            disparity_min = torch.quantile(disparity, self.truncnorm_min)
            disparity_max = torch.quantile(disparity, self.truncnorm_max)
            disparity_norm = (disparity - disparity_min) / (
                disparity_max - disparity_min + 1e-5
            )
            depth_norm = disparity_norm
        else:
            raise TypeError(
                f"Not supported normalization type: {self.norm_type}. ")

        depth_norm = depth_norm.clip(0, 1)
        depth_raw = depth.clone().unsqueeze(1).repeat(1, 3, 1, 1)
        depth = depth_norm.unsqueeze(1).repeat(1, 3, 1, 1)

        return image, depth, depth_raw


class TartanAir_VID_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        random_flip,
        norm_type,
        resolution=(480, 640),
        truncnorm_min=0.02,
        max_num_frame=None,
        min_num_frame=None,
        max_sample_stride=None,
        min_sample_stride=None,
        start=0,
        train_ratio=1.0,
    ):

        self.data_list = []

        for root, dirs, files in os.walk(data_dir):
            for _dir in dirs:
                if not _dir.startswith("depth_"):
                    continue
                res_dict = {
                    "scene": os.path.relpath(root, data_dir),
                    "cam": "left" if "left" in _dir else "right",
                }
                depth_dir = os.path.join(root, _dir)
                rgb_dir = os.path.join(root, _dir.replace("depth_", "image_"))
                res_dict["rgb_dir"] = rgb_dir
                res_dict["depth_dir"] = depth_dir

                self.data_list.append(res_dict)

        self.data_list.sort(key=lambda x: x["scene"])
        self.data_list = self.data_list[start:]

        # print(f"Resizing to {resolution}")
        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.transform = TartanAirDepthTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min
        )
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))
        if train_ratio < 1.0:
            origin_len = len(self.data_list)
            self.data_list = self.data_list[:int(origin_len * train_ratio)]
            print(
                f"TartanAir_VID_Dataset use {int(origin_len * train_ratio)} samples instead of {origin_len}...")
        else:
            print(
                f"TartanAir_VID_Dataset use origin {len(self.data_list)} samples...")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        try:
            scene_name = self.data_list[idx]["scene"]
            cam = self.data_list[idx]["cam"]
            rgb_dir = self.data_list[idx]["rgb_dir"]
            depth_dir = self.data_list[idx]["depth_dir"]

            img_path_list = []
            depth_path_list = []
            for rgb_file in os.listdir(rgb_dir):
                if not rgb_file.endswith(".png"):
                    continue
                _rgb_path = os.path.join(rgb_dir, rgb_file)
                img_path_list.append(_rgb_path)

            for depth_file in os.listdir(depth_dir):
                if not depth_file.endswith(".npy"):
                    continue
                _depth_path = os.path.join(depth_dir, depth_file)
                depth_path_list.append(_depth_path)
            img_path_list = sorted(img_path_list)
            depth_path_list = sorted(depth_path_list)
            assert len(img_path_list) == len(depth_path_list)

            # Handling index
            total_frames = len(img_path_list)
            if total_frames < self.min_num_frame:
                raise ValueError(
                    f"Not enough frames for scene {scene_name}. Need at least {self.min_num_frame} frames, but got {total_frames}."
                )

            _sample_idx = None
            for _ in range(1000):
                _stride = random.choice(self.strides)
                _num_frames = random.choice(self.num_frames)
                if _num_frames % 4 != 1:
                    continue
                _total_frames_req = _stride * (_num_frames - 1) + 1
                if _total_frames_req > total_frames:
                    continue
                start_idx = random.randint(0, total_frames - _total_frames_req)
                end_idx = start_idx + _total_frames_req
                _sample_idx = list(range(start_idx, end_idx, _stride))
                break
            if _sample_idx is None:
                raise ValueError(
                    f"Fail to sample frames for scene {scene_name}.")

            _sample_img_path = [img_path_list[i] for i in _sample_idx]
            _sample_depth_path = [depth_path_list[i] for i in _sample_idx]

            img_list = []
            depth_list = []
            for _idx, (_img_path, _depth_path) in enumerate(
                zip(_sample_img_path, _sample_depth_path)
            ):
                img_np = read_image_file(_img_path)
                dist = read_numpy_file(_depth_path)
                img_list.append(img_np)
                depth_list.append(dist)

            img_np = np.array(img_list).astype(np.float32) / 255.0
            depth_np = np.array(depth_list)

            image = torch.from_numpy(img_np).permute(0, 3, 1, 2)

            if torch.isnan(image).any() or torch.isinf(image).any():
                print(
                    f"Error loading data at index {idx}: image is nan or inf")
                return self.__getitem__(idx + 1)
            # print(
            #     f"Shape of img_np: {img_np.shape}, range: {img_np.min()} - {img_np.max()}, dtype: {img_np.dtype}"
            # )
            # print(
            #     f"Shape of depth_np: {depth_np.shape}, range: {depth_np.min()} - {depth_np.max()}, dtype: {depth_np.dtype}"
            # )
            # print(
            #     f"Shape of image: {image.shape}, range: {image.min()} - {image.max()}, dtype: {image.dtype}")
            image, depth, depth_raw = self.transform(image, depth_np)

            if torch.isnan(depth).any() or torch.isinf(depth).any():
                print(
                    f"Error loading data at index {idx}: depth is nan or inf")
                return self.__getitem__(idx + 1)
            return {
                "sample_idx": torch.tensor(idx),
                "images": image,
                "disparity": depth,
                'depth_raw': depth_raw,
                "image_path": rgb_dir,
                "depth_path": depth_dir,
                "scene_name": scene_name,
                "cam": cam,
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # In case of error, return a random sample
            return self.__getitem__(idx+1)

