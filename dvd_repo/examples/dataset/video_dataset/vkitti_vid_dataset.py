import os
import random
from math import ceil, floor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


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


class VKITTITransform:
    '''Crop images to KITTI benchmark size and randomly flip. '''

    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    def __init__(self, random_flip):
        self.random_flip = random_flip

    def __call__(self, image, depth):

        # Resize images if necessary
        image = F.interpolate(image, size=(
            self.KB_CROP_HEIGHT, self.KB_CROP_WIDTH), mode='bilinear')  # Adjust depth to new size

        depth = F.interpolate(depth, size=(
            self.KB_CROP_HEIGHT, self.KB_CROP_WIDTH), mode='nearest')  # Adjust depth to new size

        _h, _w = image.shape[-2:]

        # Crop to exact dimensions
        top = int(_h - self.KB_CROP_HEIGHT)
        left = int((_w - self.KB_CROP_WIDTH) / 2)
        # image = image.crop(
        #     (left, top, left+self.KB_CROP_WIDTH, top+self.KB_CROP_HEIGHT))
        image = image[top:top+self.KB_CROP_HEIGHT,
                      left:left+self.KB_CROP_WIDTH]
        depth = depth[top:top+self.KB_CROP_HEIGHT,
                      left:left+self.KB_CROP_WIDTH]

        # Random horizontal flipping
        if self.random_flip and torch.rand(1) > 0.5:
            # image = TF.hflip(image)
            image = torch.flip(image, [-1])
            # Flip depth tensor on width dimension
            depth = torch.flip(depth, [-1])

        return image, depth


class VKITTI_VID_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 norm_type='truncnorm',
                 truncnorm_min=0.02,
                 max_num_frame=None,
                 min_num_frame=None,
                 max_sample_stride=None,
                 min_sample_stride=None,
                 train_ratio=1.0,
                 ):
        """
        Args:
            root_dir (string): Directory with all the images and depths.
            transform (callable, optional): Optional transform to be applied on a sample.
            norm_type (str, optional): Normalization type.
            truncnorm_min (float, optional): Minimum value for truncated normal distribution.
        """
        self.root_dir = root_dir
        self.transform = VKITTITransform(random_flip=True)
        # self.scenes = ['01', '02', '06', '18', '20']
        self.scenes = ['02', '06', '18', '20']
        self.conditions = [
            '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
            'fog', 'morning', 'overcast', 'rain', 'sunset'
        ]
        self.cameras = ['0', '1']

        self.d_min = 1e-5
        self.d_max = 80

        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        self.data_list = []

        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))

        # Video logic
        for scene in self.scenes:
            for cond in self.conditions:
                for cam in self.cameras:
                    res_dict = {
                        'scene': f"{scene}/{cond}",
                        'cam': "{cam}"
                    }
                    _rgb_dir = os.path.join(
                        self.root_dir, f'Scene{scene}/{cond}/frames/rgb/Camera_{cam}')
                    _depth_dir = os.path.join(
                        self.root_dir, f'Scene{scene}/{cond}/frames/depth/Camera_{cam}')
                    res_dict['rgb_dir'] = _rgb_dir
                    res_dict['depth_dir'] = _depth_dir

                    self.data_list.append(res_dict)

        self.data_list.sort(key=lambda x: x['scene']+x['cam'])
        if train_ratio < 1.0:
            origin_len = len(self.data_list)
            self.data_list = self.data_list[:int(origin_len * train_ratio)]
            print(
                f"VKITTI_VID_Dataset use {int(origin_len * train_ratio)} samples instead of {origin_len}...")
        else:
            print(
                f"VKITTI_VID_Dataset use origin {len(self.data_list)} samples...")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            idx = idx % self.__len__()
            example = {}

            # Load data
            scene_name = self.data_list[idx]["scene"]
            cam = self.data_list[idx]["cam"]
            rgb_dir = self.data_list[idx]["rgb_dir"]
            depth_dir = self.data_list[idx]["depth_dir"]

            img_path_list = []
            depth_path_list = []
            for rgb_file in os.listdir(rgb_dir):
                if not rgb_file.endswith(".jpg"):
                    continue
                _rgb_path = os.path.join(rgb_dir, rgb_file)
                img_path_list.append(_rgb_path)

            for depth_file in os.listdir(depth_dir):
                if not depth_file.endswith(".png"):
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
                image = cv2.imread(_img_path)
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                image = torch.from_numpy(image).permute(2, 0, 1)

                dist = depth = cv2.imread(_depth_path, cv2.IMREAD_ANYCOLOR |
                                          cv2.IMREAD_ANYDEPTH)
                depth = depth / 100  # cm -> m
                depth = torch.from_numpy(depth).unsqueeze(
                    0)  # (1, 1, h, w)
                # image, depth = self.transform(image, depth)
                img_list.append(image)
                depth_list.append(depth)

            _imgs = torch.stack(img_list)
            _depths = torch.stack(depth_list)
            # print(f"Shape of _imgs: {_imgs.shape}")
            # print(f"Shape of _depths: {_depths.shape}")
            # _imgs, _depths = self.transform(_imgs, _depths)

            # Transform data simultaneously if needed
            if self.transform:
                _imgs, _depths = self.transform(_imgs, _depths)

            # print(f"After transformation, shape of _imgs: {_imgs.shape}")
            # print(f"After transformation, shape of _depths: {_depths.shape}")
            # Get mask given depth
            # depth = depth.unsqueeze(0)  # (1, h, w)
            # print(f"depth range for idx {idx}: {depth.min()} - {depth.max()}")
            valid_mask_raw = self._get_valid_mask(_depths).clone()
            sky_mask_raw = self._get_sky_mask(_depths).clone()
            example['valid_mask_values'] = valid_mask_raw
            example['sky_mask_values'] = sky_mask_raw

            if torch.isnan(_imgs).any() or torch.isinf(_imgs).any():
                print(
                    f"Error loading data at index {idx}: image is nan or inf")
                return self.__getitem__(idx+1)
            if torch.isnan(_depths).any() or torch.isinf(_depths).any():
                print(
                    f"Error loading data at index {idx}: depth is nan or inf")
                return self.__getitem__(idx)

            torch.quantile = torch_quantile

            # depth
            depth = _depths
            valid_mask = valid_mask_raw & (depth > 0)
            if self.norm_type == 'instnorm':
                dmin = depth[valid_mask].min()
                dmax = depth[valid_mask].max()
                depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5))
            elif self.norm_type == 'truncnorm':
                # refer to Marigold
                dmin = torch.quantile(depth[valid_mask], self.truncnorm_min)
                dmax = torch.quantile(depth[valid_mask], self.truncnorm_max)
                depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5))
            elif self.norm_type == 'perscene_norm':
                depth_norm = ((depth / self.d_max))
            elif self.norm_type == "disparity":
                disparity = 1 / depth
                disparity_min = disparity[valid_mask].min()
                disparity_max = disparity[valid_mask].max()
                disparity_norm = ((disparity - disparity_min) /
                                  (disparity_max - disparity_min + 1e-5))
                depth_norm = disparity_norm
            elif self.norm_type == "trunc_disparity":
                disparity = 1 / depth
                # _98_percentage = torch.quantile(disparity[valid_mask], 0.98)
                # _02_percentage = torch.quantile(disparity[valid_mask], 0.02)
                # print(f"98th percentile of disparity: {_98_percentage}")
                # print(f"02th percentile of disparity: {_02_percentage}")
                disparity_min = torch.quantile(
                    disparity[valid_mask], self.truncnorm_min)
                disparity_max = torch.quantile(
                    disparity[valid_mask], self.truncnorm_max)
                disparity_norm = ((disparity - disparity_min) /
                                  (disparity_max - disparity_min + 1e-5))
                depth_norm = disparity_norm
            else:
                raise TypeError(
                    f"Not supported normalization type: {self.norm_type}. ")

            depth_norm = depth_norm.clip(0, 1)
            depth_norm = depth_norm.repeat(1, 3, 1, 1)  # (T, 3, h, w)
            example['images'] = _imgs
            example['disparity'] = depth_norm
            return example
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return self.__getitem__(idx+1)

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.d_min), (depth < self.d_max)
        ).bool()
        return valid_mask

    def _get_sky_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.d_min), (depth >= self.d_max)
        ).bool()
        return valid_mask


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
    return collated


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.io import write_video
    from torchvision.utils import save_image

    data_root = '/hpc2hdd/JH_DATA/share/jhe812/PrivateShareGroup/jhe812_d4p_datasets/d4p_datasets/vkitti'
    transform = VKITTITransform(random_flip=True)
    dataset = VKITTI_VID_Dataset(
        root_dir=data_root,
        norm_type='trunc_disparity',
        max_num_frame=81,
        min_num_frame=81,
        min_sample_stride=1,
        max_sample_stride=1,
    )
    print(f"Len of dataset: {len(dataset)}")
    os.makedirs("debug_vki_vid", exist_ok=True)

    for i, batch in enumerate(dataset):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"{k} shape, range, dtype: {v.shape}, {v.min()}, {v.max()}, {v.dtype}")
                _v = v.to(torch.float32)
                save_image(_v[0], os.path.join(
                    "debug_vki_vid", f"{k}_{i}.png"))
                _v = (_v.permute(0, 2, 3, 1)*255).to(torch.uint8)
                if _v.shape[-1] == 1:
                    _v = _v.repeat(1, 1, 1, 3)
                write_video(os.path.join(
                    "debug_vki_vid", f"{k}_{i}.mp4"), _v, fps=5)

            elif isinstance(v, str):
                print(f"{k}: {v}")
            elif v is None:
                print(f"{k}: None")
