import io
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def read_image_file(image_file):
    """
    return a uint8 numpy array given the file path
    """
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError(f"Failed to load image: {image_file}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def read_depth_file(depth_file):
    """
    Bonn Dataset / TUM Dataset format:
    16-bit PNG, factor = 5000.0 (5000 pixel val = 1 meter)
    """
    # cv2.IMREAD_UNCHANGED is crucial for 16-bit images
    depth_png = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    if depth_png is None:
        raise ValueError(f"Failed to load depth: {depth_file}")

    # Convert to float and scale
    depth = depth_png.astype(np.float32) / 5000.0

    # Handle invalid depth (0 value usually means missing data)
    # We can keep it as 0 here and clamp it later, or set it to a far value.
    # For safety with 1/x calculation, we rely on the Transform to clamp.
    return depth


def depth2vis(depth, maxthresh=5):
    # Adjusted maxthresh for indoor scenes (Bonn is usually < 5m relevant)
    depthvis = np.clip(depth, 0, maxthresh)
    depthvis = (depthvis - depthvis.min()) / \
        (depthvis.max() - depthvis.min() + 1e-5) * 255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape + (1,)), (1, 1, 3))
    return depthvis


class Bonn_VID_Dataset(Dataset):
    def __init__(
        self,
        data_root,
        resolution=(480, 640),
        min_depth=1e-3,
        max_depth=10,
        max_num_frame=None,
        min_num_frame=None,
        max_sample_stride=None,
        min_sample_stride=None,
        start=0,
        split='test',
        **kwargs

    ):

        self.data_list = []
        self.split = split
        self.test = split == 'test'
        self.min_depth = min_depth
        self.max_depth = max_depth
        # Standardizing folder names for Bonn Dataset
        # Assuming structure: root/scene/depth_... and root/scene/image_...
        for root, dirs, files in os.walk(data_root):
            for _dir in dirs:
                # Bonn folder names usually imply "depth"
                if "depth" not in _dir:
                    continue

                res_dict = {
                    "scene": os.path.relpath(root, data_root),
                    # Determine camera side if applicable, otherwise default
                    "cam": "mono",
                }

                depth_dir = os.path.join(root, _dir)

                # replace depth with image
                rgb_dir_name = _dir.replace("depth", "rgb")

                rgb_dir = os.path.join(root, rgb_dir_name)

                # Verify paths exist
                if not os.path.exists(rgb_dir):
                    # print(f"Skipping {depth_dir}, corresponding RGB {rgb_dir} not found.")
                    continue

                res_dict["rgb_dir"] = rgb_dir
                res_dict["depth_dir"] = depth_dir

                self.data_list.append(res_dict)

        self.data_list.sort(key=lambda x: x["scene"])
        self.data_list = self.data_list[start:]

        # print(f"Resizing to {resolution}")
        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def associate_data(rgb_list, depth_list, max_diff=0.02):
        """
        输入: 
            rgb_list: list of strings (filenames), e.g., ["1548.123.png", ...]
            depth_list: list of strings (filenames)
            max_diff: maximum time difference in seconds
        输出:
            matches_rgb: list of full filenames
            matches_depth: list of full filenames
        """
        def get_timestamp(fname):
            basename = os.path.basename(fname)
            root = os.path.splitext(basename)[0]
            return float(root)

        rgb_timestamps = {get_timestamp(f): f for f in rgb_list}
        depth_timestamps = {get_timestamp(f): f for f in depth_list}

        rgb_keys = sorted(rgb_timestamps.keys())
        depth_keys = sorted(depth_timestamps.keys())

        matches_rgb = []
        matches_depth = []

        depth_idx = 0
        num_depth = len(depth_keys)

        for t_rgb in rgb_keys:
            best_dist = float('inf')
            best_t_depth = -1

            while depth_idx < num_depth - 1:
                t_depth_curr = depth_keys[depth_idx]
                t_depth_next = depth_keys[depth_idx + 1]

                dist_curr = abs(t_depth_curr - t_rgb)
                dist_next = abs(t_depth_next - t_rgb)

                if dist_next < dist_curr:
                    depth_idx += 1
                else:
                    break

            t_depth_best = depth_keys[depth_idx]
            if abs(t_depth_best - t_rgb) < max_diff:
                matches_rgb.append(rgb_timestamps[t_rgb])
                matches_depth.append(depth_timestamps[t_depth_best])

        return matches_rgb, matches_depth

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        try:
            scene_name = self.data_list[idx]["scene"]
            cam = self.data_list[idx]["cam"]
            rgb_dir = self.data_list[idx]["rgb_dir"]
            depth_dir = self.data_list[idx]["depth_dir"]

            img_path_list = []
            depth_path_list = []

            # --- FILE EXTENSION CHECK ---
            for rgb_file in os.listdir(rgb_dir):
                if not rgb_file.endswith(".png"):
                    continue
                _rgb_path = os.path.join(rgb_dir, rgb_file)
                img_path_list.append(_rgb_path)

            for depth_file in os.listdir(depth_dir):
                # Bonn Depth is .png (16-bit), NOT .npy
                if not depth_file.endswith(".png"):
                    continue
                _depth_path = os.path.join(depth_dir, depth_file)
                depth_path_list.append(_depth_path)

            img_path_list, depth_path_list = self.associate_data(
                img_path_list, depth_path_list, max_diff=0.02)
            # print(f"img_path_list: {img_path_list[:20]})")
            # print(f"depth_path_list: {depth_path_list[:20]}")
            # img_path_list = sorted(img_path_list)
            # depth_path_list = sorted(depth_path_list)

            # Basic syncing check (filename matching)
            # Sometimes RGB has more frames or slight offset,
            # ideally we match by timestamp in filename, but assuming sorted index alignment here:
            assert len(img_path_list) == len(
                depth_path_list), "RGB and depth path lists should have the same length"
            # if len(img_path_list) != len(depth_path_list):
            #      # Prune to min length if mismatch
            #      min_len = min(len(img_path_list), len(depth_path_list))
            #      img_path_list = img_path_list[:min_len]
            #      depth_path_list = depth_path_list[:min_len]

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
                # if _num_frames % 4 != 1:
                #     continue
                _total_frames_req = _stride * (_num_frames - 1) + 1

                if _total_frames_req > total_frames:
                    continue
                if not self.test:
                    start_idx = random.randint(
                        0, total_frames - _total_frames_req)
                else:
                    start_idx = 0

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
                dist = read_depth_file(_depth_path)
                img_list.append(img_np)
                depth_list.append(dist)

            img_np = np.array(img_list).astype(np.float32) / 255.0
            depth_np = np.array(depth_list)

            image = torch.from_numpy(img_np).permute(0, 3, 1, 2)
            depth = torch.from_numpy(depth_np).unsqueeze(1).repeat(1, 3, 1, 1)

            return {
                "sample_idx": torch.tensor(idx),
                "images": image,
                "disparity": depth,
                'depth_raw': depth,
                "image_path": rgb_dir,
                "depth_path": depth_dir,
                "scene_name": scene_name,
                "cam": cam,
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # traceback.print_exc() # Useful for debugging
            return self.__getitem__(idx+1)


