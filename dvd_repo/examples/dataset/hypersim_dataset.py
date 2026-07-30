import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def hypersim_distance_to_depth(npyDistance):
    intWidth = 1024
    intHeight = 768
    fltFocal = 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal

    return npyDepth


def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float64),
                       np.arange(0, W, dtype=np.float64), indexing='ij')
    meshgrid = np.stack((x, y))
    ones = np.ones((1, H*W), dtype=np.float64)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

# Some Hypersim normals are not properly oriented towards the camera.
    # The align_normals and creat_uv_mesh functions are from GeoWizard
    # https://github.com/fuxiao0719/GeoWizard/blob/5ff496579c6be35d9d86fe4d0760a6b5e6ba25c5/geowizard/training/dataloader/file_io.py#L79


def align_normals(normal, depth, K, H, W):
    '''
    Orientation of surface normals in hypersim is not always consistent
    see https://github.com/apple/ml-hypersim/issues/26
    '''
    # inv K
    K = np.array([[K[0],    0, K[2]],
                  [0, K[1], K[3]],
                  [0,    0,    1]])
    inv_K = np.linalg.inv(K)
    # reprojection depth to camera points
    xy = creat_uv_mesh(H, W)
    points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
    points = depth * points
    points = points.transpose((1, 2, 0))
    # align normal
    orient_mask = np.sum(normal * points, axis=2) < 0
    normal[orient_mask] *= -1
    return normal


class HypersimImageDepthNormalTransform:
    def __init__(self, size, random_flip, norm_type, truncnorm_min=0.02, align_cam_normal=False) -> None:
        self.size = size
        self.random_flip = random_flip
        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        self.d_max = 65
        self.align_cam_normal = align_cam_normal

    def to_tensor_and_resize_normal(self, normal):
        # to tensor
        normal = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0)
        # resize
        normal = F.interpolate(normal, size=self.size,
                               mode='nearest').squeeze()
        # shape = 3 * 768 * 1024
        return normal

    def __call__(self, image, depth, normal):
        # convert the inward normals to outward normals
        normal[:, :, 0] *= -1
        if self.align_cam_normal:
            # align normal towards camera
            H, W = normal.shape[:2]
            normal = align_normals(
                normal, depth, [886.81, 886.81, W/2, H/2], H, W)

        # resize
        image = transforms.functional.resize(
            image, self.size, interpolation=Image.BILINEAR)

        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=self.size, mode='nearest').squeeze()

        normal = self.to_tensor_and_resize_normal(normal)

        # random flip
        if self.random_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            depth = torch.flip(depth, [-1])
            normal = torch.flip(normal, [-1])
            # Flip x-component of normal map
            normal[0, :, :] = - normal[0, :, :]

        # to_tensor and normalize
        # image
        image = transforms.ToTensor()(image)
        # image = transforms.Normalize([0.5], [0.5])(image)

        # depth
        if self.norm_type == 'instnorm':
            dmin = depth.min()
            dmax = depth.max()
            # depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'truncnorm':
            # refer to Marigold
            dmin = torch.quantile(depth, self.truncnorm_min)
            dmax = torch.quantile(depth, self.truncnorm_max)
            # depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'perscene_norm':
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
            disparity_norm = ((disparity - disparity_min) /
                              (disparity_max - disparity_min + 1e-5))
            depth_norm = disparity_norm
        else:
            raise TypeError(
                f"Not supported normalization type: {self.norm_type}. ")

        depth_norm = depth_norm.clip(0, 1)
        depth = depth_norm.unsqueeze(0).repeat(3, 1, 1)

        # normal
        normal = normal.clip(-1, 1)

        return image, depth, normal


class HypersimDataset(Dataset):
    def __init__(self, data_dir,  random_flip, norm_type, resolution=(480, 720),
                 truncnorm_min=0.02, align_cam_normal=False, split="train", start=0, train_ratio=1.0):

        self.data_list = []
        split_dir = os.path.join(data_dir, split)

        # 搜索所有 tonemap.jpg
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.endswith("tonemap.jpg"):
                    img = os.path.join(root, file)
                    dep = img.replace("final_preview", "geometry_hdf5").replace(
                        "tonemap.jpg", "depth_meters.hdf5")
                    nor = img.replace("final_preview", "geometry_hdf5").replace(
                        "tonemap.jpg", "normal_cam.hdf5")
                    self.data_list.append((img, dep, nor))
        self.data_list.sort()
        self.data_list = self.data_list[start:]
        # print(
        #     f"Total {len(self.data_list)} samples found for {split} set, first ten samples: {self.data_list[:10]}")
        # # compute new resolution
        # w, h = Image.open(self.data_list[0][0]).size
        # if h > w:
        #     new_w = resolution
        #     new_h = int(resolution * h / w)
        # else:
        #     new_h = resolution
        #     new_w = int(resolution * w / h)
        # print(f"Resizing to {resolution}")
        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.transform = HypersimImageDepthNormalTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min, align_cam_normal
        )
        if train_ratio < 1.0:
            origin_len = len(self.data_list)
            self.data_list = self.data_list[:int(origin_len * train_ratio)]
            print(
                f"Hypersim use {int(origin_len * train_ratio)} samples instead of {origin_len}...")
        else:
            print(
                f"Hypersim use origin {len(self.data_list)} samples...")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        try:
            img_path, dep_path, nor_path = self.data_list[idx]

            image = Image.open(img_path).convert("RGB")

            # load depth (distance → depth)
            with h5py.File(dep_path, 'r') as f:
                dist = np.array(f["dataset"])

            depth = hypersim_distance_to_depth(dist)
            raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            raw_depth = F.interpolate(raw_depth, size=(
                self.new_h, self.new_w), mode='nearest').squeeze()
            raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

            # load normals
            with h5py.File(nor_path, 'r') as f:
                normal = np.array(f["dataset"])

            image, depth, normal = self.transform(image, depth, normal)
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(
                    f"Error loading data at index {idx}: image is nan or inf")
                return self.__getitem__(idx+1)
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                print(
                    f"Error loading data at index {idx}: depth is nan or inf")
                return self.__getitem__(idx+1)
            return {
                "sample_idx": torch.tensor(idx),
                "images": image.unsqueeze(0),
                "disparity": depth.unsqueeze(0),
                'depth': raw_depth.unsqueeze(0),
                "normal_values": normal,
                "image_path": img_path,
                "depth_path": dep_path,
                "normal_path": nor_path,
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # In case of error, return a random sample
            return self.__getitem__(idx+1)


def collate_fn_hypersim(batch):
    images = torch.stack([b["images"] for b in batch]).float()
    disparity = torch.stack([b["disparity"] for b in batch]).float()
    normal_values = torch.stack([b["normal_values"] for b in batch]).float()
    depth = torch.stack([b["depth"] for b in batch]).float()
    return {
        "images": images,
        "disparity": disparity,
        'depth': depth,
        # "normal_values
        "normal_values": normal_values,
        "image_paths": [b["image_path"] for b in batch],
        "depth_paths": [b["depth_path"] for b in batch],
        "normal_paths": [b["normal_path"] for b in batch],
    }


if __name__ == "__main__":
    import matplotlib.cm as cm
    import torch
    import torchvision.transforms.functional as TF
    import torchvision.utils as vutils
    from omegaconf import OmegaConf
    from PIL import Image

    args = OmegaConf.load("configs/hypersim.yaml")

    dataset = HypersimDataset(
        data_dir=args.train_data_dir_hypersim,
        resolution=args.resolution_hypersim,
        random_flip=args.random_flip,
        norm_type=args.norm_type,
        truncnorm_min=args.truncnorm_min,
        align_cam_normal=args.align_cam_normal,
        split="train",
    )
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn_hypersim,
    )
    dir = 'debug'
    os.makedirs(dir, exist_ok=True)
    print(f"Length of dataset {len(dataset)}")
    for step, batch in enumerate(dataloader):
        if step >= 50:
            break
        print(f"Step {step}:")
        print(
            f"  images: {batch['images'].shape} range {batch['images'].min().item()} - {batch['images'].max().item()}")
        print(
            f"Depth: {batch['depth'].shape} range {batch['depth'].min().item()} - {batch['depth'].max().item()}")
        # 取第一张
        img = batch["images"][0, 0]      # [3, H, W]
        depth = batch["depth"][0, 0]     # [3, H, W]（如果是3通道）

        # --------------------
        # 保存 RGB
        # --------------------
        img_to_save = img.clamp(0, 1)
        img_pil = TF.to_pil_image(img_to_save.cpu())
        img_pil.save(os.path.join(dir, f"step_{step}_rgb.png"))

        # --------------------
        # 保存 Depth（归一化后再存）
        # --------------------
        depth_single = depth[0]  # 如果是3通道，取第一通道

        depth_min = depth_single.min()
        depth_max = depth_single.max()
        depth_norm = (depth_single - depth_min) / \
            (depth_max - depth_min + 1e-8)

        # 转 numpy
        depth_np = depth_norm.cpu().numpy()

        # 用 Spectral colormap 映射成 RGB
        depth_color = cm.Spectral(depth_np)[:, :, :3]  # 去掉alpha通道

        # 转成 0–255 uint8
        depth_color = (depth_color * 255).astype(np.uint8)

        depth_pil = Image.fromarray(depth_color)
        depth_pil.save(os.path.join(dir, f"step_{step}_depth_spectral.png"))
        # print(
        #     f"  disparity: {batch['disparity'].shape}, range {batch['disparity'].min().item()} - {batch['disparity'].max().item()}")
        # print(
        #     f"  normal_values: {batch['normal_values'].shape}, range {batch['normal_values'].min().item()} - {batch['normal_values'].max().item()}")
