# Author: Bingxin Ke
# Last modified: 2024-02-08

import os
import tarfile

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Resize

from .base_depth_dataset import (BaseDepthDataset, DatasetMode,
                                 DepthFileNameMode)


class ETH3DDataset(BaseDepthDataset):
    HEIGHT, WIDTH = 4032, 6048

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ETH3D data parameter
            min_depth=1e-5,
            max_depth=torch.inf,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        # Read special binary data: https://www.eth3d.net/documentation#format-of-multi-view-data-image-formats
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            binary_data = self.tar_obj.extractfile("./" + rel_path)
            binary_data = binary_data.read()

        else:
            depth_path = os.path.join(self.dataset_dir, rel_path)
            with open(depth_path, "rb") as file:
                binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        depth_decoded[depth_decoded == torch.inf] = 0.0

        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
        return depth_decoded


    
    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        if self.resize_to_hw is not None:
            
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        # return rasters
    
        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    config_path = 'configs/data_eth3d.yaml'
    config = OmegaConf.load(config_path)
    eth3d_dataset = ETH3DDataset(mode=DatasetMode.EVAL, **config)
    dataloader = DataLoader(eth3d_dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        print(data.keys())
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"{k}: {v.shape}, range: {v.min()}, {v.max()}, dtype: {v.dtype} ")
            else:
                print(k, v)


