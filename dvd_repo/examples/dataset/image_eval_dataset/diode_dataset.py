# Author: Bingxin Ke
# Last modified: 2024-02-26

import os
import tarfile
from io import BytesIO

import numpy as np
import torch

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode


class DIODEDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # DIODE data parameter
            min_depth=0.6,
            max_depth=350,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        # self.filenames = self.filenames[kwargs['start']:]

    def _read_npy_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            fileobj = self.tar_obj.extractfile("./" + rel_path)
            npy_path_or_content = BytesIO(fileobj.read())
        else:
            npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        data = np.load(npy_path_or_content).squeeze()[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        return depth

    def _get_data_path(self, index):
        return self.filenames[index]

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, depth_rel_path, mask_rel_path = self._get_data_path(
            index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=None
            )
            rasters.update(depth_data)

            # valid mask
            mask = self._read_npy_file(mask_rel_path).astype(bool)
            mask = torch.from_numpy(mask).bool()
            rasters["valid_mask_raw"] = mask.clone()
            rasters["valid_mask_filled"] = mask.clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    config_path = 'configs/data_diode_all.yaml'
    config = OmegaConf.load(config_path)
    diode_dataset = DIODEDataset(mode=DatasetMode.EVAL, **config)
    dataloader = DataLoader(diode_dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        print(data.keys())
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"{k}: {v.shape}, range: {v.min()}, {v.max()}, dtype: {v.dtype} ")
            else:
                print(k, v)
