# Author: Bingxin Ke
# Last modified: 2024-02-08

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode
import torch

class ScanNetDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ScanNet data parameter
            min_depth=1e-3,
            max_depth=10,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode ScanNet depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    config_path = 'configs/data_scannet_val.yaml'
    config = OmegaConf.load(config_path)
    scannet_dataset = ScanNetDataset(mode=DatasetMode.EVAL,**config)
    dataloader = DataLoader(scannet_dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        print(data.keys())
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"{k}: {v.shape}, range: {v.min()}, {v.max()}, dtype: {v.dtype} ")
            else:
                print(k, v)
