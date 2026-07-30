# Author: Bingxin Ke
# Last modified: 2024-02-08


import torch

from .base_depth_dataset import (BaseDepthDataset, DatasetMode,
                                 DepthFileNameMode)


class NYUDataset(BaseDepthDataset):
    global_count = 0

    def __init__(
        self,
        eigen_valid_mask: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            # NYUv2 dataset parameter
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=True,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

        self.eigen_valid_mask = eigen_valid_mask

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode NYU depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth)
        # print(f"Sample {self.global_count}: valid_mask mean: {valid_mask.float().mean()}")
        # Eigen crop for evaluation
        if self.eigen_valid_mask:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            eval_mask[45:471, 41:601] = 1
            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask


