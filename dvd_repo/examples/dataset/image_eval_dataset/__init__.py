# Author: Bingxin Ke
# Last modified: 2024-03-30

import os

from .base_depth_dataset import (BaseDepthDataset, DatasetMode,  # noqa: F401
                                 get_pred_name)
from .diode_dataset import DIODEDataset
from .eth3d_dataset import ETH3DDataset
from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .scannet_dataset import ScanNetDataset

dataset_name_class_dict = {
    "nyu_v2": NYUDataset,
    "kitti": KITTIDataset,
    "eth3d": ETH3DDataset,
    "scannet": ScanNetDataset,
    "diode": DIODEDataset,
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=os.path.join(base_data_dir,cfg_data_split.filename),
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
