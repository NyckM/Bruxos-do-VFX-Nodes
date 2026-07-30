import os

from .bonn_vid_dataset import Bonn_VID_Dataset
from .kitti_vid_dataset import KITTI_VID_Dataset
from .nyuv2_dataset import NYUv2Dataset
from .scannet_dataset import Scannet_VID_Dataset
from .sintel_dataset import SintelDataset
from .tartanair_vid_dataset import TartanAir_VID_Dataset
from .vkitti_vid_dataset import VKITTI_VID_Dataset

__all__ = [
    "KITTI_VID_Dataset",
    "TartanAir_VID_Dataset",
    "VKITTI_VID_Dataset",
    "Bonn_VID_Dataset",
    "NYUV2Dataset",
    "SintelDataset",
    'Scannet_VID_Dataset'
]

dataset_name_class_dict = {
    'kitti': KITTI_VID_Dataset,
    'bonn': Bonn_VID_Dataset,
    'sintel': SintelDataset,
    'scannet': Scannet_VID_Dataset
}


def get_vid_eval_dataset(
    cfg_data_split, base_data_dir: str, **kwargs
):
    if cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            data_root=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError
    
    return dataset

