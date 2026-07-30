# Core imports
import argparse
import json
import os

import numpy as np
import torch
# Accelerate & Config imports
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tabulate import tabulate
from tqdm import tqdm

# DiffSynth imports
from diffsynth import save_video
from diffsynth.util import metric
from diffsynth.util.alignment import (align_depth_least_square_video,
                                      depth2disparity, disparity2depth)
from diffsynth.util.metric import MetricTracker
# Dataset & Model imports
from examples.dataset.image_eval_dataset import (BaseDepthDataset, DatasetMode,
                                                 get_dataset)
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def generate_depth(model, args, input_rgb):
    """Generates depth predictions using the provided model."""
    batch_size = input_rgb.shape[0]
    frame = input_rgb.shape[1]
    height, width = input_rgb.shape[-2:]

    videos = model.pipe(
        prompt=[""] * batch_size,
        negative_prompt=[""] * batch_size,
        mode=args.mode,
        height=height,
        width=width,
        num_frames=frame,
        batch_size=batch_size,
        input_image=input_rgb[:, 0],
        extra_images=input_rgb,
        extra_image_frame_index=torch.ones(
            [batch_size, frame]).to(model.pipe.device),
        input_video=input_rgb,
        denoise_step=args.denoise_step,
        cfg_scale=1,
        seed=0,
        tiled=False,
    )
    return {k: np.array(v) for k, v in videos.items() if v is not None}


def create_directories(base_dir, subdirs):
    """Creates a list of subdirectories under a base directory."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


# ---------------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument('--base_data_dir', type=str, required=True)
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')
    cli_args = parser.parse_args()

    state_dir = cli_args.ckpt
    base_data_dir = cli_args.base_data_dir
    args_path = cli_args.model_config

    if not os.path.exists(state_dir):
        raise FileNotFoundError(f"State directory does not exist: {state_dir}")

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Args file does not exist: {args_path}")

    # 2. Load Configuration
    try:
        args = OmegaConf.load(args_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load configuration from {args_path}: {e}")

    print(f"Parsed args: {args}")
    accelerator = Accelerator()

    # 3. Model Initialization
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=args.lora_rank,
        lora_base_model=args.lora_base_model,
        args=args,
    )

    ckpt_path = os.path.join(state_dir, "model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")

    dit_state_dict = {k.replace("pipe.dit.", ""): v for k,
                      v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to(accelerator.device)

    # 4. Evaluation Configuration
    IMAGE_DATASET_CONFIGS = {
        # "diode": "configs/img_config/data_diode_all.yaml",
        # "eth3d": "configs/img_config/data_eth3d.yaml",
        # "kitti": "configs/img_config/data_kitti_eigen_test.yaml",
        "nyu": "configs/img_config/data_nyu_test.yaml",
        # "scannet": "configs/img_config/data_scannet_val.yaml",
    }

    eval_metrics = [
        "abs_relative_difference", "squared_relative_difference", "rmse_linear",
        "rmse_log", "log10", "delta1_acc", "delta2_acc", "delta3_acc",
        "i_rmse", "silog_rmse"
    ]
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    # 5. Evaluation Loop
    for dst_name, config_path in IMAGE_DATASET_CONFIGS.items():
        cfg_data = OmegaConf.load(config_path)
        print(f"Dataset Config: {cfg_data}")

        dataset: BaseDepthDataset = get_dataset(
            cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL,
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=1, num_workers=2,
            pin_memory=True, prefetch_factor=2,
        )

        # Setup paths and metrics
        save_dir = os.path.join(state_dir, 'test', dst_name)
        create_directories(save_dir, [
                           'rgb', 'pred', 'depth', 'depth_rgb', 'gt_disp', 'valid_mask', 'pred_disp_npz'])

        metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
        metric_tracker.reset()

        metrics_filename = os.path.join(save_dir, "eval_metrics.txt")
        per_sample_filename = os.path.join(
            save_dir, "_per_sample_metrics.json")
        eval_text = "Evaluation metrics:\n"

        # Batch Processing
        for idx, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                # Data preparation
                input_rgb = batch["rgb_int"].to(
                    accelerator.device).unsqueeze(1).float() / 255.0
                input_depth = batch["depth_raw_linear"].to(
                    accelerator.device).unsqueeze(1)
                valid_mask = batch['valid_mask_raw'].to(
                    accelerator.device).unsqueeze(1)
                sample_idx = batch["index"].to(accelerator.device)

                # Inference
                res_dict = generate_depth(model, args, input_rgb)

                # Convert Tensors to Numpy for processing
                pred_depth_batch = res_dict.get("depth")
                pred_rgb_batch = res_dict.get('rgb')
                gt_depth_batch = input_depth.cpu().numpy().transpose(0, 1, 3, 4, 2)
                input_rgb_batch = input_rgb.cpu().numpy().transpose(0, 1, 3, 4, 2)
                valid_mask_batch = valid_mask.cpu().numpy().transpose(0, 1, 3, 4, 2)

                batch_size = pred_depth_batch.shape[0]

                for b in range(batch_size):
                    save_idx = sample_idx[b].item()

                    # Extract single sample
                    _pred_depth = pred_depth_batch[b]
                    _pred_rgb = pred_rgb_batch[b] if pred_rgb_batch is not None else None
                    _gt_depth = gt_depth_batch[b]
                    _input_rgb = input_rgb_batch[b]
                    _valid_mask = valid_mask_batch[b]

                    # Formatting arrays (mean across channels)
                    _rgb_depth = _pred_depth
                    _pred_depth = np.mean(_pred_depth, axis=-1, keepdims=True)
                    _gt_depth = np.mean(_gt_depth, axis=-1, keepdims=True)
                    _valid_mask = np.mean(
                        _valid_mask, axis=-1, keepdims=True).astype(bool)

                    # Save NPZ
                    np.savez_compressed(os.path.join(
                        save_dir, 'pred_disp_npz', f"{save_idx:05d}.npz"), _pred_depth)

                    # Disparity & Masking logic
                    _gt_disparity, _gt_non_neg_mask = depth2disparity(
                        depth=_gt_depth, return_mask=True)
                    pred_non_neg_mask = _pred_depth > 0
                    valid_nonnegative_mask = _valid_mask & _gt_non_neg_mask & pred_non_neg_mask

                    # Visualization scaling
                    _vis_disp = (_gt_disparity - _gt_disparity[valid_nonnegative_mask].min()) / \
                                (_gt_disparity[valid_nonnegative_mask].max(
                                ) - _gt_disparity[valid_nonnegative_mask].min() + 1e-8)
                    _vis_depth = (_gt_depth - _gt_depth[valid_nonnegative_mask].min()) / \
                                 (_gt_depth[valid_nonnegative_mask].max(
                                 ) - _gt_depth[valid_nonnegative_mask].min() + 1e-8)

                    # Save Videos
                    videos_to_save = [
                        (valid_nonnegative_mask.copy().astype(
                            np.float32), 'valid_mask'),
                        (_vis_disp, 'gt_disp'),
                        (_rgb_depth, 'depth_rgb'),
                        (_input_rgb, 'rgb'),
                        (_pred_depth, 'pred'),
                        (_vis_depth, 'depth')
                    ]
                    for video_data, folder_name in videos_to_save:
                        save_path = os.path.join(
                            save_dir, folder_name, f"{save_idx:05d}.mp4")
                        save_video(video_data, save_path, fps=5, quality=6)

                    # Alignment & Depth Conversion
                    disparity_pred, scale, shift = align_depth_least_square_video(
                        gt_arr=_gt_disparity, pred_arr=_pred_depth,
                        valid_mask_arr=valid_nonnegative_mask, return_scale_shift=True
                    )

                    disparity_pred = np.clip(
                        disparity_pred, a_min=1e-3, a_max=None)
                    depth_pred = disparity2depth(disparity_pred)
                    depth_pred = np.clip(
                        depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth)
                    depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

                    # Tracking Metrics
                    sample_metric = {'idx': f"{save_idx:05d}"}

                    _valid_mask_ts = torch.from_numpy(
                        _valid_mask).to(accelerator.device).squeeze()
                    depth_pred_ts = torch.from_numpy(
                        depth_pred).to(accelerator.device).squeeze()
                    gt_depth_ts = torch.from_numpy(_gt_depth).to(
                        accelerator.device).squeeze()

                    for met_func in metric_funcs:
                        _metric_name = met_func.__name__
                        _metric = met_func(
                            depth_pred_ts, gt_depth_ts, _valid_mask_ts).item()
                        sample_metric[_metric_name] = f"{_metric:.6f}"
                        metric_tracker.update(_metric_name, _metric)

                    # Save per-sample metrics
                    try:
                        with open(per_sample_filename, "a+") as f:
                            f.write(json.dumps(sample_metric) + "\n")
                        print(f"Metrics this sample {sample_metric}")
                    except IOError as e:
                        print(
                            f"Warning: Failed to write per-sample metrics: {e}")

        # -------------------- Final Metric Saving --------------------
        eval_text += tabulate([metric_tracker.result().keys(),
                              metric_tracker.result().values()]) + "\n"

        try:
            with open(metrics_filename, "a+") as f:
                f.write(eval_text)
            print(f"Evaluation metrics saved to {metrics_filename}")
        except IOError as e:
            print(f"Warning: Failed to write evaluation metrics: {e}")

        print(f"Validation videos saved to {save_dir}\n" + "-"*50)
