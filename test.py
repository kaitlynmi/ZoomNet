# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import builder, configurator, io, misc, ops, pipeline, recorder


def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/zoomnet/cod_zoomnet.py", type=str)
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs.json", type=str)
    parser.add_argument("--model-name", default="ZoomNet", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--load-from", type=str)  # trained weight
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--minmax-results", action="store_true")
    parser.add_argument("--info", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--evaluate", action='store_true', help="Enable metrics calculation between GT and predictions.")
    parser.add_argument("--output-option", type=str, choices=["mask", "overlay", 'none'], default="mask", 
                        help="Choose between outputting the prediction masks or mask overlay on original image")
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)
    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size
    if args.load_from is not None:
        config.load_from = args.load_from
    if args.info is not None:
        config.experiment_tag = args.info
    config.output_option = args.output_option  # Store output option in config
    if config.output_option != 'none':
        if args.save_path is not None:
            if os.path.exists(args.save_path):
                if len(os.listdir(args.save_path)) != 0:
                    raise ValueError(f"--save-path is not an empty folder.")
            else:
                print(f"{args.save_path} does not exist, create it.")
                os.makedirs(args.save_path)
        config.save_path = args.save_path
    else:
        config.save_path = None
    config.test.to_minmax = args.minmax_results
    config.evaluate = args.evaluate is not None  # Store evaluation flag in config
    if args.threshold is not None:
        assert args.threshold < 1 and args.threshold >=0 
        config.threshold = args.threshold
    else: 
        config.threshold = None

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            raise KeyError(f"{te_dataset} not in {args.datasets_info}!!!")
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    return config

def denormalize_image(image_tensor):
    """
    Reverses the normalization of an image tensor and converts it to a NumPy array.
    Adjust the mean and std according to your preprocessing.
    """
    mean = np.array([0.485, 0.456, 0.406])  # dataset's mean
    std = np.array([0.229, 0.224, 0.225])   # dataset's std

    image_np = image_tensor.numpy().transpose(1, 2, 0)  # From CxHxW to HxWxC
    image_np = (image_np * std + mean) * 255.0
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return image_np


def overlay_mask_on_image(image, mask, alpha=0.5, color='green'):
    """Utility function to overlay a mask on the original image with specified color"""
    # Ensure the mask is single-channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Create a color mask based on the specified color
    color_mask = np.zeros_like(image)
    if color == 'green': 
        color_mask[:, :, 1] = mask  # Apply mask to the green channel
    elif color == 'red':
        color_mask[:, :, 2] = mask
    elif color == 'blue':
        color_mask[:, :, 0] = mask
    else:
        raise ValueError('color has to be [green, red, blue]')

    # Overlay the mask onto the image
    overlayed_image = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)
    return overlayed_image


def test_once(
    model,
    data_loader,
    save_path,
    tta_setting,
    clip_range=None,
    show_bar=False,
    desc="[TE]",
    to_minmax=False,
    evaluate=True,
    output_option="mask",
    threshold = 0.1,
):
    model.is_training = False
    if evaluate:
        cal_total_seg_metrics = recorder.CalTotalMetric()
        # loss_recorder = recorder.AvgMeter()
        # metrics_calculator = recorder.MetricsCalculator()
        # loss_recorder.reset()
        # metrics_calculator.reset()
        if threshold is None: threshold = 0.1 

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        if evaluate:
            assert 'mask_path' in batch["info"]
        batch_images = misc.to_device(batch["data"], device=model.device)
        if tta_setting.enable:
            logits = pipeline.test_aug(
                model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
            )
        else:
            logits = model(data=batch_images)      
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

        for i, pred in enumerate(probs):
            image_path = batch["info"]["image_path"][i]
            image = io.read_color_array(image_path)
            image_h, image_w = image.shape[:2]
            
            if evaluate:
                mask_path = batch["info"]["mask_path"][i]
                mask_array = io.read_gray_array(mask_path, dtype=np.uint8)

            # Resize prediction to match the image size
            pred = ops.imresize(pred, target_h=image_h, target_w=image_w, interp="linear")

            if to_minmax:
                pred = ops.minmax(pred)
            
            if threshold is None:
                pred_uint8 = (pred * 255).astype(np.uint8)
            else:
                pred_binary = (pred >= threshold).astype(np.uint8)
                pred_uint8 = (pred_binary * 255).astype(np.uint8)

            # Handle output based on the selected option
            if output_option == "mask":
                save_name = os.path.basename(image_path)
                ops.save_array_as_image(data_array=pred_uint8, save_name=save_name, save_dir=save_path)

            elif output_option == "overlay":
                if evaluate:
                    # Compute true positives and false positives
                    true_positive = (pred_binary & mask_array) * 255
                    false_positive = (pred_binary & (~mask_array)) * 255
                    # false_negative = ((~pred_binary) & mask_array)* 255

                    # Overlaying based on evaluation
                    overlayed_image_tp = overlay_mask_on_image(image, true_positive, color='green')  # Green for TP
                    overlayed_image_fp = overlay_mask_on_image(overlayed_image_tp, false_positive, color='red')  # Red for FP
                    # overlayed_image_fn = overlay_mask_on_image(overlayed_image_fp, false_negative, color='blue')
                    save_name = os.path.basename(image_path)
                    ops.save_array_as_image(data_array=overlayed_image_fp, save_name=save_name, save_dir=save_path)
                else: 
                    overlayed_image = overlay_mask_on_image(image, pred_uint8)
                    save_name = os.path.basename(image_path)
                    ops.save_array_as_image(data_array=overlayed_image, save_name=save_name, save_dir=save_path)
            # Evaluate if needed
            if evaluate:
                mask_path = batch["info"]["mask_path"][i]
                mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
                cal_total_seg_metrics.step(pred_uint8, mask_array, mask_path)
                # metrics_calculator.update_from_numpy(pred_uint8, mask_array)

    if evaluate:
        fixed_seg_results = cal_total_seg_metrics.get_results()
        # metrics = metrics_calculator.compute_metrics()
        # if metrics:
        #     metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        #     print(f"Metrics: {metrics_str}")
        return fixed_seg_results
    return None


@torch.no_grad()
def testing(model, cfg):
    pred_save_path = None
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):
        if cfg.save_path:
            pred_save_path = os.path.join(cfg.save_path, data_name)
            print(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
            evaluate=cfg.evaluate,  # Pass the evaluate flag
            output_option=cfg.output_option,  # Pass the output option
            threshold = cfg.threshold,
        )
        if cfg.evaluate and seg_results:
            print(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")
        else:
            print(f"Testing without evaluation on the dataset {data_name}")


def main():
    cfg = parse_config()

    model, model_code = builder.build_obj_from_registry(
        registry_name="MODELS", obj_name=cfg.model_name, return_code=True
    )
    io.load_weight(model=model, load_path=cfg.load_from)

    model.device = "cuda:0"
    model.to(model.device)
    model.eval()

    testing(model=model, cfg=cfg)


if __name__ == "__main__":
    main()

