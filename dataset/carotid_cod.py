# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import albumentations as A
import cv2
import torch

from dataset.base_dataset import _BaseSODDataset
from dataset.transforms.resize import ms_resize, ss_resize
from dataset.transforms.rotate import UniRotate
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys
from utils.io.image import read_color_array, read_gray_array


@DATASETS.register(name="carotid_cod_te_img")
class Carotid_TestDataset_NoMask(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        # Get only the 'image' paths (no 'mask' information)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=[])  # Removed 'mask' key
        self.total_image_paths = self.datasets["image"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if len(self.total_image_paths) == 0:
            raise ValueError("The dataset is empty. Check if the dataset path is correct.")

        print(f"Loaded {len(self.total_image_paths)} images for testing (without masks).")

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]

        # Load and normalize image
        image = read_color_array(image_path)
        image = self.image_norm(image=image)["image"]

        # Resize image at multiple scales
        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
            },
            info=dict(
                image_path=image_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)

@DATASETS.register(name="carotid_cod_te")
class Carotid_TestDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]

        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)

        image = self.image_norm(image=image)["image"]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "mask": mask_1_0,
            },
            info=dict(
                image_path=image_path,
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)

@DATASETS.register(name="carotid_cod_val")
class Carotid_ValidateDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if len(self.total_image_paths) == 0:
            raise ValueError("The dataset is empty. Check if the dataset path is correct.")
        if len(self.total_image_paths) != len(self.total_mask_paths):
            raise ValueError("Mismatch between number of images and masks.")

        print(f"Loaded {len(self.total_image_paths)} images and masks for validation.")

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]

        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)

        image = self.image_norm(image=image)["image"]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "mask": mask_1_0,
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)



@DATASETS.register(name="carotid_cod_tr")
class Carotid_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        # print(f'total_image_paths: {self.total_image_paths}')
        self.total_mask_paths = self.datasets["mask"]
        # print(f'total_mask_paths: {self.total_mask_paths}')
        if len(self.total_image_paths) == 0:
            raise ValueError("The dataset is empty. Check if the dataset path is correct.")
        print(f"Loaded {len(self.total_image_paths)} images for training.")
        self.resize = A.Resize
        self.normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, 
                                   scale_limit=(0.0, 0.0), 
                                   rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, 
                                   interpolation=cv2.INTER_LINEAR, p=0.5),
            ],
        )
        # More Augmentation
        self.image_transform = A.Compose([
            # A.ToGray(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]

        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)

        transformed = self.joint_trans(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        image = self.image_transform(image=image)["image"]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "mask": mask_1_0,
            }
        )

    def __len__(self):
        return len(self.total_image_paths)
