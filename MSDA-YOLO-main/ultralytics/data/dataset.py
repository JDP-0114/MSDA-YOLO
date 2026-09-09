# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True))
        return transforms


class YOLOMultiModalImageDataset(YOLODataset):
    """
    YOLO多模态图像数据集 - 支持RGB+X模态的目标检测
    
    支持功能：
    - RGB + 深度图像 (RGB + Depth)  
    - RGB + 红外图像 (RGB + IR)
    - 早期融合策略（输入层6通道拼接）
    - 多模态缓存机制
    """
    
    def __init__(self, *args, data=None, task="detect", modalities=None, train_modality=None, **kwargs):
        """
        初始化多模态数据集
        
        Args:
            modalities (list): 模态列表，如 ['rgb', 'ir'] 或 ['rgb', 'depth']
            train_modality (str, optional): 训练时使用的单模态类型，如 'rgb', 'depth', 'thermal'
                如果为None，则进行双模态训练；如果指定，则进行单模态训练并应用模态填充
        """
        # 存储模态配置
        self.modalities = modalities or ["rgb", "depth"]
        self.x_modality = [m for m in self.modalities if m != 'rgb'][0]  # 获取X模态
        
        # 存储训练模态配置
        self.train_modality = train_modality
        
        # 🔧 修复：先设置data属性，再调用检测方法
        self.data = data
        
        # 检测自体模态需求
        self.self_modal_type = self._detect_self_modal_requirement(kwargs)
        
        # 如果检测到自体模态，导入生成器
        if self.self_modal_type:
            from ultralytics.models.yolo.multimodal import generate_self_modality
            self.generate_self_modality = generate_self_modality
            LOGGER.info(f"🔧 自体模态检测: 启用{self.self_modal_type}模态生成")
        
        # 如果启用单模态训练，导入模态填充函数
        if self.train_modality:
            from ultralytics.models.yolo.multimodal.modal_filling import generate_modality_filling
            self.generate_modality_filling = generate_modality_filling
            LOGGER.info(f"启用单模态训练模式: {self.train_modality}-only，将应用智能模态填充")
        
        # 验证模态配置
        self._validate_modalities()
        
        # 初始化多模态缓存存储
        self.x_ims = None  # X模态图像缓存
        self.x_im_hw0 = None  # X模态原始尺寸缓存  
        self.x_im_hw = None   # X模态调整后尺寸缓存
        
        # 初始化X模态路径映射缓存
        self.x_path_cache = {}  # 从RGB文件名到X模态完整路径的映射
        
        # 预先构建模态路径映射（在父类初始化前）
        self._pre_build_modality_paths(*args, **kwargs)
        
        # 调用父类初始化
        super().__init__(*args, data=data, task=task, **kwargs)
        
        # 重新构建完整的模态路径映射（使用父类初始化后的img_path）
        self._build_modality_paths()
        
        # 构建X模态路径缓存映射（核心优化）
        self._build_x_path_cache()
        
        # 验证数据完整性
        self._validate_multimodal_files()
        
        LOGGER.info(f"多模态数据集初始化完成: {self.modalities}, 模态数量: {len(self.modalities)}")

    def _detect_self_modal_requirement(self, kwargs) -> Optional[str]:
        """
        检测模型配置中的自体模态需求
        
        Args:
            kwargs: 初始化参数，可能包含模型配置信息
            
        Returns:
            str or None: 检测到的自体模态类型 ('edge', 'texture', 'gradient') 或 None
        """
        # 尝试从多个来源获取模型配置
        model_config = None
        
        # 1. 从kwargs中获取模型配置
        if 'model_config' in kwargs:
            model_config = kwargs['model_config']
        
        # 2. 从data中获取模型配置
        elif self.data and isinstance(self.data, dict) and 'model_config' in self.data:
            model_config = self.data['model_config']
        
        # 3. 尝试从全局模型状态获取（如果有的话）
        elif hasattr(self, '_model_yaml_config'):
            model_config = self._model_yaml_config
        
        if not model_config:
            return None
        
        # 检查backbone和head层配置中的第5字段
        all_layers = []
        if isinstance(model_config, dict):
            all_layers.extend(model_config.get('backbone', []))
            all_layers.extend(model_config.get('head', []))
        
        # 检测自体模态字段
        self_modal_types = ['self_edge', 'self_texture', 'self_gradient']
        
        for layer_config in all_layers:
            if len(layer_config) >= 5:
                input_source = layer_config[4]
                if input_source in self_modal_types:
                    # 提取模态类型（去掉'self_'前缀）
                    modal_type = input_source.replace('self_', '')
                    LOGGER.info(f"🔍 检测到自体模态配置: {input_source} -> {modal_type}")
                    return modal_type
        
        return None

    def _pre_build_modality_paths(self, *args, **kwargs):
        """
        在父类初始化前预先构建基础的模态路径映射
        这是为了支持缓存过程中的路径查找
        """
        self.modality_paths = {}
        
        # 尝试从参数中获取img_path
        img_path = kwargs.get('img_path')
        if not img_path and args:
            # 如果kwargs中没有，尝试从args中获取（通常img_path是第一个参数）
            img_path = args[0] if args else None
        
        if img_path:
            # 使用提供的img_path构建路径映射
            self.modality_paths['rgb'] = img_path
            
            if hasattr(self.data, 'get') and isinstance(self.data, dict):
                modalities_config = self.data.get('modalities', {})
                
                if isinstance(modalities_config, dict) and self.x_modality in modalities_config:
                    x_relative_path = modalities_config[self.x_modality]
                    x_full_path = self._construct_x_modality_path(img_path, x_relative_path)
                    self.modality_paths[self.x_modality] = x_full_path
                else:
                    # 使用默认路径构造
                    x_full_path = self._construct_x_modality_path(img_path, f"images_{self.x_modality}")
                    self.modality_paths[self.x_modality] = x_full_path
            else:
                # 如果没有配置，使用默认路径
                x_full_path = self._construct_x_modality_path(img_path, f"images_{self.x_modality}")
                self.modality_paths[self.x_modality] = x_full_path
        else:
            # 如果无法获取img_path，设置空的路径映射（父类初始化后会重新构建）
            self.modality_paths['rgb'] = ''
            self.modality_paths[self.x_modality] = ''
        
        LOGGER.info(f"预构建模态路径映射: {self.modality_paths}")

    def _validate_modalities(self):
        """验证模态配置的有效性 - 配置驱动，不限制具体模态类型"""
        if not isinstance(self.modalities, list) or len(self.modalities) != 2:
            raise ValueError(f"模态配置必须是包含2个元素的列表，当前: {self.modalities}")
        
        if 'rgb' not in self.modalities:
            raise ValueError(f"模态配置必须包含'rgb'，当前: {self.modalities}")
        
        # ✅ 移除硬编码限制，信任用户配置
        # 用户通过配置文件明确指定了模态类型，系统应该信任并支持
        x_modalities = [m for m in self.modalities if m != 'rgb']
        if not x_modalities:
            raise ValueError(f"必须包含至少一个非RGB模态，当前: {self.modalities}")
            
        # ✅ 记录用户配置的模态组合，不进行限制性验证
        LOGGER.info(f"✅ 使用用户配置的模态组合: {self.modalities}")
        LOGGER.info(f"✅ X模态类型: {x_modalities[0]} (配置驱动，无限制)")

    def _build_modality_paths(self):
        """
        构建模态路径映射
        处理两种配置格式:
        1. 列表格式: modalities: ['rgb', 'ir'] - 自动推断路径
        2. 字典格式: modalities: {rgb: 'images', ir: 'images_ir'} - 显式指定路径
        """
        # 初始化路径字典
        self.modality_paths = {}
        
        # 获取当前RGB图像的完整路径（从img_path获取）
        rgb_base_path = self.img_path  # 这是当前模式下的RGB路径
        self.modality_paths['rgb'] = rgb_base_path
        
        # 获取数据集根目录和模式信息
        if hasattr(self.data, 'get') and isinstance(self.data, dict):
            dataset_root = self.data.get('path', '')
            modalities_config = self.data.get('modalities', {})
            
            # 处理列表格式的modalities配置
            if isinstance(modalities_config, list):
                LOGGER.info(f"处理列表格式的模态配置: {modalities_config}")
                # 从列表中找到非RGB的模态作为X模态
                non_rgb_modalities = [m for m in modalities_config if m != 'rgb']
                if non_rgb_modalities:
                    # 使用第一个非RGB模态作为X模态
                    x_mod = non_rgb_modalities[0]
                    # 根据RGB路径构造X模态的完整路径
                    x_full_path = self._construct_x_modality_path(rgb_base_path, f"images_{x_mod}")
                    self.modality_paths[x_mod] = x_full_path
                    LOGGER.info(f"自动生成X模态路径映射: {x_mod} -> {x_full_path}")
            
            # 处理字典格式的modalities配置
            elif isinstance(modalities_config, dict):
                LOGGER.info(f"处理字典格式的模态配置: {modalities_config}")
                if self.x_modality in modalities_config:
                    x_relative_path = modalities_config[self.x_modality]
                    # 根据RGB路径构造X模态的完整路径
                    x_full_path = self._construct_x_modality_path(rgb_base_path, x_relative_path)
                    self.modality_paths[self.x_modality] = x_full_path
                else:
                    # 默认路径构造: images_depth, images_thermal等
                    x_full_path = self._construct_x_modality_path(rgb_base_path, f"images_{self.x_modality}")
                    self.modality_paths[self.x_modality] = x_full_path
                    
            else:
                # 如果modalities配置为空或不是预期格式，使用默认路径
                LOGGER.warning(f"未识别的modalities配置格式: {type(modalities_config)}, 使用默认路径")
                x_full_path = self._construct_x_modality_path(rgb_base_path, f"images_{self.x_modality}")
                self.modality_paths[self.x_modality] = x_full_path
        else:
            # 如果data不是字典或没有get方法，使用默认路径
            x_full_path = self._construct_x_modality_path(rgb_base_path, f"images_{self.x_modality}")
            self.modality_paths[self.x_modality] = x_full_path
        
        LOGGER.info(f"模态路径映射完成: {self.modality_paths}")

    def _construct_x_modality_path(self, rgb_path, x_relative_dir):
        """
        根据RGB路径构造X模态的完整路径
        
        Args:
            rgb_path (str): RGB图像路径, 如 '/path/to/dataset/images/train'
            x_relative_dir (str): X模态的相对目录名, 如 'images_ir'
            
        Returns:
            str: X模态的完整路径, 如 '/path/to/dataset/images_ir/train'
        """
        # 分析RGB路径结构
        rgb_path = os.path.normpath(rgb_path)
        path_parts = rgb_path.split(os.sep)
        
        # 找到'images'在路径中的位置
        images_idx = -1
        for i, part in enumerate(path_parts):
            if 'images' in part and not part.startswith('images_'):  # 匹配'images'但不匹配'images_ir'等
                images_idx = i
                break
        
        if images_idx >= 0:
            # 替换'images'为x_relative_dir，保持路径结构
            new_path_parts = path_parts.copy()
            new_path_parts[images_idx] = x_relative_dir
            x_full_path = os.sep.join(new_path_parts)
        else:
            # 如果没有找到'images'目录，采用兄弟目录策略
            parent_dir = os.path.dirname(rgb_path)
            mode_suffix = os.path.basename(rgb_path)  # 获取 train/val/test
            x_full_path = os.path.join(os.path.dirname(parent_dir), x_relative_dir, mode_suffix)
        
        return x_full_path

    def _build_x_path_cache(self):
        """
        构建X模态路径缓存映射
        
        一次性扫描X模态目录，建立从RGB文件名到X模态完整路径的直接映射。
        这样可以避免运行时的多次文件系统调用，大幅提升性能。
        """
        LOGGER.info(f"开始构建{self.x_modality}模态路径缓存...")
        
        x_base_path = self.modality_paths.get(self.x_modality)
        if not x_base_path or not os.path.exists(x_base_path):
            LOGGER.warning(f"X模态基础路径不存在: {x_base_path}")
            return
        
        # 支持的图像扩展名
        supported_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        
        # 递归扫描X模态目录下的所有图像文件
        x_files_found = 0
        for root, dirs, files in os.walk(x_base_path):
            for file in files:
                file_lower = file.lower()
                # 检查是否为支持的图像文件
                if any(file_lower.endswith(ext) for ext in supported_extensions):
                    # 提取基础文件名（不含扩展名）
                    base_name = os.path.splitext(file)[0]
                    # 构建完整路径
                    full_path = os.path.join(root, file)
                    # 建立映射：基础文件名 -> 完整路径
                    self.x_path_cache[base_name] = full_path
                    x_files_found += 1
        
        LOGGER.info(f"构建{self.x_modality}模态路径缓存完成: 找到 {x_files_found} 个文件")
        
        # 输出缓存统计信息
        if x_files_found > 0:
            sample_entries = list(self.x_path_cache.items())[:3]
            LOGGER.info(f"缓存示例: {sample_entries}")

    def _find_corresponding_x_image_optimized(self, rgb_path):
        """
        优化版本：使用预缓存映射快速查找对应的X模态图像
        
        将原来的O(n)文件系统调用优化为O(1)字典查找
        
        Args:
            rgb_path: RGB图像路径
            
        Returns:
            str: X模态图像的完整路径
            
        Raises:
            FileNotFoundError: 如果找不到对应的X模态图像
        """
        # 提取RGB图像的基础文件名
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        
        # 从预缓存中查找对应的X模态路径
        x_path = self.x_path_cache.get(base_name)
        
        if x_path and os.path.exists(x_path):
            return x_path
        
        # 如果缓存中没有找到，回退到原始方法（兼容性保证）
        LOGGER.debug(f"缓存未命中，回退到原始查找方法: {base_name}")
        return self._find_corresponding_x_image_fallback(rgb_path)

    def _find_corresponding_x_image_fallback(self, rgb_path):
        """
        回退方法：原始的文件系统查找逻辑
        
        当预缓存未命中时使用，确保兼容性
        """
        # 提取基础信息
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        rgb_ext = os.path.splitext(rgb_path)[1]
        
        # 使用预构建的X模态路径映射
        x_base_path = self.modality_paths.get(self.x_modality)
        if not x_base_path:
            raise ValueError(f"X模态路径映射未找到: {self.x_modality}")
        
        # 构造X模态图像的完整路径
        # 从RGB路径中提取相对路径部分（去除RGB base path）
        rgb_base_path = self.modality_paths.get('rgb', '')
        if rgb_base_path and rgb_path.startswith(rgb_base_path):
            # 获取相对于RGB base path的相对路径
            relative_path = os.path.relpath(rgb_path, rgb_base_path)
            # 构造对应的X模态路径
            x_dir = os.path.dirname(os.path.join(x_base_path, relative_path))
        else:
            # 后备方案：使用原来的路径替换策略
            rgb_dir = os.path.dirname(rgb_path)
            if 'images' in rgb_dir:
                x_dir = rgb_dir.replace('images', f'images_{self.x_modality}', 1)
            else:
                x_dir = rgb_dir.replace(os.path.basename(rgb_dir), f"{os.path.basename(rgb_dir)}_{self.x_modality}")
        
        # 按优先级搜索扩展名
        extensions = [rgb_ext, '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        for ext in extensions:
            x_path = os.path.join(x_dir, f"{base_name}{ext}")
            if os.path.exists(x_path):
                # 将找到的路径加入缓存以供下次使用
                self.x_path_cache[base_name] = x_path
                return x_path
        
        # 未找到对应文件时的处理
        raise FileNotFoundError(f"No corresponding {self.x_modality} modality image for {rgb_path}. Searched in: {x_dir}")

    def _find_corresponding_x_image(self, rgb_path):
        """
        根据RGB图像路径找到对应的X模态图像（优化版本）
        
        使用预缓存映射进行快速查找，避免多次文件系统调用
        """
        return self._find_corresponding_x_image_optimized(rgb_path)

    def _load_x_modality(self, x_path):
        """
        加载X模态图像并确保为3通道格式
        
        Args:
            x_path: X模态图像路径
            
        Returns:
            numpy.ndarray: 3通道X模态图像 [H,W,3]
        """
        # 加载图像
        x_img = cv2.imread(x_path)
        if x_img is None:
            raise FileNotFoundError(f"Failed to load X modality image: {x_path}")
        
        # 确保为3通道
        x_img = self._ensure_3channel(x_img)
        
        return x_img

    def _ensure_3channel(self, img):
        """
        确保图像为3通道，非3通道时重复通道
        """
        if len(img.shape) == 2:  # 灰度图
            img = np.stack([img, img, img], axis=2)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:  # 单通道
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # 丢弃Alpha通道
            elif img.shape[2] != 3:  # 其他通道数
                # 取前3个通道或重复到3通道
                if img.shape[2] > 3:
                    img = img[:, :, :3]
                else:
                    # 重复通道到3通道
                    img = np.repeat(img, 3 // img.shape[2] + 1, axis=2)[:, :, :3]
        
        return img

    def _construct_6channel_tensor(self, rgb_img, x_img):
        """
        构造6通道张量
        
        Args:
            rgb_img: RGB图像 [H,W,3]
            x_img: X模态图像 [H,W,3] 
            
        Returns:
            multimodal_img: 6通道图像 [H,W,6]
        """
        # 确保两个图像尺寸一致
        if rgb_img.shape[:2] != x_img.shape[:2]:
            x_img = cv2.resize(x_img, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # 确保都是3通道
        rgb_img = self._ensure_3channel(rgb_img)
        x_img = self._ensure_3channel(x_img)
        
        # 检查是否需要进行单模态训练填充
        if self.train_modality:
            return self._apply_single_modality_filling(rgb_img, x_img)
        
        # 双模态训练：直接拼接为6通道 [H,W,6]
        multimodal_img = np.concatenate([rgb_img, x_img], axis=2)
        
        return multimodal_img
    
    def _apply_single_modality_filling(self, rgb_img, x_img):
        """
        应用单模态训练的模态填充逻辑
        
        Args:
            rgb_img: RGB图像 [H,W,3]
            x_img: X模态图像 [H,W,3]
            
        Returns:
            multimodal_img: 6通道图像 [H,W,6]，其中一个模态被填充数据替代
        """
        # 将numpy图像转换为torch tensor进行填充处理
        import torch
        
        # 转换格式：[H,W,3] -> [1,3,H,W] (添加batch维度并调整通道顺序)
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x_tensor = torch.from_numpy(x_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        if self.train_modality == 'rgb':
            # RGB-only训练：保留RGB，填充X模态
            filled_x_tensor = self.generate_modality_filling(
                source_tensor=rgb_tensor,
                source_modality='rgb',
                target_modality=self.x_modality
            )
            final_rgb_tensor = rgb_tensor
            final_x_tensor = filled_x_tensor
            
        elif self.train_modality in ['depth', 'thermal', 'ir']:
            # X-only训练：保留X模态，填充RGB
            filled_rgb_tensor = self.generate_modality_filling(
                source_tensor=x_tensor,
                source_modality=self.train_modality,
                target_modality='rgb'
            )
            final_rgb_tensor = filled_rgb_tensor
            final_x_tensor = x_tensor
            
        else:
            raise ValueError(f"不支持的train_modality: {self.train_modality}")
        
        # 转换回numpy格式：[1,3,H,W] -> [H,W,3]
        final_rgb_np = (final_rgb_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        final_x_np = (final_x_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # 拼接为6通道 [H,W,6]
        multimodal_img = np.concatenate([final_rgb_np, final_x_np], axis=2)
        
        return multimodal_img

    def get_image_and_label(self, index):
        """
        核心方法: 加载RGB+X图像并合并为6通道
        现在支持多模态缓存机制和自体模态生成
        
        Args:
            index: 图像索引
            
        Returns:
            dict: 包含6通道图像和标签信息的字典
        """
        # 调用父类方法获取基础标签信息
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape is for rect, remove it
        
        try:
            # 加载RGB图像（使用缓存）
            rgb_img, ori_shape, resized_shape = self.load_image(index)
            
            # 检查是否需要生成自体模态
            if self.self_modal_type:
                # 自体模态生成模式：从RGB生成X模态
                multimodal_img = self._generate_self_modal_6channel(rgb_img)
                LOGGER.debug(f"🔧 自体模态生成: RGB -> {self.self_modal_type} -> 6通道")
            else:
                # 传统多模态模式：加载X模态图像
                x_img, x_ori_shape, x_resized_shape = self._load_x_image_cached(index, target_shape=rgb_img.shape[:2])
                
                # 构造6通道图像
                multimodal_img = self._construct_6channel_tensor(rgb_img, x_img)
            
            # 更新标签信息
            label["img"] = multimodal_img
            label["ori_shape"] = ori_shape  
            label["resized_shape"] = resized_shape
            label["ratio_pad"] = (
                resized_shape[0] / ori_shape[0],
                resized_shape[1] / ori_shape[1],
            )
            
            if self.rect:
                label["rect_shape"] = self.batch_shapes[self.batch[index]]
            
            return self.update_labels_info(label)
            
        except FileNotFoundError as e:
            LOGGER.warning(f"Skipping index {index}: {e}")
            # 如果找不到对应的X模态图像，返回None或者使用零填充
            # 这里选择抛出异常，让调用者处理
            raise

    def _generate_self_modal_6channel(self, rgb_img):
        """
        从RGB图像生成自体模态并构建6通道输入
        
        Args:
            rgb_img: RGB图像 [H,W,3] numpy数组
            
        Returns:
            multimodal_img: 6通道图像 [H,W,6] numpy数组
        """
        import torch
        
        # 转换格式：[H,W,3] -> [1,3,H,W] (添加batch维度并调整通道顺序)
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 生成自体模态
        self_modal_tensor = self.generate_self_modality(
            rgb_tensor=rgb_tensor,
            modal_type=self.self_modal_type
        )
        
        # 转换回numpy格式：[1,3,H,W] -> [H,W,3]
        rgb_np = (rgb_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        self_modal_np = (self_modal_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        
        # 拼接为6通道 [H,W,6]
        multimodal_img = np.concatenate([rgb_np, self_modal_np], axis=2)
        
        return multimodal_img

    def _validate_multimodal_files(self):
        """验证多模态文件的完整性，并构建有效索引列表"""
        if not hasattr(self, 'im_files') or not self.im_files:
            return
        
        # 如果是自体模态模式，不需要验证X模态文件
        if self.self_modal_type:
            # 自体模态模式：所有RGB图像都是有效的
            self.valid_multimodal_indices = list(range(len(self.im_files)))
            LOGGER.info(f"🔧 自体模态模式: 所有 {len(self.im_files)} 个RGB图像都有效 (无需X模态文件)")
            return
            
        missing_files = []
        valid_indices = []  # 有效的多模态索引列表
        cache_hits = 0  # 缓存命中次数
        cache_misses = 0  # 缓存未命中次数
        
        LOGGER.info(f"开始验证多模态文件完整性，共 {len(self.im_files)} 个RGB图像")
        
        for i, rgb_path in enumerate(self.im_files):
            try:
                base_name = os.path.splitext(os.path.basename(rgb_path))[0]
                
                # 检查是否在缓存中
                if base_name in self.x_path_cache:
                    cache_hits += 1
                    # 验证缓存中的路径是否仍然存在
                    x_path = self.x_path_cache[base_name]
                    if os.path.exists(x_path):
                        valid_indices.append(i)
                        continue
                    else:
                        # 缓存路径已失效，从缓存中移除
                        del self.x_path_cache[base_name]
                        cache_misses += 1
                else:
                    cache_misses += 1
                
                # 缓存未命中或路径失效，执行查找
                self._find_corresponding_x_image(rgb_path)
                valid_indices.append(i)  # 记录有效索引
                
            except FileNotFoundError:
                missing_files.append((i, rgb_path))
        
        # 输出性能统计
        total_checks = len(self.im_files)
        cache_hit_rate = (cache_hits / total_checks * 100) if total_checks > 0 else 0
        LOGGER.info(f"路径缓存性能统计 - 命中: {cache_hits}, 未命中: {cache_misses}, 命中率: {cache_hit_rate:.1f}%")
        
        if missing_files:
            LOGGER.warning(f"Missing {self.x_modality} modality files for {len(missing_files)} RGB images")
            
            # 过滤掉缺失对应文件的样本
            valid_indices = [i for i in range(len(self.im_files)) if i not in [x[0] for x in missing_files]]
            
            if valid_indices:
                self.im_files = [self.im_files[i] for i in valid_indices]
                if hasattr(self, 'labels') and self.labels:
                    self.labels = [self.labels[i] for i in valid_indices]
                    
                LOGGER.info(f"Filtered dataset to {len(self.im_files)} complete image pairs")
            else:
                raise ValueError(f"No valid {self.x_modality} modality files found for any RGB images")
        else:
            # 如果所有文件都有对应的多模态数据，valid_indices就是全部索引
            valid_indices = list(range(len(self.im_files)))
        
        # 存储有效索引，供Mosaic/MixUp使用
        self.valid_multimodal_indices = valid_indices
        LOGGER.info(f"Found {len(self.valid_multimodal_indices)} valid multimodal image pairs")

    def get_valid_indices(self):
        """
        获取有效的多模态索引列表，供Mosaic/MixUp使用
        
        Returns:
            list: 有效的索引列表，这些索引对应的图像都有完整的多模态数据
        """
        if hasattr(self, 'valid_multimodal_indices'):
            return self.valid_multimodal_indices
        else:
            # 如果没有预计算，返回全部索引（向后兼容）
            return list(range(len(self.im_files)))

    def __getitem__(self, index):
        """
        获取指定索引的多模态数据
        
        Returns:
            dict: 包含6通道图像和标签的字典
        """
        return self.transforms(self.get_image_and_label(index))

    def build_transforms(self, hyp=None):
        """
        构建多模态数据增强transforms
        
        重写父类方法，替换标准RandomHSV为MultiModalRandomHSV，
        并使用多模态Mosaic/MixUp确保索引一致性，保护X模态数据。
        
        Args:
            hyp (dict): 超参数字典，包含增强参数
            
        Returns:
            Compose: 多模态数据增强变换组合
        """
        # 导入所需的变换类
        from ultralytics.data.augment import (
            Compose,
            Format,
            RandomFlip,
            RandomPerspective,
            LetterBox,
            CopyPaste,
        )
        from ultralytics.data.multimodal_augment import (
            MultiModalRandomHSV,
            MultiModalMosaic,
            MultiModalMixUp,
        )
        
        # 获取图像尺寸
        if self.augment:
            # 训练时的增强变换
            hyp = hyp or self.data.get("hyp", {})
            
            transforms = [
                # 使用多模态Mosaic替代标准Mosaic
                MultiModalMosaic(dataset=self, imgsz=self.imgsz, p=hyp.get("mosaic", 1.0)),
                CopyPaste(p=hyp.get("copy_paste", 0.0)),
                RandomPerspective(
                    degrees=hyp.get("degrees", 0.0),
                    translate=hyp.get("translate", 0.1),
                    scale=hyp.get("scale", 0.5),
                    shear=hyp.get("shear", 0.0),
                    perspective=hyp.get("perspective", 0.0),
                ),
                # 使用多模态MixUp替代标准MixUp
                MultiModalMixUp(dataset=self, p=hyp.get("mixup", 0.0)),
                RandomFlip(direction="vertical", p=hyp.get("flipud", 0.0)),
                RandomFlip(direction="horizontal", p=hyp.get("fliplr", 0.5)),
            ]
            
            # 使用多模态HSV变换替代标准RandomHSV
            if hyp.get("hsv_h", 0.0) or hyp.get("hsv_s", 0.0) or hyp.get("hsv_v", 0.0):
                transforms.append(
                    MultiModalRandomHSV(
                        hgain=hyp.get("hsv_h", 0.015),
                        sgain=hyp.get("hsv_s", 0.7),
                        vgain=hyp.get("hsv_v", 0.4),
                    )
                )
        else:
            # 验证时的基础变换（无增强）
            transforms = []
        
        # 添加格式化变换
        transforms.extend([
            LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True),
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.get("mask_ratio", 4.0),
                mask_overlap=hyp.get("overlap_mask", True),
            ),
        ])
        
        return Compose(transforms)

    def cache_images(self, cache):
        """扩展缓存机制以支持多模态图像缓存"""
        # 初始化多模态缓存存储
        self.x_ims = [None] * self.ni
        self.x_im_hw0 = [None] * self.ni
        self.x_im_hw = [None] * self.ni
        
        # 调用父类方法缓存RGB图像
        super().cache_images(cache)
        
        # 缓存X模态图像
        if cache:
            self._cache_x_modality_images(cache)

    def _cache_x_modality_images(self, cache):
        """缓存X模态图像"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self._cache_x_images_to_disk if cache == "disk" else self._load_x_image_for_cache
        
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    # 磁盘缓存时，x是文件路径，计算文件大小
                    x_npy_file = self._get_x_npy_file(i)
                    if x_npy_file.exists():
                        b += x_npy_file.stat().st_size
                else:  # 'ram'
                    if x is not None:
                        self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i] = x
                        b += self.x_ims[i].nbytes if self.x_ims[i] is not None else 0
                pbar.desc = f"{self.prefix}Caching {self.x_modality} images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def _load_x_image_for_cache(self, i):
        """为缓存加载X模态图像"""
        try:
            # 获取RGB图像路径并找到对应的X模态图像
            rgb_path = self.im_files[i]
            x_path = self._find_corresponding_x_image(rgb_path)
            
            # 加载X模态图像
            x_img = self._load_x_modality(x_path)
            
            # 获取RGB图像信息用于尺寸匹配
            if self.ims[i] is not None:
                # 如果RGB已缓存，使用其尺寸
                rgb_img = self.ims[i]
                h0, w0 = self.im_hw0[i]
                target_shape = rgb_img.shape[:2]
            else:
                # 如果RGB未缓存，先加载RGB获取目标尺寸
                rgb_img, (h0, w0), target_shape = self.load_image(i)
            
            # 调整X模态图像尺寸以匹配RGB
            if x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))
            
            return x_img, (h0, w0), x_img.shape[:2]
            
        except FileNotFoundError as e:
            LOGGER.warning(f"无法加载X模态图像 {i}: {e}")
            return None, None, None

    def _cache_x_images_to_disk(self, i):
        """将X模态图像保存为.npy文件到磁盘"""
        x_npy_file = self._get_x_npy_file(i)
        if not x_npy_file.exists():
            try:
                # 加载X模态图像数据
                result = self._load_x_image_for_cache(i)
                if result[0] is not None:
                    x_img, _, _ = result
                    np.save(x_npy_file.as_posix(), x_img, allow_pickle=False)
            except Exception as e:
                LOGGER.warning(f"无法缓存X模态图像到磁盘 {i}: {e}")

    def _get_x_npy_file(self, i):
        """获取X模态图像的.npy缓存文件路径"""
        rgb_npy_file = self.npy_files[i]
        # 将RGB的.npy文件名修改为X模态的
        x_npy_file = rgb_npy_file.parent / f"{rgb_npy_file.stem}_{self.x_modality}.npy"
        return x_npy_file

    def load_image(self, i, rect_mode=True):
        """重写load_image方法，支持多模态缓存加载"""
        # 加载RGB图像（使用父类方法）
        rgb_img, ori_shape, resized_shape = super().load_image(i, rect_mode)
        return rgb_img, ori_shape, resized_shape

    def _load_x_image_cached(self, i, target_shape=None):
        """从缓存或磁盘加载X模态图像"""
        # 检查内存缓存
        if self.x_ims is not None and self.x_ims[i] is not None:
            return self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i]
        
        # 检查磁盘缓存
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            try:
                x_img = np.load(x_npy_file)
                h0, w0 = x_img.shape[:2]  # 假设缓存时已调整尺寸
                return x_img, (h0, w0), x_img.shape[:2]
            except Exception as e:
                LOGGER.warning(f"加载X模态缓存文件失败 {x_npy_file}: {e}")
        
        # 从原始文件加载
        try:
            rgb_path = self.im_files[i]
            x_path = self._find_corresponding_x_image(rgb_path)
            x_img = self._load_x_modality(x_path)
            
            # 如果提供了目标尺寸，调整X模态图像尺寸
            if target_shape and x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))
            
            h0, w0 = x_img.shape[:2]
            return x_img, (h0, w0), x_img.shape[:2]
            
        except FileNotFoundError as e:
            LOGGER.error(f"无法加载X模态图像 {i}: {e}")
            raise

    def check_cache_ram(self, safety_margin=0.5):
        """检查多模态图像缓存的内存需求"""
        # 检查RGB图像内存需求（使用父类方法）
        rgb_cache_ok = super().check_cache_ram(safety_margin)
        
        # 检查X模态图像内存需求
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        
        for _ in range(n):
            try:
                # 随机选择一个样本估算X模态图像大小
                rgb_path = random.choice(self.im_files)
                x_path = self._find_corresponding_x_image(rgb_path)
                x_img = self._load_x_modality(x_path)
                
                ratio = self.imgsz / max(x_img.shape[0], x_img.shape[1])
                b += x_img.nbytes * ratio**2
            except Exception:
                # 如果加载失败，使用RGB图像大小作为估算
                rgb_img = cv2.imread(random.choice(self.im_files))
                if rgb_img is not None:
                    ratio = self.imgsz / max(rgb_img.shape[0], rgb_img.shape[1])
                    b += rgb_img.nbytes * ratio**2
        
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache X modality
        mem = psutil.virtual_memory()
        x_cache_ok = mem_required < mem.available
        
        if not x_cache_ok:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache {self.x_modality} images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if x_cache_ok else f'not caching {self.x_modality} images ⚠️'}"
            )
        
        # 只有当RGB和X模态都能缓存时才返回True
        return rgb_cache_ok and x_cache_ok


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        assert task == "detect", "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        return []

    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        labels = []
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f'{x["id"]:d}': x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]])
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        return labels

    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
