# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态数据增强模块

此模块包含专门为多模态YOLO模型设计的数据增强类。
支持RGB+X模态的同步增强处理，确保多模态数据的一致性。
"""

import numpy as np
import cv2
from ultralytics.utils import LOGGER
from ultralytics.data.augment import Mosaic, MixUp


class MultiModalRandomHSV:
    """
    多模态随机HSV增强类
    
    专门为6通道多模态图像设计的HSV增强。
    只对前3个通道(RGB)应用HSV变换，后3个通道(X模态)保持不变。
    
    这样可以避免对深度图、红外图等X模态应用不适合的颜色变换。
    
    Attributes:
        hgain (float): 色调变化的最大范围 [0, 1]
        sgain (float): 饱和度变化的最大范围 [0, 1]  
        vgain (float): 亮度变化的最大范围 [0, 1]
        
    Methods:
        __call__: 应用多模态HSV增强到输入标签
        
    Examples:
        >>> augmenter = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> labels = {"img": multimodal_img}  # 6通道图像 [H,W,6]
        >>> augmented_labels = augmenter(labels)
    """
    
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        初始化多模态随机HSV增强器
        
        Args:
            hgain (float): 色调变化的最大范围，应在[0, 1]范围内
            sgain (float): 饱和度变化的最大范围，应在[0, 1]范围内  
            vgain (float): 亮度变化的最大范围，应在[0, 1]范围内
            
        Examples:
            >>> hsv_aug = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        
    def __call__(self, labels):
        """
        对多模态图像应用随机HSV增强
        
        此方法只对6通道图像的前3个通道(RGB)应用HSV变换，
        后3个通道(X模态)保持不变，避免破坏X模态数据的特性。
        
        Args:
            labels (Dict): 包含图像数据的标签字典，必须包含'img'键
                         'img': 6通道多模态图像 numpy数组 [H,W,6]
                         
        Returns:
            (Dict): 返回修改后的标签字典，'img'为增强后的6通道图像
            
        Examples:
            >>> augmenter = MultiModalRandomHSV(hgain=0.5, sgain=0.5, vgain=0.5) 
            >>> labels = {"img": np.random.rand(640, 640, 6).astype(np.uint8)}
            >>> result = augmenter(labels)
            >>> enhanced_img = result["img"]
        """
        img = labels["img"]
        
        # 验证输入图像格式
        if len(img.shape) != 3 or img.shape[2] != 6:
            LOGGER.warning(f"MultiModalRandomHSV expects 6-channel image, got shape {img.shape}")
            return labels
            
        # 如果没有设置任何增强参数，直接返回
        if not (self.hgain or self.sgain or self.vgain):
            return labels
            
        # 分离RGB和X模态
        rgb_img = img[:, :, :3].copy()  # 前3通道：RGB
        x_img = img[:, :, 3:].copy()    # 后3通道：X模态
        
        # 生成随机增强参数
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        
        # 转换RGB到HSV并应用增强
        hue, sat, val = cv2.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV))
        dtype = rgb_img.dtype  # 保持原始数据类型
        
        # 创建查找表
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        # 应用查找表变换
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        
        # 转换回BGR
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=rgb_img)
        
        # 重新组合6通道图像：增强后的RGB + 原始X模态
        enhanced_img = np.concatenate([rgb_img, x_img], axis=2)
        
        # 更新标签字典
        labels["img"] = enhanced_img
        
        return labels


# 可扩展的多模态增强基类
class BaseMultiModalTransform:
    """
    多模态变换基类
    
    为多模态数据增强提供通用接口和工具方法。
    子类应该实现__call__方法来定义具体的增强逻辑。
    
    Methods:
        split_modalities: 将6通道图像分离为RGB和X模态
        merge_modalities: 将RGB和X模态合并为6通道图像
        validate_input: 验证输入图像格式
    """
    
    @staticmethod
    def split_modalities(img):
        """
        将6通道图像分离为RGB和X模态
        
        Args:
            img (np.ndarray): 6通道输入图像 [H,W,6]
            
        Returns:
            tuple: (rgb_img, x_img) RGB图像和X模态图像
        """
        if len(img.shape) != 3 or img.shape[2] != 6:
            raise ValueError(f"Expected 6-channel image, got shape {img.shape}")
        
        rgb_img = img[:, :, :3]  # 前3通道：RGB
        x_img = img[:, :, 3:]    # 后3通道：X模态
        return rgb_img, x_img
    
    @staticmethod  
    def merge_modalities(rgb_img, x_img):
        """
        将RGB和X模态合并为6通道图像
        
        Args:
            rgb_img (np.ndarray): RGB图像 [H,W,3]
            x_img (np.ndarray): X模态图像 [H,W,3]
            
        Returns:
            np.ndarray: 6通道合并图像 [H,W,6]
        """
        return np.concatenate([rgb_img, x_img], axis=2)
    
    @staticmethod
    def validate_input(img):
        """
        验证输入图像格式
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            bool: 如果是有效的6通道图像返回True
        """
        return len(img.shape) == 3 and img.shape[2] == 6


# 示例：可扩展的多模态几何变换
class MultiModalRandomFlip(BaseMultiModalTransform):
    """
    多模态随机翻转增强
    
    对6通道图像进行同步的水平或垂直翻转，
    确保RGB和X模态保持对应关系。
    
    Attributes:
        p (float): 翻转概率 [0, 1]
        direction (str): 翻转方向 'horizontal' 或 'vertical'
    """
    
    def __init__(self, p=0.5, direction="horizontal"):
        """
        初始化多模态随机翻转
        
        Args:
            p (float): 翻转概率
            direction (str): 翻转方向
        """
        assert direction in {"horizontal", "vertical"}, f"direction must be 'horizontal' or 'vertical', got {direction}"
        assert 0 <= p <= 1.0, f"probability must be in [0, 1], got {p}"
        
        self.p = p
        self.direction = direction
        
    def __call__(self, labels):
        """
        应用多模态随机翻转
        
        Args:
            labels (Dict): 包含图像的标签字典
            
        Returns:
            Dict: 处理后的标签字典
        """
        img = labels["img"]
        
        if not self.validate_input(img):
            LOGGER.warning(f"MultiModalRandomFlip expects 6-channel image, got shape {img.shape}")
            return labels
            
        # 根据概率决定是否翻转
        if np.random.random() > self.p:
            return labels
            
        # 同步翻转整个6通道图像
        if self.direction == "horizontal":
            img = np.fliplr(img)
        else:  # vertical
            img = np.flipud(img)
            
        labels["img"] = np.ascontiguousarray(img)
        return labels


# 工具函数
def create_multimodal_transforms(rgb_transforms, preserve_x_modality=True):
    """
    从标准RGB变换创建多模态变换的工厂函数
    
    Args:
        rgb_transforms (list): RGB变换列表
        preserve_x_modality (bool): 是否保护X模态不受变换影响
        
    Returns:
        list: 适配的多模态变换列表
        
    Notes:
        这是一个扩展接口，用于将来可能的自动适配功能
    """
    # 这里可以实现自动适配逻辑
    # 目前返回空列表，作为接口预留
    LOGGER.info("create_multimodal_transforms is under development")
    return []


# 多模态Mosaic和MixUp类
class MultiModalMosaic(Mosaic):
    """
    多模态Mosaic增强类
    
    继承自标准Mosaic类，确保随机选择的图像索引都有完整的多模态数据。
    通过调用数据集的get_valid_indices()方法获取有效索引列表。
    """
    
    def get_indexes(self, buffer=True):
        """
        获取多模态Mosaic拼接的随机索引
        
        重写父类方法，确保选择的索引都有完整的多模态数据
        
        Args:
            buffer (bool): 是否从buffer选择图像（与父类保持兼容）
            
        Returns:
            list: n-1个随机有效索引（n为mosaic网格数）
        """
        # 获取有效的多模态索引
        if hasattr(self.dataset, 'get_valid_indices'):
            valid_indices = self.dataset.get_valid_indices()
        else:
            # 向后兼容：如果数据集没有get_valid_indices方法，使用全部索引
            valid_indices = list(range(len(self.dataset)))
            LOGGER.warning("Dataset does not have get_valid_indices method, using all indices for Mosaic")
        
        # 需要的索引数量：n-1（n是mosaic网格数，默认4，所以需要3个额外图像）
        num_needed = self.n - 1
        
        # 从有效索引中随机选择
        if len(valid_indices) < num_needed:
            LOGGER.warning(f"Not enough valid multimodal images ({len(valid_indices)}) for Mosaic augmentation")
            # 如果有效索引不够，用重复选择策略
            if len(valid_indices) == 0:
                return [0] * num_needed  # 极端情况，返回默认索引
            # 为了确保有重复，我们从有效索引中随机选择，允许重复
            return [valid_indices[np.random.randint(0, len(valid_indices))] for _ in range(num_needed)]
        
        # 随机选择num_needed个不同的有效索引
        selected_indices = np.random.choice(len(valid_indices), num_needed, replace=False)
        return [valid_indices[i] for i in selected_indices]


class MultiModalMixUp(MixUp):
    """
    多模态MixUp增强类
    
    继承自标准MixUp类，确保随机选择的图像索引都有完整的多模态数据。
    通过调用数据集的get_valid_indices()方法获取有效索引列表。
    """
    
    def get_indexes(self):
        """
        获取多模态MixUp混合的随机索引
        
        重写父类方法，确保选择的索引都有完整的多模态数据
            
        Returns:
            list: 1个随机有效索引
        """
        # 获取有效的多模态索引
        if hasattr(self.dataset, 'get_valid_indices'):
            valid_indices = self.dataset.get_valid_indices()
        else:
            # 向后兼容：如果数据集没有get_valid_indices方法，使用全部索引
            valid_indices = list(range(len(self.dataset)))
            LOGGER.warning("Dataset does not have get_valid_indices method, using all indices for MixUp")
        
        # 从有效索引中随机选择1个（MixUp需要2个图像，包括当前1个+随机1个）
        if len(valid_indices) < 2:
            LOGGER.warning(f"Not enough valid multimodal images ({len(valid_indices)}) for MixUp augmentation")
            return [valid_indices[0] if valid_indices else 0]
        
        # 随机选择1个有效索引
        return [valid_indices[np.random.choice(len(valid_indices))]]


__all__ = [
    "MultiModalRandomHSV",
    "BaseMultiModalTransform", 
    "MultiModalRandomFlip",
    "create_multimodal_transforms",
    "MultiModalMosaic",
    "MultiModalMixUp",
] 