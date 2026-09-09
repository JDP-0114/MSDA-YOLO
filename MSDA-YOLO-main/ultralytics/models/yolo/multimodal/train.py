# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.utils.torch_utils import de_parallel


class MultiModalDetectionTrainer(DetectionTrainer):
    """
    多模态检测训练器，专门处理RGB+X模态的训练流程。
    
    这个类继承DetectionTrainer，重写关键方法以支持多模态数据集和6通道输入。
    支持RGB+深度、RGB+热红外等多模态组合的完整训练流程。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化多模态检测训练器。
        
        Args:
            cfg (str | DictConfig, optional): 配置文件路径或配置字典
            overrides (dict, optional): 配置覆盖参数
            _callbacks (list, optional): 回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"  # 确保任务类型正确
        super().__init__(cfg, overrides, _callbacks)
        
        # Get modality parameter from standard cfg system (与推理器保持一致)
        # Modality validation is handled by cfg system, no local validation needed
        self.modality = getattr(self.args, 'modality', None)
        
        # Initialize modality-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # Log initialization with modality information
        if self.modality:
            LOGGER.info(f"初始化MultiModalDetectionTrainer - 单模态训练模式: {self.modality}-only")
        else:
            LOGGER.info("初始化MultiModalDetectionTrainer - 双模态训练模式")

    def _parse_multimodal_config(self):
        """
        解析和验证数据配置文件中的多模态设置。
        
        解析data.yaml中的modalities和models字段，确保配置正确性，
        提供默认配置和友好的错误信息。
        
        优先支持用户指定的单模态训练参数。
        
        Returns:
            dict: 解析后的多模态配置
            
        Raises:
            ValueError: 当多模态配置不正确时
        """
        # 优先检查用户指定的modality参数（单模态训练）
        if self.modality:
            # 构建单模态配置
            if self.modality == 'rgb':
                # RGB单模态：使用RGB + 默认X模态（depth）进行零填充
                config = {
                    'models': ['rgb', 'depth'],
                    'modalities': {
                        'rgb': 'images',
                        'depth': 'images_depth'
                    }
                }
            else:
                # X模态单模态：使用RGB + 指定X模态进行零填充
                config = {
                    'models': ['rgb', self.modality],
                    'modalities': {
                        'rgb': 'images',
                        self.modality: f'images_{self.modality}'
                    }
                }
            
            LOGGER.info(f"训练使用用户指定的单模态配置: {self.modality}-only")
            return config
        
        # 双模态训练：使用原有配置解析逻辑
        config = {
            'models': ['rgb', 'depth'],  # 默认模态组合
            'modalities': {  # 默认模态路径映射
                'rgb': 'images',
                'depth': 'images_depth'
            }
        }
        
        if not self.data:
            LOGGER.warning("训练器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config
        
        # 解析models字段（使用的模态组合）
        if 'models' in self.data:
            models = self.data['models']
            
            # 验证models格式
            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'models'必须是列表格式，当前为: {type(models)}")
            
            if len(models) != 2:
                raise ValueError(f"多模态检测要求恰好2个模态，当前提供: {len(models)} - {models}")
            
            if 'rgb' not in models:
                raise ValueError(f"多模态组合必须包含'rgb'模态，当前: {models}")
            
            config['models'] = models
            LOGGER.info(f"使用配置中的模态组合: {models}")
        else:
            LOGGER.info(f"未找到'models'配置，使用默认组合: {config['models']}")
        
        # 解析modalities字段（模态路径映射）
        if 'modalities' in self.data:
            modalities = self.data['modalities']
            
            # 验证modalities格式
            if not isinstance(modalities, dict):
                raise ValueError(f"data.yaml中的'modalities'必须是字典格式，当前为: {type(modalities)}")
            
            # 验证所有必需模态都有路径配置
            for modality in config['models']:
                if modality not in modalities:
                    if modality == 'rgb':
                        modalities[modality] = 'images'  # RGB默认路径
                        LOGGER.warning(f"未找到'{modality}'模态路径配置，使用默认: images")
                    else:
                        modalities[modality] = f'images_{modality}'  # X模态默认路径
                        LOGGER.warning(f"未找到'{modality}'模态路径配置，使用默认: images_{modality}")
            
            config['modalities'] = modalities
            LOGGER.info(f"使用配置中的模态路径映射: {modalities}")
        else:
            # 为当前模态组合生成默认路径映射
            x_modality = [m for m in config['models'] if m != 'rgb'][0]
            config['modalities']['rgb'] = 'images'
            config['modalities'][x_modality] = f'images_{x_modality}'
            LOGGER.info(f"未找到'modalities'配置，生成默认路径映射: {config['modalities']}")
        
        # ✅ 移除硬编码限制，改为配置驱动
        # 用户通过配置明确指定了模态类型，系统应该信任并支持
        x_modality = [m for m in config['models'] if m != 'rgb'][0]
        LOGGER.info(f"✅ 使用用户配置的X模态: {x_modality} (配置驱动，支持任意模态类型)")
        
        return config

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态数据集。
        
        重写父类方法，通过传递multi_modal_image=True参数启用YOLOMultiModalImageDataset，
        实现RGB+X模态的6通道数据加载和处理。
        
        Args:
            img_path (str): 图像路径
            mode (str): 模式（train/val/test）
            batch (int, optional): 批次大小
            
        Returns:
            YOLOMultiModalImageDataset: 多模态数据集对象
        """
        # 获取模型stride参数（与DetectionTrainer保持一致）
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        
        # 懒加载：按需解析多模态配置
        if not hasattr(self, 'multimodal_config'):
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态配置解析完成 - 模态: {self.multimodal_config['models']}")
        
        # 使用解析后的模态配置
        modalities = self.multimodal_config['models']
        
        LOGGER.info(f"构建多模态数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")
        
        # 如果启用单模态训练，记录模态填充信息
        if self.modality:
            LOGGER.info(f"启用单模态训练: {self.modality}-only，将应用智能模态填充")
        
        # 调用build_yolo_dataset，传递multi_modal_image=True启用多模态数据集
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode, 
            rect=mode == "val",  # 验证模式使用矩形训练
            stride=gs,
            multi_modal_image=True,  # 关键参数：启用YOLOMultiModalImageDataset
            modalities=modalities,  # 传递模态配置
            train_modality=self.modality,  # 新增参数：传递模态控制参数
        )

    def get_validator(self):
        """
        返回多模态检测验证器。
        
        重写父类方法，确保使用与训练阶段一致的多模态验证器，
        避免在验证阶段出现通道数不匹配的问题。
        
        Returns:
            MultiModalDetectionValidator: 多模态检测验证器实例
        """
        from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
        
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = MultiModalDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        
        # 关键修复：手动设置data属性，确保验证器能正确获取多模态配置
        validator.data = self.data
        validator.stride = self.model.stride if self.model else 32
        
        return validator

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本及其标注。
        
        将6通道输入拆分为RGB和X模态，采用多种可视化策略：
        1. 分离保存：RGB和X模态分别保存
        2. 并排显示：在同一图像中展示两种模态
        3. 智能处理X模态的可视化（单通道重复或真实3通道）
        
        Args:
            batch (dict): 包含图像、标注等信息的批次数据
            ni (int): 批次索引
        """
        import numpy as np
        import cv2
        from ultralytics.utils.plotting import plot_images
        
        # 获取6通道图像数据
        multimodal_images = batch["img"]  # Shape: (batch, 6, H, W)
        
        # 分离RGB和X模态 - 修复：交换通道分离顺序
        # 如果数据加载时顺序反了，这里通过交换来修正
        # LOGGER.info("注意：交换了RGB和X模态的通道分离顺序来修复可视化问题")
        rgb_images = multimodal_images[:, 3:, :, :]      # 后3通道：实际的RGB（修复）
        x_modal_images = multimodal_images[:, :3, :, :]  # 前3通道：实际的X模态（修复）
        
        # 获取X模态类型用于可视化处理
        x_modality = [m for m in self.multimodal_config['models'] if m != 'rgb'][0]
        
        # 处理X模态数据用于可视化
        x_visual = self._process_x_modality_for_visualization(x_modal_images, x_modality)
        
        # 策略1：分别保存RGB和X模态
        # LOGGER.info(f"保存RGB模态可视化 - 批次 {ni}")
        plot_images(
            images=rgb_images,
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_rgb.jpg",
            on_plot=self.on_plot,
        )
        
        # LOGGER.info(f"保存{x_modality}模态可视化 - 批次 {ni}")
        plot_images(
            images=x_visual,
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
            fname=self.save_dir / f"train_batch{ni}_{x_modality}.jpg",
            on_plot=self.on_plot,
        )
        
        # 策略2：创建并排可视化
        try:
            side_by_side_images = self._create_side_by_side_visualization(rgb_images, x_visual)
            plot_images(
                images=side_by_side_images,
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=self._adjust_bboxes_for_side_by_side(batch["bboxes"]),
                paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                fname=self.save_dir / f"train_batch{ni}_multimodal.jpg",
                on_plot=self.on_plot,
            )
            # LOGGER.info(f"保存多模态合成可视化 - 批次 {ni}")
        except Exception as e:
            # LOGGER.warning(f"并排可视化创建失败: {e}")
            pass

    def _process_x_modality_for_visualization(self, x_modal_images, x_modality):
        """
        处理X模态数据用于可视化。
        
        根据X模态类型进行适当的可视化处理：
        - 对于单通道重复的模态（如深度图），提取单通道并应用伪彩色
        - 对于真实3通道模态（如热红外RGB），直接使用
        
        Args:
            x_modal_images (torch.Tensor): X模态图像数据 (batch, 3, H, W)
            x_modality (str): X模态类型
            
        Returns:
            torch.Tensor: 处理后的3通道可视化数据
        """
        import torch
        import numpy as np
        import cv2
        
        # 检查是否为单通道重复（如深度图的 [D,D,D] 格式）
        if torch.allclose(x_modal_images[:, 0:1, :, :], x_modal_images[:, 1:2, :, :]) and \
           torch.allclose(x_modal_images[:, 1:2, :, :], x_modal_images[:, 2:3, :, :]):
            
            # LOGGER.info(f"检测到单通道重复的{x_modality}模态，应用伪彩色映射")
            
            # 提取单通道数据
            single_channel = x_modal_images[:, 0:1, :, :]  # (batch, 1, H, W)
            
            # 应用伪彩色映射
            colorized_images = []
            for i in range(single_channel.shape[0]):
                # 转换为numpy并归一化到0-255
                img_np = single_channel[i, 0].cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                # 应用颜色映射（根据模态类型选择）
                if x_modality in ['depth']:
                    colormap = cv2.COLORMAP_PLASMA  # 深度用等离子体色彩
                elif x_modality in ['thermal', 'infrared', 'ir']:
                    colormap = cv2.COLORMAP_INFERNO  # 热红外用地狱色彩
                else:
                    colormap = cv2.COLORMAP_JET  # 其他用彩虹色彩
                
                colored_img = cv2.applyColorMap(img_np, colormap)
                colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
                
                # 转换回tensor格式 (3, H, W)
                colored_tensor = torch.from_numpy(colored_img.transpose(2, 0, 1)).float()
                if colored_tensor.max() > 1.0:
                    colored_tensor /= 255.0
                    
                colorized_images.append(colored_tensor)
            
            return torch.stack(colorized_images)
        
        else:
            # LOGGER.info(f"使用{x_modality}模态的原始3通道数据")
            return x_modal_images

    def _create_side_by_side_visualization(self, rgb_images, x_images):
        """
        创建RGB和X模态的并排可视化。
        
        Args:
            rgb_images (torch.Tensor): RGB图像 (batch, 3, H, W)
            x_images (torch.Tensor): X模态图像 (batch, 3, H, W)
            
        Returns:
            torch.Tensor: 并排的可视化图像 (batch, 3, H, 2*W)
        """
        import torch
        
        batch_size, channels, height, width = rgb_images.shape
        
        # 创建并排图像 (batch, 3, H, 2*W)
        side_by_side = torch.zeros(batch_size, channels, height, width * 2, 
                                 dtype=rgb_images.dtype, device=rgb_images.device)
        
        # 左侧放RGB，右侧放X模态
        side_by_side[:, :, :, :width] = rgb_images
        side_by_side[:, :, :, width:] = x_images
        
        return side_by_side

    def _adjust_bboxes_for_side_by_side(self, bboxes):
        """
        调整边界框坐标用于并排可视化。
        
        为了在并排图像中正确显示标注，需要：
        1. 复制原始边界框到左侧（RGB区域）
        2. 将边界框平移到右侧（X模态区域）
        
        Args:
            bboxes (torch.Tensor): 原始边界框 (N, 6) [batch_idx, cls, x, y, w, h]
            
        Returns:
            torch.Tensor: 调整后的边界框，包含两个区域的标注
        """
        import torch
        
        if len(bboxes) == 0:
            return bboxes
        
        # 复制原始边界框
        left_bboxes = bboxes.clone()  # RGB区域的边界框
        right_bboxes = bboxes.clone()  # X模态区域的边界框
        
        # 将右侧边界框的x坐标平移0.5（因为图像宽度翻倍了，相对坐标需要调整）
        # YOLO格式的坐标是相对坐标 [0,1]，所以平移0.5相当于向右移动一半图像宽度
        right_bboxes[:, 2] = (right_bboxes[:, 2] + 1.0) / 2.0  # x坐标平移并缩放
        left_bboxes[:, 2] = left_bboxes[:, 2] / 2.0  # 左侧x坐标缩放
        
        # 宽度坐标也需要相应缩放
        left_bboxes[:, 4] = left_bboxes[:, 4] / 2.0   # 宽度缩放
        right_bboxes[:, 4] = right_bboxes[:, 4] / 2.0  # 宽度缩放
        
        # 合并两个区域的边界框
        combined_bboxes = torch.cat([left_bboxes, right_bboxes], dim=0)
        
        return combined_bboxes 