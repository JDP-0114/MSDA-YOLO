# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load
from ultralytics.cfg import DEFAULT_CFG_DICT


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes


class YOLOMM(Model):
    """YOLO MultiModal (YOLOMM) object detection model for RGB+X modalities."""

    def __init__(self, model="yolo11n-mm.yaml", task="detect", verbose=False):
        """
        Initialize YOLOMM (YOLO MultiModal) model.

        Args:
            model (str | Path): Path to the model configuration file (.yaml) or weights (.pt).
            task (str): Task type, fixed to 'detect' for object detection.
            verbose (bool): If True, prints additional information during initialization.
        """
        # Force task to be 'detect' for multimodal detection
        super().__init__(model=model, task="detect", verbose=verbose)
        
        # 验证模型是否为多模态配置
        if hasattr(self.model, 'ch') and self.model.ch != 6:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"YOLOMM期望6通道输入，但模型配置为{self.model.ch}通道")

    @property
    def task_map(self):
        """Map task to model, trainer, validator, and predictor classes for multimodal detection."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.multimodal.MultiModalDetectionTrainer,
                "validator": yolo.multimodal.MultiModalDetectionValidator,
                "predictor": yolo.multimodal.MultiModalDetectionPredictor,  # 使用多模态预测器
            }
        }

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initialize a new model and inference mode for YOLOMM with flexible channel configuration.
        
        Args:
            cfg (str): Model configuration file path
            task (str): Task type (forced to 'detect')
            model: Existing model (optional)
            verbose (bool): Verbose output
        """
        from ultralytics.utils import LOGGER
        from ultralytics.nn.tasks import yaml_model_load
        
        # 强制任务类型为detect
        task = "detect"
        cfg_dict = yaml_model_load(cfg)
        
        # 设置配置和任务
        self.cfg = cfg
        self.task = task
        
        # 让多模态路由器决定输入通道配置，而非强制6通道
        # 早期融合：ch=6，中期融合：ch=3（让路由器处理模态分发）
        self.model = model or self.task_map[task]["model"](cfg_dict, ch=3, verbose=verbose)  # 使用标准3通道，让多模态路由器处理
        
        # 设置基本属性
        self.ckpt = None
        self.ckpt_path = None
        
        # 重要：设置overrides字典，包含train方法需要的"model"和"task"键
        self.overrides = {
            "model": self.cfg,  # 这是train方法需要访问的键
            "task": self.task,
        }
        
        self.metrics = None
        self.session = None
        
        # 设置模型属性（与父类保持一致）
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg
        
        if verbose:
            LOGGER.info(f"YOLOMM model initialized: {cfg}")

    def train(self, trainer=None, **kwargs):
        """
        Train the YOLOMM model using multimodal dataset.
        
        Args:
            trainer: Custom trainer (optional)
            **kwargs: Additional training arguments including modality parameter
            
        Returns:
            Training results
        """
        # 处理modality参数：保留参数在kwargs中传递给训练器
        if 'modality' in kwargs:
            if 'modality' not in self.overrides:
                self.overrides['modality'] = kwargs['modality']  # 不使用pop()，保留参数
        
        # 确保使用多模态训练器
        if trainer is None:
            trainer = self.task_map[self.task]["trainer"]
        
        return super().train(trainer=trainer, **kwargs)

    def val(self, validator=None, **kwargs):
        """
        Validate the YOLOMM model using multimodal validation dataset.
        
        Args:
            validator: Custom validator (optional)
            **kwargs: Additional validation arguments including modality parameter
            
        Returns:
            Validation results
        """
        # 处理modality参数：保留参数在kwargs中传递给验证器
        if 'modality' in kwargs:
            if 'modality' not in self.overrides:
                self.overrides['modality'] = kwargs['modality']  # 不使用pop()，保留参数
        
        # 确保使用多模态验证器
        if validator is None:
            validator = self.task_map[self.task]["validator"]
        
        return super().val(validator=validator, **kwargs)

    def predict(self, source=None, modality=None, **kwargs):
        """
        Perform object detection prediction on the given image(s) with optional modality control.
        
        Args:
            source (str | list): Path to image(s) or list of image paths for dual-modal inference
            modality (str, optional): Modality mode - 'rgb', 'depth', 'thermal', etc. for single-modal inference
            **kwargs: Additional prediction arguments
            
        Returns:
            Prediction results
        """
        # 模态推理逻辑（具体实现在推理器中完成）
        
        # 处理modality参数：通过添加到kwargs中，让父类的predict方法处理
        if modality:
            kwargs['modality'] = modality
        
        # 调用父类predict方法，让它正确处理参数（包括save等）
        return super().predict(source=source, **kwargs)
