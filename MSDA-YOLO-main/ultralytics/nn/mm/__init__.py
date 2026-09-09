# Ultralytics Multimodal Extension - Universal RGB+X Framework
# Supports both YOLO and RTDETR architectures
# Author: YOLOMM Team
# Version: v1.0

from .router import MultiModalRouter
from .parser import MultiModalConfigParser  
from .utils import (
    validate_mm_config_format,
    mm_system_status,
    check_mm_model_attributes
)

__all__ = [
    'MultiModalRouter',
    'MultiModalConfigParser', 
    'validate_mm_config_format',
    'mm_system_status',
    'check_mm_model_attributes'
]

__version__ = 'v1.0' 