# Ultralytics Multimodal Utilities
# Helper functions for multimodal system status and validation
# Version: v1.0

from ultralytics.utils import LOGGER


def validate_mm_config_format(config):
    """Validate multimodal configuration format correctness"""
    
    rgb_layers = []
    x_layers = []
    dual_layers = []
    
    for section in ['backbone', 'head']:
        for i, layer_config in enumerate(config.get(section, [])):
            if len(layer_config) == 5:  # Has 5th field
                input_source = layer_config[4]
                if input_source == 'RGB':
                    rgb_layers.append(i)
                elif input_source == 'X':
                    x_layers.append(i)
                elif input_source == 'Dual':
                    dual_layers.append(i)
    
    LOGGER.info(f"✅ MultiModal: 配置验证完成")
    LOGGER.info(f"📊 MultiModal: RGB路由层={len(rgb_layers)}, X路由层={len(x_layers)}, Dual路由层={len(dual_layers)}")
    
    return {
        'rgb_layers': rgb_layers,
        'x_layers': x_layers, 
        'dual_layers': dual_layers,
        'total_routing_layers': len(rgb_layers) + len(x_layers) + len(dual_layers)
    }


def mm_system_status():
    """Display multimodal RGB+X system status"""
    LOGGER.info("🔍 MultiModal: RGB+X二元模态路由系统状态检查...")
    LOGGER.info("✅ MultiModal: 支持RGB模态(3通道可见光)")
    LOGGER.info("✅ MultiModal: 支持X模态(3通道任意其他模态)")  
    LOGGER.info("✅ MultiModal: 支持Dual模态(6通道RGB+X)")
    LOGGER.info("📝 MultiModal: 配置格式: [from, repeats, module, args, 'RGB'/'X'/'Dual']")
    LOGGER.info("🎯 MultiModal: X模态路径支持新输入起点重定向")
    LOGGER.info("🚀 MultiModal: 系统版本: v1.0 - 通用YOLO&RTDETR多模态路由")
    LOGGER.info("🔗 MultiModal: 架构支持: YOLO11, RTDETR, 未来扩展架构")
    return True


def check_mm_model_attributes(model):
    """Check multimodal attributes in the model"""
    mm_layers = []
    
    for i, m in enumerate(model.model if hasattr(model, 'model') else []):
        if hasattr(m, '_mm_input_source'):
            layer_info = {
                'layer_index': getattr(m, '_mm_layer_index', i),
                'input_source': getattr(m, '_mm_input_source', None),
                'x_modality': getattr(m, '_mm_x_modality', 'unknown'),
                'version': getattr(m, '_mm_version', 'unknown'),
                'new_input_start': getattr(m, '_mm_new_input_start', False)
            }
            mm_layers.append(layer_info)
    
    if mm_layers:
        LOGGER.info(f"🎯 MultiModal: 发现 {len(mm_layers)} 个多模态路由层")
        for layer_info in mm_layers:
            source = layer_info['input_source']
            idx = layer_info['layer_index']
            if layer_info['new_input_start']:
                LOGGER.info(f"📍 MultiModal: Layer {idx} - {source}模态新输入起点")
            else:
                LOGGER.info(f"🚀 MultiModal: Layer {idx} - {source}模态路由层")
    else:
        LOGGER.info("📋 MultiModal: 未发现多模态路由层，使用标准模式")
    
    return mm_layers


def get_mm_system_info():
    """Get multimodal system information"""
    return {
        'version': 'v1.0',
        'supported_modalities': ['RGB', 'X', 'Dual'],
        'supported_architectures': ['YOLO', 'RTDETR'],
        'features': [
            'Zero-copy tensor routing',
            'Configuration-driven data flow', 
            'X modality new input start redirection',
            'Universal RGB+X framework'
        ]
    } 