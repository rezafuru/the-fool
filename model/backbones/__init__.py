from model.backbones.pruned_torchvision import get_pruned_resnet50
from model.backbones.timm_models import get_timm_model
from model.backbones.ultralytics_models import get_ultralytics_model

__all__ = ["get_ultralytics_model", "get_timm_model", "get_pruned_resnet50"]
