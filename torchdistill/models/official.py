from torch.nn import SyncBatchNorm
from torchvision import models

OFFICIAL_MODEL_DICT = dict()
OFFICIAL_MODEL_DICT.update(models.__dict__)
OFFICIAL_MODEL_DICT.update(models.detection.__dict__)
OFFICIAL_MODEL_DICT.update(models.segmentation.__dict__)


def get_image_classification_model(model_config, distributed=False):
    model_name = model_config["name"]
    quantized = model_config.get("quantized", False)
    if not quantized and model_name in models.__dict__:
        model = models.__dict__[model_name](**model_config["params"])
    elif quantized and model_name in models.quantization.__dict__:
        model = models.quantization.__dict__[model_name](**model_config["params"])
    else:
        return None

    sync_bn = model_config.get("sync_bn", False)
    if distributed and sync_bn:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def get_object_detection_model(model_config):
    model_name = model_config["name"]
    if model_name not in models.detection.__dict__:
        return None
    return models.detection.__dict__[model_name](**model_config["params"])


def get_semantic_segmentation_model(model_config):
    model_name = model_config["name"]
    if model_name not in models.segmentation.__dict__:
        return None
    return models.segmentation.__dict__[model_name](**model_config["params"])


def get_vision_model(model_config):
    model_name = model_config["name"]
    return OFFICIAL_MODEL_DICT[model_name](**model_config["params"])
