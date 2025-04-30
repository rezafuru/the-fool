import os
import types
from collections import OrderedDict
from typing import Optional, Union

import torch
import timm
from pytorchyolo.models import Darknet

from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_func
import compressai.zoo as cai_zoo
import torch.nn.functional as F

from pytorchyolo import models as torchyolo

model = torchyolo.load_model(
    model_path="torchyolo/yolov3.cfg",
    # "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights"
)

logger = def_logger.getChild(__name__)

_HEAD_IDX = {
    "yolov3": 79,
    "yolov3-tiny": 16,
}


class TorchYoloWrapper(nn.Module):
    def __init__(self, darknet: Darknet, model_name: str):
        super().__init__()
        head_idx = _HEAD_IDX[model_name]
        self.darknet = darknet
        self.head = nn.Sequential(*darknet.module_list[:head_idx])
        self.darknet.module_list = self.darknet.module_list[head_idx:]
        self.darknet.module_defs = self.darknet.module_defs[head_idx:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.darknet(x)
        return x


@register_model_func
def get_torchyolo_model(
    model_name: str,
    model_path: Union[str, bytes, os.PathLike],
    weights_path: Union[str, bytes, os.PathLike] = None,
) -> TorchYoloWrapper:

    model = torchyolo.load_model(model_path=model_path, weights_path=weights_path)
    return TorchYoloWrapper(model, model_name)
