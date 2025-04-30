from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from torch import Tensor, nn

from model.layers import ConvBlock
from model.ntc.image.image_base import ModularImageCompressionModel
from model.registry import get_synthesis_network, register_bottleneck_wrapper

import torch


def get_drop_indices(logits: Tensor) -> Tensor:
    """
    Given a tensor of logits, return the indices where the 0th class has the highest score.

    Parameters:
    - logits (torch.Tensor): A tensor of shape (batch_size, n_classes) containing the scores for each class.

    Returns:
    - indices (torch.Tensor): A tensor containing the indices of the samples to be dropped.
    """

    # Find the class with the maximum score for each sample in the batch
    _, max_indices = torch.max(logits, dim=1)

    # Get the indices where the 0th class has the highest score
    drop_indices = (max_indices == 0).nonzero(as_tuple=True)[0]

    return drop_indices


class BottleneckWrapper(nn.Module):
    def __init__(self, compressor: ModularImageCompressionModel):
        super().__init__()
        self.compressor = compressor

    def compress(self, x: Tensor, *args, **kwargs):
        return self.compressor.compress(x, *args, **kwargs)

    def decompress(self, strings: Tuple[List[str], List[str]], shape: torch.Size, *args, **kwargs):
        return self.compressor.decompress(strings, shape, *args, **kwargs)

    def update(self, *args, **kwargs):
        return self.compressor.update(*args, **kwargs)

    @property
    def entropy_bottleneck(self) -> EntropyBottleneck:
        return self.compressor.entropy_bottleneck

    @property
    def gaussian_conditional(self) -> Optional[GaussianConditional]:
        if hasattr(self.compressor, "gaussian_conditional"):
            return self.compressor.gaussian_conditional
        return None

    @property
    def g_a(self) -> nn.Module:
        return NotImplemented

    @property
    def g_s(self) -> nn.Module:
        return NotImplemented

    def aux_loss(self) -> Tensor:
        return self.compressor.aux_loss()


class NeuralThresholdFilterWrapper(BottleneckWrapper):
    # todo: like below, but uses sigmoid with a threshold values (default 0.5)
    pass


@register_bottleneck_wrapper
class ImageRestorationWrapper(BottleneckWrapper):
    def __init__(
            self,
            compressor: ModularImageCompressionModel,
            restoration_synthesis_config: Dict[str, Any],
    ):
        super().__init__(compressor)
        self.restor_g_s = get_synthesis_network(
            restoration_synthesis_config["name"],
            **restoration_synthesis_config["params"]
        )

    def forward(self, x: Tensor, use_restoration: bool = False):
        if use_restoration:
            orig_g_s = self.compressor.g_s
            self.compressor.g_s = self.restor_g_s
            res = self.compressor(x)
            self.compressor.g_s = orig_g_s
            return res
        return self.compressor(x)

    def decompress(self, *args, **kwargs):
        # y_h = self.compressor(*args, **kwargs)
        orig_g_s = self.compressor.g_s
        self.compressor.g_s = self.restor_g_s
        x_h = self.decompress(*args, **kwargs)
        self.compressor.g_s = orig_g_s
        return x_h
