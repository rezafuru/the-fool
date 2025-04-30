from typing import Optional

import torch
from torch import Tensor, nn

from model.registry import register_context_component


@register_context_component
class MaskedConv2d(nn.Conv2d):
    """

    Default Context Prediction by Minnen et al. in https://arxiv.org/abs/1809.02736
    Originally introduced in https://arxiv.org/abs/1606.05328.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        mask_type: str = "A",
        *args,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels or 2 * in_channels,
            *args,
            **kwargs
        )

        assert mask_type in ("A", "B"), "Invalid Mask Type"
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


@register_context_component
class Conv1x1EntropyParameter(nn.Module):
    """
    Default Entropy Parameter by Minnen et al. in https://arxiv.org/abs/1809.02736
    """

    def __init__(self, gc_channels):
        super(Conv1x1EntropyParameter, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(gc_channels * 12 // 3, gc_channels * 10 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(gc_channels * 10 // 3, gc_channels * 8 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(gc_channels * 8 // 3, gc_channels * 6 // 3, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)
