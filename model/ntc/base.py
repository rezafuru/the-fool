import torch
from timm.models.layers import trunc_normal_
from torch import nn
from abc import ABC


class BaseNTC(nn.Module, ABC):
    """
    Base class of non-linear transforms for data compression
    """

    def __init__(self):
        super().__init__()

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^conv1|bn1|maxpool",
            blocks=r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)",
        )
        return matcher
