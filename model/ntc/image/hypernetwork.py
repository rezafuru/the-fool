from abc import abstractmethod
from functools import partial
from typing import Any, Mapping, Optional

import torch
from torch import nn

from model.layers import ResidualBlockWithStride, conv3x3, conv5x5
from model.ntc.base import BaseNTC
from model.registry import register_hyper_network
from model.se import ChannelSELayer, ChannelSpatialSELayer, SpatialSELayer


class HyperNetwork(BaseNTC):
    def __init__(self):
        super(HyperNetwork, self).__init__()

    @property
    def hyper_analysis(self) -> nn.Module:
        return self.h_a

    @property
    def hyper_synthesis(self) -> nn.Module:
        return self.h_s


@register_hyper_network
class BaselineSHPHyperNetwork(HyperNetwork):
    def __init__(
        self, eb_channels: int, gc_channels: int, in_channels: Optional[int] = None
    ):
        super().__init__()
        self.h_a = nn.Sequential(
            conv3x3(in_channels or gc_channels, eb_channels, stride=1),
            nn.LeakyReLU(inplace=False),
            conv5x5(eb_channels, eb_channels, stride=2),
            nn.LeakyReLU(inplace=False),
            conv5x5(eb_channels, eb_channels, stride=2),
        )
        self.h_s = nn.Sequential(
            conv5x5(in_ch=eb_channels, out_ch=eb_channels, stride=2, upsample=True),
            nn.LeakyReLU(inplace=False),
            conv5x5(in_ch=eb_channels, out_ch=eb_channels, stride=2, upsample=True),
            nn.LeakyReLU(inplace=False),
            conv3x3(in_ch=eb_channels, out_ch=gc_channels, stride=1),
            nn.LeakyReLU(inplace=False),
        )


@register_hyper_network
class HypernetworkWithResidualBlocks(HyperNetwork):
    def __init__(self, eb_channels: int, gc_channels: int, mshp: bool = False):
        super().__init__()
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(
                gc_channels,
                eb_channels,
                stride=1,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
            ResidualBlockWithStride(
                eb_channels,
                eb_channels,
                stride=2,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
            ResidualBlockWithStride(
                eb_channels,
                eb_channels,
                stride=2,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
        )
        self.h_s = nn.Sequential(
            ResidualBlockWithStride(
                eb_channels,
                gc_channels,
                stride=2,
                upsample=True,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
            ResidualBlockWithStride(
                gc_channels,
                gc_channels * 3 // 2 if mshp else gc_channels,
                stride=2,
                upsample=True,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
            ResidualBlockWithStride(
                gc_channels * 3 // 2 if mshp else gc_channels,
                gc_channels * 2 if mshp else gc_channels,
                stride=1,
                activation=partial(nn.LeakyReLU, inplace=False),
            ),
        )


@register_hyper_network
class BaselineMSHPHyperNetwork(BaselineSHPHyperNetwork):
    def __init__(
        self, eb_channels: int, gc_channels: int, in_channels: Optional[int] = None
    ):
        super().__init__(eb_channels, gc_channels, in_channels)
        self.h_s = nn.Sequential(
            conv5x5(in_ch=eb_channels, out_ch=gc_channels, stride=2, upsample=True),
            nn.LeakyReLU(inplace=False),
            conv5x5(
                in_ch=gc_channels, out_ch=gc_channels * 3 // 2, stride=2, upsample=True
            ),
            nn.LeakyReLU(inplace=False),
            conv3x3(in_ch=gc_channels * 3 // 2, out_ch=gc_channels * 2, stride=1),
        )


@register_hyper_network
class SmallHyperNetwork(HyperNetwork):
    def __init__(self, eb_channels: int, gc_channels: int):
        super().__init__()
        self.h_a = nn.Sequential(
            conv5x5(gc_channels, eb_channels, stride=2),
            nn.LeakyReLU(inplace=False),
            conv5x5(eb_channels, eb_channels, stride=2),
        )
        self.h_s = nn.Sequential(
            conv5x5(
                in_ch=eb_channels, out_ch=gc_channels * 3 // 2, stride=2, upsample=True
            ),
            nn.LeakyReLU(inplace=False),
            conv5x5(
                in_ch=gc_channels * 3 // 2,
                out_ch=gc_channels * 2,
                stride=2,
                upsample=True,
            ),
            nn.LeakyReLU(inplace=False),
        )


@register_hyper_network
class HypernetworkWithAttention(HyperNetwork):
    def __init__(
        self,
        eb_channels: int,
        gc_channels: int,
        in_channels: Optional[int] = None,
        mshp: bool = True,
        se_type: str = "CSSE",
    ):
        assert se_type in ["CSSE", "CSE", "SSE"]
        super().__init__()

        if se_type == "CSSE":
            se_block = partial(ChannelSpatialSELayer, reduction_ratio=2)
        elif se_type == "CSE":
            se_block = partial(ChannelSELayer, reduction_rate=2)
        else:
            se_block = SpatialSELayer
        self.h_a = nn.Sequential(
            nn.Sequential(
                conv3x3(in_channels or gc_channels, eb_channels, stride=1),
                nn.LeakyReLU(inplace=False),
                se_block(num_channels=eb_channels),
            ),
            nn.Sequential(
                conv5x5(eb_channels, eb_channels, stride=2),
                nn.LeakyReLU(inplace=False),
                se_block(num_channels=eb_channels),
            ),
            conv5x5(eb_channels, eb_channels, stride=2),
        )
        if mshp:
            self.h_s = nn.Sequential(
                nn.Sequential(
                    conv5x5(
                        in_ch=eb_channels, out_ch=gc_channels, stride=2, upsample=True
                    ),
                    nn.LeakyReLU(inplace=False),
                    se_block(num_channels=gc_channels),
                ),
                nn.Sequential(
                    conv5x5(
                        in_ch=gc_channels,
                        out_ch=gc_channels * 3 // 2,
                        stride=2,
                        upsample=True,
                    ),
                    nn.LeakyReLU(inplace=False),
                    se_block(num_channels=gc_channels * 3 // 2),
                ),
                conv3x3(in_ch=gc_channels * 3 // 2, out_ch=gc_channels * 2, stride=1),
            )
        else:
            self.h_s = nn.Sequential(
                nn.Sequential(
                    conv5x5(
                        in_ch=eb_channels, out_ch=eb_channels, stride=2, upsample=True
                    ),
                    nn.LeakyReLU(inplace=False),
                    se_block(num_channels=eb_channels),
                ),
                nn.Sequential(
                    conv5x5(
                        in_ch=eb_channels, out_ch=eb_channels, stride=2, upsample=True
                    ),
                    nn.LeakyReLU(inplace=False),
                    se_block(num_channels=eb_channels),
                ),
                nn.Sequential(
                    conv3x3(in_ch=eb_channels, out_ch=gc_channels, stride=1),
                    nn.LeakyReLU(inplace=False),
                ),
            )
