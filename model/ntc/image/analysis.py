from functools import partial
from typing import List, Optional, Tuple

import torch
from compressai.layers import AttentionBlock, GDN

from model.attention import AttentionBlock3D, CNNSelfAttention, CrossAttentionBlockCheng
from model.layers import (
    ConvBlock,
    ConvBlock3D,
    Identity_,
    ResidualBlockWithStride,
    ResidualBlockWithStride3D,
    conv,
)
from model.ntc.base import BaseNTC
from torch import Tensor, nn

from model.registry import register_analysis_network
from model.se import ChannelSELayer, ChannelSpatialSELayer, SpatialSELayer


@register_analysis_network
class FPBaselineAnalysis(BaseNTC):
    """

    128, 192 for quality level 1-4, 192, 320 for quality levels 5-8
    """

    def __init__(
            self,
            in_channels: int = 3,
            network_channels: int = 128,
            latent_channels: int = 192,
            use_relu: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            conv(kernel_size=5, in_ch=in_channels, out_ch=network_channels, stride=2),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels),
            conv(
                kernel_size=5, in_ch=network_channels, out_ch=network_channels, stride=2
            ),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels),
            conv(
                kernel_size=5, in_ch=network_channels, out_ch=network_channels, stride=2
            ),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels),
            conv(
                kernel_size=5, in_ch=network_channels, out_ch=latent_channels, stride=2
            ),
        )

    def forward(self, x):
        z = self.layers(x)
        return z


@register_analysis_network
class FrankenSplitAnalysis(BaseNTC):
    """Analysis network used in https://ieeexplore.ieee.org/document/10480247 for SVBI baseline"""

    def __init__(
            self,
            block_channels: List[int],
            target_channels: int,
            in_channels: int = 3,
            downsample_factor: int = 8,
            norm: bool = False,
            **kwargs
    ):
        super(FrankenSplitAnalysis, self).__init__()
        blocks = []
        for ch in block_channels:
            blocks.append(
                ResidualBlockWithStride(
                    in_ch=in_channels,
                    out_ch=ch,
                    activation=nn.ReLU,
                    stride=2 if downsample_factor >= 2 else 1,
                    norm=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03, affine=True)
                    if norm
                    else None,
                )
            )
            in_channels = ch
            downsample_factor //= 2
        blocks.append(
            ResidualBlockWithStride(
                in_ch=in_channels,
                out_ch=target_channels,
                activation=nn.ReLU,
                stride=2 if downsample_factor >= 2 else 1,
            )
        )
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layers(x)
        return x


@register_analysis_network
class SequencedAnalysisWith3DAttention(BaseNTC):
    def __init__(
            self,
            seq_len: int,
            block_channels: List[int],
            target_channels: int,
            in_channels: int = 3,
            downsample_factor: int = 8,
            use_attn: tuple[int] = (True, True, True),
            norm: bool = False,
            ptemp_w: Optional[int] = None,
            **kwargs
    ):
        assert len(block_channels) + 1 == len(use_attn)
        super().__init__()
        self.seq_len = seq_len
        blocks = []
        for idx, ch in enumerate(block_channels):
            blocks.append(
                nn.Sequential(
                    ResidualBlockWithStride(
                        in_ch=in_channels,
                        out_ch=ch,
                        activation=nn.ReLU,
                        stride=2 if downsample_factor >= 2 else 1,
                    ),
                    AttentionBlock3D(channels=ch, seq_len=self.seq_len, ptemp_w=ptemp_w)
                    if use_attn[idx]
                    else nn.Identity(),
                )
            )
            in_channels = ch
            downsample_factor //= 2
        blocks.append(
            nn.Sequential(
                ResidualBlockWithStride(
                    in_ch=in_channels,
                    out_ch=target_channels,
                    activation=nn.ReLU,
                    stride=2 if downsample_factor >= 2 else 1,
                ),
                AttentionBlock3D(
                    channels=target_channels, seq_len=self.seq_len, ptemp_w=ptemp_w
                )
                if use_attn[-1]
                else nn.Identity(),
            )
        )
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: List[Tensor] | Tensor) -> Tensor:
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
            B, D, C, H, W = x.shape
            x = x.reshape(B * D, C, H, W)

        x = self.layers(x)
        return x


@register_analysis_network
class AnalysisWithKeyPointsCrossAttention(BaseNTC):
    def __init__(
            self,
            block_channels: List[int],
            target_channels: int,
            in_channels: int = 3,
            downsample_factor: int = 8,
            norm: bool = False,
            cross_att_layer: List[bool] = (True, True, True),
            **kwargs
    ):
        super().__init__()
        block_channels.append(target_channels)
        assert len(block_channels) == len(cross_att_layer)
        rbs_x1 = []
        rbs_x2 = []
        cas = []
        for idx, ch in enumerate(block_channels):
            rbs_x1.append(
                ResidualBlockWithStride(
                    in_ch=in_channels,
                    out_ch=ch,
                    activation=nn.ReLU,
                    stride=2 if downsample_factor >= 2 else 1,
                )
            )
            rbs_x2.append(
                ResidualBlockWithStride(
                    in_ch=in_channels,
                    out_ch=ch,
                    activation=nn.ReLU,
                    stride=2 if downsample_factor >= 2 else 1,
                )
            )
            cas.append(
                CrossAttentionBlockCheng(
                    N=ch,
                    blocks_after_att=1 if idx == len(cross_att_layer) - 1 else None,
                )
                if cross_att_layer[idx]
                else Identity_()
            )

            in_channels = ch
            downsample_factor //= 2
        # rbs_x.append(
        #     ResidualBlockWithStride(
        #         in_ch=in_channels,
        #         out_ch=target_channels,
        #         activation=nn.ReLU,
        #         stride=2 if downsample_factor >= 2 else 1,
        #     )
        # )

        self.rbs_x1 = nn.ModuleList(rbs_x1)
        self.rbs_x2 = nn.ModuleList(rbs_x2)
        self.cas = nn.ModuleList(cas)

    def forward(
            self, x1: Tensor, x2: Optional[Tensor] = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if x2 is None:
            x = self.forward_satt(x1)
            return x, x
        x1, x2 = self.forward_catt(x1, x2)
        return x1, x2

    def forward_catt(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        for idx in range(len(self.rbs_x2)):
            x1 = self.rbs_x1[idx](x1)
            x2 = self.rbs_x2[idx](x2)
            x1 = self.cas[idx]((x1, x2))
        # x = self.rbs_x[-1](x)
        return x1, x2

    def forward_satt(self, x1: Tensor) -> Tensor:
        for idx in range(len(self.rbs_x2)):
            x1 = self.rbs_x1[idx](x1)
            x1 = self.cas[idx](x1)
        return x1
