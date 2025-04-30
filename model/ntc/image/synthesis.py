from functools import partial
from typing import List, Optional, Tuple

import torch
from compressai.layers import AttentionBlock, GDN
from torch import Tensor, nn
from ultralytics.nn.modules import C2f, C3

from model.attention import AttentionBlock3D, CNNSelfAttention, CrossAttentionBlockCheng
from model.layers import (
    ConvBlock,
    ConvBlock3D,
    Identity_,
    ResidualBlockWithStride,
    conv,
)
from model.ntc.base import BaseNTC
from model.registry import register_synthesis_network



@register_synthesis_network
class FPBaselineSynthesis(BaseNTC):
    """

    128, 192 for quality level 1-4, 192, 320 for quality levels 5-8
    """

    def __init__(
            self,
            out_channels: int = 3,
            latent_channels: int = 192,
            network_channels: int = 128,
            use_relu: bool = True,
            upsample_factor: int = 8,
    ):
        super().__init__()
        upsample_factor *= 2
        self.layers = nn.Sequential(
            conv(
                kernel_size=5,
                in_ch=latent_channels,
                out_ch=network_channels,
                **{
                    "stride": 2,
                    "upsample": True,
                }
                if (upsample_factor := upsample_factor // 2) > 1
                else {
                    "stride": 1,
                    "upsample": False,
                },
            ),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels, inverse=True),
            conv(
                kernel_size=5,
                in_ch=network_channels,
                out_ch=network_channels,
                **{
                    "stride": 2,
                    "upsample": True,
                }
                if (upsample_factor := upsample_factor // 2) > 1
                else {
                    "stride": 1,
                    "upsample": False,
                },
            ),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels, inverse=True),
            conv(
                kernel_size=5,
                in_ch=network_channels,
                out_ch=network_channels,
                **{
                    "stride": 2,
                    "upsample": True,
                }
                if (upsample_factor := upsample_factor // 2) > 1
                else {
                    "stride": 1,
                    "upsample": False,
                },
            ),
            nn.ReLU(inplace=False) if use_relu else GDN(network_channels, inverse=True),
            conv(
                kernel_size=5,
                in_ch=network_channels,
                out_ch=out_channels,
                **{
                    "stride": 2,
                    "upsample": True,
                }
                if upsample_factor > 1
                else {
                    "stride": 1,
                    "upsample": False,
                },
            ),
        )

    def forward(self, x):
        z = self.layers(x)
        return z


@register_synthesis_network
class GenericResidualSynthesisNetwork(BaseNTC):
    def __init__(
            self,
            channels: List[int],
            kernel_size: int,
            norm: bool = False,
            upsample_factor=1,
    ):
        super(GenericResidualSynthesisNetwork, self).__init__()
        in_channels, channels = channels[0], channels[1:]
        layers = []
        for out_channels in channels:
            upsample = upsample_factor >= 2
            layers.append(
                ResidualBlockWithStride(
                    in_ch=in_channels,
                    out_ch=out_channels,
                    kernel_size=kernel_size,
                    stride=2 if upsample else 1,
                    upsample=upsample,
                    activation=nn.LeakyReLU,
                    norm=nn.BatchNorm2d if norm else None,
                )
            )
            in_channels = out_channels
            upsample_factor /= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@register_synthesis_network
class YOLOTransformationSynthesis(BaseNTC):
    def __init__(
            self,
            yolo_model: str,
            channels: List[int],
            kernel_size: int = 3,
            upsample_factor: int = 1,
            norm: bool = False,
            norm_last: bool = True,
    ):
        super().__init__()
        yolo_layer = C2f if "v8" in yolo_model else C3
        in_channels, yolo_block_channels, target_channels = (
            channels[0],
            channels[1:-1],
            channels[-1],
        )
        layers = []

        for out_channels in yolo_block_channels:
            upsample = upsample_factor >= 2
            layers.append(
                nn.Sequential(
                    ResidualBlockWithStride(
                        in_ch=in_channels,
                        out_ch=out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        upsample=True,
                        activation=partial(nn.SiLU, inplace=False),
                        norm=partial(
                            nn.BatchNorm2d, eps=0.001, momentum=0.03, affine=True
                        )
                        if norm
                        else None,
                    ),
                    yolo_layer(out_channels, out_channels),
                )
                if upsample
                else yolo_layer(in_channels, out_channels, shortcut=True),
            )
            in_channels = out_channels
            upsample_factor /= 2
        upsample = upsample_factor >= 2
        layers.append(
            ConvBlock(
                in_ch=in_channels,
                out_ch=target_channels,
                kernel_size=kernel_size,
                stride=2 if upsample else 1,
                upsample=upsample,
                activation=partial(nn.SiLU, inplace=False),
                norm_layer=partial(
                    nn.BatchNorm2d, eps=0.001, momentum=0.03, affine=True
                )
                if norm_last
                else None,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@register_synthesis_network
class SequencedSynthesisWith3DAttention(BaseNTC):
    def __init__(
            self,
            seq_len: int,
            channels: List[int],
            use_attn: tuple[bool] = (True, True, True),
            in_channels: int = 3,
            upsample_factor: int = 1,
            norm: bool = False,
            ptemp_w: Optional[int] = None,
            **kwargs,
    ):
        assert len(channels) - 1 == len(use_attn)
        super().__init__()
        self.seq_len = seq_len
        in_channels, channels = channels[0], channels[1:]
        blocks = []
        for idx, ch in enumerate(channels):
            upsample = upsample_factor >= 2
            blocks.append(
                nn.Sequential(
                    ResidualBlockWithStride(
                        in_ch=in_channels,
                        out_ch=ch,
                        activation=nn.ReLU,
                        stride=2 if upsample else 1,
                        upsample=upsample,
                    ),
                    AttentionBlock3D(channels=ch, seq_len=self.seq_len, ptemp_w=ptemp_w)
                    if use_attn[idx]
                    else nn.Identity(),
                )
            )
            in_channels = ch
            upsample_factor //= 2
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        B_D, C, H, W = x.shape
        B = B_D // self.seq_len
        x = x.reshape(B, self.seq_len, C, H, W)
        x = [t.squeeze(dim=1) for t in x.split(1, dim=1)]
        return x
