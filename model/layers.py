import math
from typing import Optional, Union

import torch
from torch import Tensor, nn

from misc.util import to_3tuple
from torchdistill.models.registry import register_model_class


def conv_nd(dims, *args, **kwargs):
    """
    Lazily just taken from https://github.com/CompVis/latent-diffusion instead of refactoring to use my own blocks
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


def conv3d(
    kernel_size: tuple[int] | int,
    in_ch: int,
    out_ch: int,
    stride: tuple[int] | int = 1,
    bias: bool = True,
    upsample: bool = False,
    **kwargs,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    kernel_size = to_3tuple(kernel_size)
    stride = to_3tuple(stride)
    return (
        nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=(k // 2 for k in kernel_size),
            bias=bias,
            output_padding=(s - 1 for s in stride),
            **kwargs,
        )
        if upsample
        else nn.Conv3d(
            in_ch,
            out_ch,
            padding=(k // 2 for k in kernel_size),
            kernel_size=kernel_size,
            stride=stride,
            **kwargs,
        )
    )


def conv(
    kernel_size: int,
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    bias: bool = True,
    upsample: bool = False,
    **kwargs,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    return (
        nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            output_padding=stride - 1,
            **kwargs,
        )
        if upsample
        else nn.Conv2d(
            in_ch,
            out_ch,
            padding=kernel_size // 2,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs,
        )
    )


def conv5x5(
    in_ch: int, out_ch: int, stride: int = 1, bias: bool = True, upsample: bool = False
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    return conv(
        kernel_size=5,
        in_ch=in_ch,
        out_ch=out_ch,
        stride=stride,
        bias=bias,
        upsample=upsample,
    )


def conv3x3(
    in_ch: int, out_ch: int, stride: int = 1, bias: bool = True, upsample: bool = False
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    return conv(3, in_ch, out_ch, stride, bias, upsample)


def conv1x1(
    in_ch: int, out_ch: int, stride: int = 1, bias: bool = True, upsample: bool = False
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    return conv(1, in_ch, out_ch, stride, bias, upsample)


def conv1x1x1(
    in_ch: int,
    out_ch: int,
    stride: tuple[int] = (1, 1, 1),
    bias: bool = True,
    upsample: bool = False,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    return conv3d((1, 1, 1), in_ch, out_ch, stride, bias, upsample)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        upsample=False,
        norm_layer=None,
        activation=nn.ReLU,
        **kwargs,
    ):
        super(ConvBlock, self).__init__()
        self.conv = conv(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            bias=norm_layer is None,
            upsample=upsample,
            **kwargs,
        )
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.act = activation()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Identity_(nn.Identity):
    def forward(self, x):
        if isinstance(x, tuple):
            return x[0]
        return x


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int] | int,
        stride: tuple[int] = (1, 1, 1),
        upsample=False,
        norm_layer=None,
        activation=nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.conv = conv3d(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            bias=norm_layer is None,
            upsample=upsample,
            **kwargs,
        )
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.act = activation()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlockWithStride(nn.Module):
    """
    Residual block that stacks two 3x3 convos
    The first convolution up or downsamples according to stride
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        out_ch2: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 2,
        norm: Optional[nn.Module] = None,
        activation: nn.Module = nn.LeakyReLU,
        upsample: bool = False,
    ):
        super().__init__()
        self.norm = nn.Identity() if norm is None else norm(num_features=out_ch)
        self.act = activation(inplace=False)
        out_ch2 = out_ch2 or out_ch
        self.conv1 = ConvBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            upsample=upsample,
        )
        self.conv2 = ConvBlock(
            in_ch=out_ch,
            out_ch=out_ch2,
            kernel_size=kernel_size,
            upsample=False,
            stride=1,
        )
        self.skip = (
            conv1x1(in_ch, out_ch2, stride=stride, upsample=upsample)
            if in_ch != out_ch2 or stride != 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act(self.conv1(x))
        out = self.act(self.norm(self.conv2(out)))
        out = out + self.skip(identity)
        return out


class ResidualBlockWithStride3D(nn.Module):
    """ """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        out_ch2: Optional[int] = None,
        kernel_size: Optional[int] = 3,
        stride: int = 2,
        norm: Optional[nn.Module] = None,
        activation: nn.Module = nn.LeakyReLU,
        upsample: bool = False,
    ):
        super().__init__()
        self.norm = nn.Identity() if norm is None else norm(num_features=out_ch)
        self.act = activation(inplace=False)
        out_ch2 = out_ch2 or out_ch
        self.conv1 = ConvBlock3D(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            upsample=upsample,
        )
        self.conv2 = ConvBlock3D(
            in_ch=out_ch,
            out_ch=out_ch2,
            kernel_size=kernel_size,
            upsample=False,
            stride=1,
        )
        self.skip = (
            conv1x1x1(in_ch, out_ch2, stride=stride, upsample=upsample)
            if in_ch != out_ch2 or stride != 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act(self.conv1(x))
        out = self.act(self.norm(self.conv2(out)))
        out = out + self.skip(identity)
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


@register_model_class
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Identity,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
