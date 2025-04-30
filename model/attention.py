from functools import partial
from inspect import isfunction
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, einsum, nn
from torch.nn import functional as F
from einops import rearrange, repeat
from misc.util import normalization, zero_module
from model.layers import (
    ConvBlock,
    QKVAttention,
    conv1x1,
    conv3d,
    conv1x1x1,
    conv3x3,
    conv_nd,
)
from torchdistill.models.registry import get_model, register_model_class
from torch.utils.checkpoint import checkpoint


def exists(val) -> bool:
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


@register_model_class
class AttentionBlock3D(nn.Module):
    """"""

    def __init__(self, channels: int, seq_len: int, use_checkpoint: bool = False, ptemp_w: Optional[int] = None):
        super().__init__()
        self.seq_len = seq_len
        self.use_checkpoint = use_checkpoint

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1x1(channels, channels // 2),
                    nn.ReLU(inplace=True),
                    conv3d(
                        kernel_size=(ptemp_w or seq_len, 3, 3),
                        stride=(1, 1, 1),
                        in_ch=channels // 2,
                        out_ch=channels // 2,
                    ),
                    nn.ReLU(inplace=True),
                    conv1x1x1(channels // 2, channels),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return (
                    checkpoint(self._forward, x) if use_checkpoint else self._forward(x)
                )

            def _forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1x1(channels, channels),
        )

    def forward(self, x):
        return checkpoint(self._forward, x) if self.use_checkpoint else self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        B_D, C, H, W = x.shape
        B = B_D // self.seq_len
        x = x.reshape(B, self.seq_len, C, H, W).transpose(1, 2)
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        out = out.transpose(1, 2).reshape(B * self.seq_len, C, H, W)
        return out


class _CNNSelfAttention(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.queries = conv1x1(in_ch=in_ch, out_ch=out_ch)
        self.keys = conv1x1(in_ch=in_ch, out_ch=out_ch)
        self.values = conv1x1(in_ch=in_ch, out_ch=out_ch)

    def forward(self, x):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # Reshape queries and keys to have a separate dimension for the dot product
        batch, channels, height, width = queries.size()
        queries = queries.view(batch, channels, -1)
        queries = queries.permute(0, 2, 1)  # [batch, height * width, channels]
        keys = keys.view(batch, channels, -1)

        # Calculate attention map (height * width by height * width)
        attention_map = torch.bmm(
            queries, keys
        )  # [batch, height * width, height * width]
        attention_map = F.softmax(attention_map, dim=-1)

        # Reshape values for the weighted sum
        values = values.view(batch, channels, -1)
        values = values.permute(0, 2, 1)  # [batch, height * width, channels]

        # Apply the attention map to the values
        out = torch.bmm(attention_map, values)  # [batch, height * width, channels]
        out = out.permute(0, 2, 1).view(
            batch, channels, height, width
        )  # [batch, channels, height, width]

        return out


@register_model_class
class CNNSelfAttention(_CNNSelfAttention):
    def __init__(self, in_ch: int, out_ch: int, use_checkpoint: bool = False):
        super().__init__(in_ch, out_ch)

        self.fwd = (
            partial(checkpoint, super().forward, use_reentrant=True)
            if use_checkpoint
            else super().forward
        )

    def forward(self, x):
        return self.fwd(x)


class _CNNCrossAttention(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.queries = conv1x1(in_ch=in_ch, out_ch=out_ch)
        self.keys = conv1x1(in_ch=in_ch, out_ch=out_ch)
        self.values = conv1x1(in_ch=in_ch, out_ch=out_ch)

    def forward(self, x1, x2):
        queries = self.queries(x1)
        keys = self.keys(x2)
        values = self.values(x2)

        # Reshape queries and keys to have a separate dimension for the dot product
        batch, channels, height, width = queries.size()
        queries = queries.view(batch, channels, -1)
        queries = queries.permute(0, 2, 1)  # [batch, height * width, channels]
        keys = keys.view(batch, channels, -1)

        # Calculate attention map (height * width by height * width)
        attention_map = torch.bmm(
            queries, keys
        )  # [batch, height * width, height * width]
        attention_map = F.softmax(attention_map, dim=-1)

        # Reshape values for the weighted sum
        values = values.view(batch, channels, -1)
        values = values.permute(0, 2, 1)  # [batch, height * width, channels]

        # Apply the attention map to the values
        out = torch.bmm(attention_map, values)  # [batch, height * width, channels]
        out = out.permute(0, 2, 1).view(
            batch, channels, height, width
        )  # [batch, channels, height, width]

        return out


@register_model_class
class CNNCrossAttention(_CNNCrossAttention):
    class _identity2(nn.Identity):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return x1, x2

    def __init__(self, in_ch: int, out_ch: int, use_checkpoint: bool = False):
        super().__init__(in_ch, out_ch)

        self.fwd = (
            partial(checkpoint, super().forward, use_reentrant=True)
            if use_checkpoint
            else super().forward
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.fwd(x1, x2)


@register_model_class
class CrossAttentionBlockCheng(nn.Module):
    def __init__(self, N: int, blocks_after_att: Optional[int] = None, *args, **kwargs):
        super().__init__()
        """Cross attention block.
        Based on Cheng et al.'s AttentionBlock CompressAI implementation
        """

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=False),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=False),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=False)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

        self.conv_after = (
            nn.Sequential(*[ResidualUnit() for _ in range(blocks_after_att)])
            if blocks_after_att
            else nn.Identity()
        )

    def forward(self, x1: Tensor | Tuple[Tensor], x2: Tensor = None) -> Tensor:
        if x2 is None:
            if isinstance(x1, tuple):
                x1, x2 = x1
            else:
                x2 = x1
        identity = x1
        a = self.conv_a(x1)
        b = self.conv_b(x2)
        out = a * torch.sigmoid(b)
        out = self.conv_after(out)

        out_with_identity = out + identity
        return out_with_identity


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return (
            checkpoint(self._forward, x, context)
            if self.checkpoint
            else self._forward(x)
        )

        # return checkpoint(
        #     self._forward, (x, context), self.parameters(), self.checkpoint
        # )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


@register_model_class
class SpatialTransformer(nn.Module):
    """Ported from https://github.com/CompVis/latent-diffusion

    Transformer block for image-like data.
    First, project the input (aka embedding) and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape back to image

    Note: Seems to use depth=1 universally in all configurations
    """

    def __init__(
            self,
            in_channels,
            num_heads,
            depth=1,
            dropout=0.0,
            context_dim=None,
            use_checkpoint=True,
            conv_after=False,
    ):
        super().__init__()
        d_head = in_channels // num_heads
        self.in_channels = in_channels
        inner_dim = (
                num_heads * d_head
        )  # seems like in_channels == inner_dim for all configs in LD
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    checkpoint=use_checkpoint,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.conv_after = (
            ConvBlock(in_ch=in_channels, out_ch=in_channels, kernel_size=3)
            if conv_after
            else nn.Identity()
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        x = self.conv_after(x)
        return x + x_in


@register_model_class
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        # 1x1 1d convolutions
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)

        # <=> x.mean (dim=2) for 3D tensor => 1, C, 1 => the average value of the NW dimensional vector for each c in C
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # <=> .unsqueeze(dim=0)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        # Selecting feature vector corresponding to the averaged feature representation
        return x[:, :, 0]


@register_model_class
class AttentionBlockHo(nn.Module):
    """
    Ported from

    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            conv_after: bool = False,
            num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        # channels * 3, to evenly split between  q k and v
        self.norm = normalization(channels, num_groups=num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.conv_after = (
            ConvBlock(in_ch=channels, out_ch=channels, kernel_size=3)
            if conv_after
            else nn.Identity()
        )

    def forward(self, x, *args, **kwargs):
        return (
            checkpoint(self._forward, x, *args, **kwargs)
            if self.use_checkpoint
            else self._forward(x, *args, **kwargs)
        )

    def _forward(self, x, *args, **kwargs):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        res = (x + h).reshape(b, c, *spatial)
        res = self.conv_after(res)
        return res


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module: nn.ModuleList):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None, *args, **kwargs):
        assert dummy_arg is not None
        for m in self.module:
            x = m(x, *args, **kwargs)
        return x


@register_model_class
class AttentionLayer(nn.Module):
    def __init__(
            self,
            attention_block_config: Dict[str, Any],
            no_blocks: int = 1,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        self.att_blocks = nn.ModuleList(
            [
                get_model(
                    attention_block_config["type"], **attention_block_config["params"], use_checkpoint=use_checkpoint
                )
                for _ in range(no_blocks)
            ]
        )
        # self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.use_checkpoint = use_checkpoint
        # self.wrapper = (
        #     ModuleWrapperIgnores2ndArg(self.att_blocks) if use_checkpoint else None
        # )

    def forward(self, x, *args, **kwargs):
        return (
            checkpoint(self._forward, x, *args, **kwargs, use_reentrant=False)
            if self.use_checkpoint
            else self._forward(x, *args, **kwargs)
        )

    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for m in self.att_blocks:
            x = m(x, *args, **kwargs)
        return x
