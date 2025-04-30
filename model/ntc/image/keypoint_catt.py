from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import kornia as K
import torch
from compressai.entropy_models import EntropyBottleneck
from kornia import feature as KF
from torch import Tensor, nn

from misc.util import top_n_to_one
from model.attention import AttentionLayer, CrossAttentionBlockCheng
from model.layers import ConvBlock, ResidualBlockWithStride
from model.ntc.image.image_base import ModularImageCompressionModel, ScaleHyperprior
from model.registry import register_compression_module
from torchdistill.common.module_util import freeze_module_params
from torchdistill.models.registry import get_model, register_model_class


@register_model_class
@register_compression_module
class SideinfoWithKPCATT(ScaleHyperprior):
    """
    """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            # feature_extractor_config: dict[str, any],
            analysis_config: dict[str, Any],
            synthesis_config: dict[str, Any],
            hyper_network_config: dict[str, any],
            attention_config: Mapping[str, Any],
            num_keypoints: int = 16384,
            recon_ca_latent: bool = False,
            detach_input_from_kps: bool = True,
            concat_kp_scores_input: bool = False,
            latent_channels: int = 96,
            small_ds: bool = False,
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
        )

        # feature_extractor = get_model(
        #     feature_extractor_config["name"], **feature_extractor_config["params"]
        # )
        # self.feature_extractor = feature_extractor
        self.num_keynet_features = num_keypoints
        self.kp_detector = KF.KeyNet(pretrained=True)
        freeze_module_params(self.kp_detector)
        self.ca = AttentionLayer(**attention_config)
        self.recon_ca_latent = recon_ca_latent
        self.ds = (
            nn.Sequential(
                ConvBlock(in_ch=3, out_ch=48, kernel_size=3, stride=2),
                ConvBlock(in_ch=48, out_ch=64, kernel_size=3, stride=2),
                ConvBlock(in_ch=64, out_ch=latent_channels, kernel_size=3, stride=2),
            )
            if small_ds
            else nn.Sequential(
                ResidualBlockWithStride(in_ch=3, out_ch=48, stride=2),
                ResidualBlockWithStride(
                    in_ch=48,
                    out_ch=64,
                    stride=2,
                ),
                ResidualBlockWithStride(
                    in_ch=64,
                    out_ch=latent_channels,
                    stride=2,
                ),
            )
        )
        self.detach_input_from_kps = detach_input_from_kps
        self.concat_kp_scores_input = concat_kp_scores_input

    def forward(self, x: Tensor | List[Tensor], return_likelihoods=False):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
            B, D, C, H, W = x.shape
            x = x.reshape(B * D, C, H, W)
        self.kp_detector.eval()
        keynet_scores = self.kp_detector(K.color.rgb_to_grayscale(x))
        keypoints = top_n_to_one(keynet_scores, self.num_keynet_features)
        keypixels = keypoints * (x.detach() if self.detach_input_from_kps else x)
        if self.concat_kp_scores_input:
            x = torch.cat([x, keypixels.mean(dim=1, keepdims=True)], dim=1)
        y = self.g_a(x)
        y_kps = self.ds(keypixels)
        y_ca = self.ca(y, y_kps)
        z = self.h_a(y_ca)
        y = y_ca if self.recon_ca_latent else y  # whether to use kps only as context for side-info
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }
        else:
            return x_hat

    def compress(
            self, x: Tensor
    ) -> Mapping[str, Union[Tuple[Iterable[str], Iterable[str]], torch.Size]]:
        y = self.g_a(x)
        keynet_scores = self.kp_detector(K.color.rgb_to_grayscale(x))
        keynet_scores = keynet_scores * x.detach()
        y_kn = self.ds(keynet_scores)
        y_kn = top_n_to_one(y_kn, self.num_keynet_features)
        y_ca = self.ca(y, y_kn)
        z = self.h_a(y_ca)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(
            self, strings: Tuple[List[str], List[str]], shape: torch.Size
    ) -> Tensor:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)
        return x_hat
