from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from model.ntc.context.models import ContextModel
from torchdistill.common.constant import def_logger

from model.ntc.image.hypernetwork import HyperNetwork
from model.registry import (
    get_analysis_network,
    get_compressai_model,
    get_compression_module,
    get_context_model,
    get_hyper_network,
    get_synthesis_network,
    register_compression_module,
)
from torchdistill.common.module_util import freeze_module_params
from torchdistill.models.registry import (
    get_model,
    register_model_class,
    register_model_func,
)

logger = def_logger.getChild(__name__)


# Reminder: Removed analyzers, since I'm just using CompressAI methods
class ModularImageCompressionModel(CompressionModel):
    """
    Variational Image Compression Entropy Models with exchangable networks
    """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
    ):
        super().__init__()
        self.entropy_bottleneck_channels = entropy_bottleneck_channels
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.g_a = get_analysis_network(
            analysis_config["name"], **analysis_config["params"]
        )
        self.g_s = get_synthesis_network(
            synthesis_config["name"], **synthesis_config["params"]
        )
        self.updated = False

    def forward(
            self, x: Tensor, return_likelihoods: bool
    ) -> Union[Tensor, Mapping[str, Mapping[str, Tensor]]]:
        return NotImplementedError

    def forward_train(
            self, x: Tensor, return_likelihoods: bool
    ) -> Union[Tensor, Mapping[str, Mapping[str, Tensor]]]:
        return NotImplementedError

    # Too much of a hasle to Union every possible subclass output
    def compress(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def decompress(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def update(self, scale_table=None, force=False) -> bool:
        logger.info("Updating Bottleneck..")
        updated = super().update(scale_table=scale_table, force=force)
        self.updated = True
        return updated

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        raise NotImplementedError

    def get_decoder_modules(self) -> Iterable[nn.Module]:
        raise NotImplementedError


@register_compression_module
class FactorizedPrior(ModularImageCompressionModel):
    """
    Factorized Prior from compressAI with exchangable transforms by Ballé et al. https://arxiv.org/abs/1802.01436>
    """

    def __init__(
            self,
            entropy_bottleneck_channels,
            analysis_config,
            synthesis_config=None,
            **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels, analysis_config, synthesis_config)

    def get_means(self, x) -> Tensor:
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward(
            self, x, return_likelihoods=False
    ) -> Union[Tensor, Mapping[str, Mapping[str, Tensor]]]:
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}
        else:
            return x_hat

    def compress(self, x: Tensor, *args, **kwargs) -> Tuple[Iterable[str], torch.Size]:
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(
            self, strings: List[str], shape: torch.Size, *args, **kwargs
    ) -> Tensor:
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        return [self.g_a, self.entropy_bottleneck]

    def get_decoder_modules(self) -> Iterable[nn.Module]:
        return [self.g_s]


@register_compression_module
class FactorizedPriorWithFeatureExtractor(ModularImageCompressionModel):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels,
            feature_extractor_config: dict[str, any],
            analysis_config,
            synthesis_config=None,
            **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels, analysis_config, synthesis_config)

        feature_extractor = get_model(
            feature_extractor_config["name"], **feature_extractor_config["params"]
        )
        freeze_module_params(feature_extractor)
        self.feature_extractor = feature_extractor

    def forward(
            self, x, return_likelihoods=False
    ) -> Union[Tensor, Mapping[str, Mapping[str, Tensor]]]:
        self.feature_extractor.eval()
        h = self.feature_extractor(x)
        y = self.g_a(h)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}
        else:
            return x_hat

    def compress(self, x: Tensor, *args, **kwargs) -> Tuple[Iterable[str], torch.Size]:
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        return [self.g_a, self.entropy_bottleneck]

    def get_decoder_modules(self) -> Iterable[nn.Module]:
        return [self.g_s]


@register_compression_module
class ScaleHyperprior(ModularImageCompressionModel):
    r"""

    Scale Hyperprior from compressAI with exchangable transforms by Ballé et al. https://arxiv.org/abs/1802.01436>

    """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Optional[Mapping[str, Any]] = None,
            hyper_network_config: Mapping[str, Any] = None,
            **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels, analysis_config, synthesis_config)
        self.gaussian_params_channels = gaussian_params_channels
        self.hyper_network: HyperNetwork = get_hyper_network(
            hyper_network_config["name"], **hyper_network_config["params"]
        )
        self.gaussian_conditional = GaussianConditional(None)

    @property
    def h_a(self) -> nn.Module:
        return self.hyper_network.hyper_analysis

    @property
    def h_s(self) -> nn.Module:
        return self.hyper_network.hyper_synthesis

    def get_means(self, x) -> Tensor:
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return medians

    def forward(self, x, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods, "z": z_likelihoods}
        else:
            return x_hat

    def compress(
            self, x
    ) -> Mapping[str, Union[Tuple[Iterable[str], Iterable[str]], torch.Size]]:
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_shape = z.size()[-2:]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": (y_strings, z_strings), "shape": z.size()[-2:]}

    def decompress(
            self, strings: Tuple[List[str], List[str]], shape: torch.Size
    ) -> Tensor:
        assert isinstance(strings, tuple) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat


@register_model_class
@register_compression_module
class MeanScaleHyperprior(ScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
        )

    def forward(self, x: Tensor, return_likelihoods: bool = False):
        y = self.g_a(x)
        z = self.h_a(y)
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


@register_compression_module
class RecursiveMeanScaleHyperprior(MeanScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            hyper_network_config2: Mapping[str, any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
            **kwargs,
        )
        self.hyper_network_2: HyperNetwork = get_hyper_network(
            hyper_network_config2["name"], **hyper_network_config2["params"]
        )
        self.gaussian_conditional_2 = GaussianConditional(None)

    def forward(self, x, return_likelihoods=False):
        y = self.g_a(x)  # oc: primary_2
        z_1 = self.h_a(y)  # ic: g_c -> primary_2 oc: e_b -> primary 2
        z_2 = self.h_a_2(z_1)  # ic: g_c -> primary_2 oc: e_b -> primary 2
        z_2_hat, z_2_likelihoods = self.entropy_bottleneck(z_2)
        gaussian_params_1 = self.h_s_2(z_2_hat)
        scales_hat_1, means_hat_1 = gaussian_params_1.chunk(2, 1)
        z_1_hat, z_1_likelihoods = self.gaussian_conditional(
            z_1, scales_hat_1, means=means_hat_1
        )
        gaussian_params_2 = self.h_s(z_1_hat)
        scales_hat_2, means_hat_2 = gaussian_params_2.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional_2(
            y, scales_hat_2, means=means_hat_2
        )
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return {
                "x_hat": x_hat,
                "likelihoods": {
                    "y": y_likelihoods,
                    "z_1": z_1_likelihoods,
                    "z_2": z_2_likelihoods,
                },
            }
        return x_hat

    def compress(
            self, x: Tensor
    ) -> Mapping[str, Union[Tuple[Iterable[str], Iterable[str]], torch.Size]]:
        y = self.g_a(x)
        z_1 = self.h_a(y)
        z_2 = self.h_a_2(z_1)

        z_2_strings = self.entropy_bottleneck.compress(z_2)
        z_2_hat = self.entropy_bottleneck.decompress(z_2_strings, z_2.size()[-2:])

        gaussian_params_1 = self.h_s_2(z_2_hat)
        scales_hat_1, means_hat_1 = gaussian_params_1.chunk(2, 1)
        indexes_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
        z_1_strings = self.gaussian_conditional.compress(
            z_1, indexes_1, means=means_hat_1
        )
        z_1_hat = self.gaussian_conditional.decompress(
            z_1_strings, indexes_1, means=means_hat_1
        )

        gaussian_params_2 = self.h_s(z_1_hat)
        scales_hat_2, means_hat_2 = gaussian_params_2.chunk(2, 1)
        indexes_2 = self.gaussian_conditional_2.build_indexes(scales_hat_2)
        y_strings = self.gaussian_conditional_2.compress(
            y, indexes_2, means=means_hat_2
        )
        return {
            "strings": [y_strings, z_1_strings, z_2_strings],
            "shape": z_1.size()[-2:],
        }

    def decompress(
            self, strings: Tuple[List[str], List[str], List[str]], shape: torch.Size
    ) -> Tensor:
        assert isinstance(strings, list) and len(strings) == 3
        z_2_hat = self.entropy_bottleneck.decompress(strings[2], shape)

        gaussian_params_2 = self.h_s(z_2_hat)
        scales_hat_2, means_hat_2 = gaussian_params_2.chunk(2, 1)
        indexes_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
        z_1_hat = self.gaussian_conditional.decompress(
            strings[1], indexes_2, means=means_hat_2
        )

        gaussian_params_1 = self.h_s(z_1_hat)
        scales_hat_1, means_hat_1 = gaussian_params_1.chunk(2, 1)
        indexes_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes_1, means=means_hat_1
        )

        x_hat = self.g_s(y_hat)
        return x_hat

    @property
    def h_a_2(self) -> nn.Module:
        return self.hyper_network_2.hyper_analysis

    @property
    def h_s_2(self) -> nn.Module:
        return self.hyper_network_2.hyper_synthesis


@register_compression_module
class MSHPWithSideInfoTileSharedWeights(MeanScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
            **kwargs,
        )

    def forward_train(self, x: Tensor, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(y)
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

    def forward(self, x: List[Tensor] | Tensor, return_likelihoods=False):
        if isinstance(x, list):
            x = torch.stack(x, dim=2)
            B, C, D, H, W = x.shape
            x = x.reshape(B * D, C, H, W)  # Attention: This seems very wrong
        return super().forward(x, return_likelihoods)

    def compress(
            self, x: Tensor
    ) -> Mapping[str, Union[Tuple[Iterable[str], Iterable[str]], torch.Size]]:
        if isinstance(x, list):
            x = torch.stack(x, dim=2)
            B, C, D, H, W = x.shape
            x = x.reshape(B * D, C, H, W)
        return super().compress(x)

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


@register_compression_module
class MeanScaleHyperpriorWithFeatureExtractor(ScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            feature_extractor_config: Dict[str, Any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
        )

        feature_extractor = get_model(
            feature_extractor_config["name"], **feature_extractor_config["params"]
        )
        freeze_module_params(feature_extractor)
        self.feature_extractor = feature_extractor

    def forward(self, x: Tensor, return_likelihoods=False):
        self.feature_extractor.eval()
        h = self.feature_extractor(x)
        y = self.g_a(h)
        z = self.h_a(y)
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
        h = self.feature_extractor(x)
        y = self.g_a(h)
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


@register_compression_module
class MeanScaleHyperprior(ScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
        )

    def forward(self, x: Tensor, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(y)
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


@register_compression_module
class MSHPWithSideInfoSharedWeightsAndGlobalFeatures(MeanScaleHyperprior):
    """ """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
            **kwargs,
        )

    def forward_train(self, x: Tensor, return_likelihoods=False):
        y = self.g_a(x)
        z = self.h_a(y)
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

    def forward(self, x: List[Tensor] | Tensor, return_likelihoods=False):
        if isinstance(x, list):
            x = torch.stack(x, dim=2)
            B, C, D, H, W = x.shape
            x = x.reshape(B * D, C, H, W)
        y = self.g_a(x)
        z = self.h_a(y)
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
        if isinstance(x, list):
            x = torch.stack(x, dim=2)
            B, C, D, H, W = x.shape
            x = x.reshape(B * D, C, H, W)
        else:
            B, C, H, W = x.shape
            D = 1

        return super().compress(x)

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


@register_compression_module
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    """
    Joint Autoregressive Hierarchical Priors model from compressAI with exchangable transforms and context model
        https://arxiv.org/abs/1809.02736
    """

    def __init__(
            self,
            entropy_bottleneck_channels: int,
            gaussian_params_channels: int,
            analysis_config: Mapping[str, Any],
            synthesis_config: Mapping[str, Any],
            hyper_network_config: Mapping[str, any],
            context_model_config: Mapping[str, Any],
            **kwargs,
    ):
        super().__init__(
            entropy_bottleneck_channels,
            gaussian_params_channels,
            analysis_config,
            synthesis_config,
            hyper_network_config,
        )
        self.context_model: ContextModel = get_context_model(
            context_model_config["name"], **context_model_config["params"]
        )

    @property
    def entropy_parameters(self) -> nn.Module:
        return self.context_model.entropy_parameters

    @property
    def context_prediction(self) -> nn.Module:
        return self.context_model.context_prediction

    def forward(
            self, x: Tensor, return_likelihoods: bool = False
    ) -> Union[Tensor, Mapping[str, Mapping[str, Tensor]]]:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }
        return x_hat

    def compress(
            self, x
    ) -> Mapping[str, Union[Tuple[Iterable[str], Iterable[str]], torch.Size]]:
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        from compressai.ans import BufferedRansEncoder

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape) -> Tensor:
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (
                z_hat.size(0),
                self.gaussian_params_channels,
                y_height + 2 * padding,
                y_width + 2 * padding,
            ),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        from compressai.ans import RansDecoder

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv


@register_model_func
def get_custom_compressor(name: str, **kwargs: Any) -> ModularImageCompressionModel:
    compressor = get_compression_module(name, **kwargs)
    return compressor


@register_model_func
def get_compressai_compressor(name: str, **kwargs: Any) -> CompressionModel:
    compressor = get_compressai_model(name, **kwargs)
    return compressor
