from typing import Any, Dict, Optional

import torch
from PIL.Image import Image
from compressai.models import CompressionModel
from torch import nn
from torch import Tensor
from collections import OrderedDict

from torchvision import transforms

# from model.gan import get_gan_model
from model.registry import (
    get_compressai_model,
    get_compression_module,
    get_bottleneck_wrapper,
)
from torchdistill.common.constant import def_logger
from torchdistill.datasets.util import build_transform
from torchdistill.models.registry import (
    get_model,
    register_model_class,
    register_model_func,
)

from misc.analyzers import AnalyzableModule
from torchdistill.rate_analyzer import RateAnalyzer

logger = def_logger.getChild(__name__)


@register_model_class
class NetworkWithPerceptualRecovery(nn.Module):
    def __init__(self, stoch_dec_config: Dict[str, Any], backbone_config: Dict[str, Any]):
        super().__init__()
        self.stoch_dec = get_stoch_decoder(**stoch_dec_config)
        self.backbone = get_model(
            model_name=backbone_config["name"], **backbone_config["params"]
        )

    def load_state_dict(self, state_dict: Dict[str, Any], **kwargs):
        if "gan" in state_dict:
            logger.info("Loading gan")
            self.load_compressor_state_dict(state_dict.get("compressor"))
        if "backbone" in state_dict:
            logger.info("Loading backbone")
            self.load_backbone_state_dict(state_dict.get("backbone"))

    def load_compressor_state_dict(self, state_dict: Any, **kwargs):
        self.stoch_dec.load_state_dict(state_dict)
        super().load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.backbone(x)
        recon = self.stoch_dec.generator(z)
        return recon

    @property
    def G(self):
        return self.stoch_dec.generator

    @property
    def D(self):
        return self.stoch_dec.discriminator


class NetworkWithCompressionModule(AnalyzableModule):
    """ """

    def __init__(
        self,
        compression_module: CompressionModel,
        backbone: nn.Module,
        analyzers_config: Dict[str, Any] = None,
    ):
        if analyzers_config is None:
            analyzers_config = dict()
        super(NetworkWithCompressionModule, self).__init__(analyzers_config)
        self.compression_module = compression_module
        self.backbone = backbone
        self.analyze_after_compress = analyzers_config.get(
            "analyze_after_compress", False
        )
        self.compressor_updated = False

    def update(self, force=False) -> bool:
        updated = self.compression_module.update(force=force)
        self.compressor_updated = updated
        self.compressor_updated = True
        return updated

    def compress(self, obj):
        return self.compression_module.compress(obj)

    def decompress(self, compressed_obj):
        return self.compression_module.decompress(compressed_obj)

    def aux_loss(self) -> Tensor:
        return self.compression_module.aux_loss()

    def load_state_dict(self, state_dict, **kwargs):
        compression_module_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith("compression_module."):
                compression_module_state_dict[
                    key.replace("compression_module.", "", 1)
                ] = state_dict[key]

        self.compression_module.load_state_dict(compression_module_state_dict)
        super().load_state_dict(state_dict, strict=False)

    def load_state_dict(
        self, state_dict: Dict[str, Any], include_predictors: bool = False, **kwargs
    ):
        if include_predictors and "predictors" in state_dict:
            logger.info("Loading predictors")
            self.load_predictors_state_dict(state_dict.get("predictors"))
        if "compressor" in state_dict:
            logger.info("Loading compressor")
            self.load_compressor_state_dict(state_dict.get("compressor"))
        if "backbone" in state_dict:
            logger.info("Loading backbone")
            self.load_backbone_state_dict(state_dict.get("backbone"))

    def load_compressor_state_dict(self, state_dict: Any, strict=False, **kwargs):
        # compression_module_state_dict = OrderedDict()
        # for key in list(state_dict.keys()):
        #     if key.startswith("compression_module."):
        #         compression_module_state_dict[key.replace("compression_module.", '', 1)] = state_dict[key]

        self.compression_module.load_state_dict(state_dict, strict=strict)
        super().load_state_dict(state_dict, strict=False)

    def load_backbone_state_dict(self, state_dict: Any, **kwargs):
        self.backbone.load_state_dict(state_dict)

    @property
    def inference_mode(self) -> bool:
        return self.compression_module.inference_mode

    @inference_mode.setter
    def inference_mode(self, value: bool):
        self.compression_module.inference_mode = value


@register_model_class
class NetworkWithFeatureReconstruction(NetworkWithCompressionModule):
    def __init__(
        self,
        compressor: CompressionModel,
        backbone: nn.Module,
        analyzers_config: Dict[str, Any] = None,
    ):
        super().__init__(compressor, backbone, analyzers_config)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        recon = self.compression_module(features)
        return recon

    def update(self, force=False) -> bool:
        # We only care about syynthesis transform
        return True


@register_model_class
class DiscriminativeModelWithBottleneck(NetworkWithCompressionModule):
    """ """

    def __init__(
        self,
        compressor: CompressionModel,
        backbone: nn.Module,
        analyzers_config: Dict[str, Any] = None,
    ):
        super(DiscriminativeModelWithBottleneck, self).__init__(
            compressor, backbone, analyzers_config
        )
        self._use_compression: bool = False

    def forward_compress(self):
        return self.compressor_updated and self._use_compression and not self.training

    def forward(self, x, *args, **kwargs):
        if self.forward_compress():
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(**compressed_obj)
        else:
            h = self.compression_module(x)
        scores = self.backbone(h)
        return scores

    @property
    def use_compression(self):
        return self._use_compression

    @use_compression.setter
    def use_compression(self, value):
        self._use_compression = value

    @property
    def stride(self):
        return self.backbone.stride

    @property
    def names(self):
        return self.backbone.names


"""
TODO: refactor 
    - filter_configs List[Dict]
    - recursively wrap compressor
    - process/return accordingly
"""


@register_model_class
class DiscriminativeModelWithBottleneckAndNeuralFilter(
    DiscriminativeModelWithBottleneck
):
    def __init__(
        self,
        compressor: CompressionModel,
        filter_config: Dict[str, Dict[str, Any]],
        backbone: nn.Module,
        analyzers_config: Dict[str, Any] = None,
        filter_threshold: Optional[float] = None,
    ):
        super().__init__(
            compressor,
            backbone,
            analyzers_config,
        )
        self.filter_threshold = filter_config.get("filter_threshold", None)
        self.compression_module = get_bottleneck_wrapper(
            filter_config["name"],
            **{
                "compressor": self.compression_module,
                **(filter_config.get("params") or dict()),
            },
        )

    def forward(self, x):
        if self.forward_compress() and not self.training:
            # todo: Drop preds on compression, return empty string if dropped
            compressed_obj = self.compression_module.compress(x)
            if self.activated_analysis:
                self.analyze(compressed_obj, img_shape=x.shape)
            h = self.compression_module.decompress(**compressed_obj)
        # todo
        # elif self.filter_threshold is not None:
        # h, filter_decisions = self.compression_module(
        #     x, drop_threshold=self.filter_threshold
        # )
        else:
            h = self.compression_module(x)
            filter_decisions = None
        scores = self.backbone(h)
        return scores


@register_model_class
class NetworkWithInputCompression(nn.Module):
    def __init__(
        self,
        predictor_config: Dict[str, Any],
        compressor_config: Dict[str, Any],
        post_transform_params: Dict[str, Any],
        neural: bool,
        skip_prediction: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.predictor = get_model(
            predictor_config["name"], **predictor_config["params"]
        )
        self.compressor_is_neural = neural
        self.compression_module = compressor_config
        self.post_transform = build_transform(post_transform_params)
        self.neural = neural
        self.skip_prediction = skip_prediction
        self.device = device
        self.to_pil = transforms.ToPILImage()

    @property
    def compression_module(self):
        return self._compression_module

    @compression_module.setter
    def compression_module(self, compressor_config):
        if self.compressor_is_neural:
            self._compression_module = get_compressai_model(
                compressor_config["name"], **compressor_config["params"]
            )
        else:
            self._compression_module = build_transform(compressor_config).transforms[0]

    def forward_compress_decompress(self, x: Image, save_to=None, x_unprocessed=None):
        h = x.height
        w = x.width
        if not self.neural:
            x_h, file_size = self.compression_module(x_unprocessed, save_to=save_to)
        else:
            raise NotImplementedError
        # byte to bit
        bpp = file_size * 8 / (h * w)
        return x_h, bpp

    def forward(
        self, x, return_avg_bpp: bool = False, skip_prediction: bool = False, **kwargs
    ):
        tmp_list = list()
        bpps = list()
        for sub_x in x:
            if isinstance(sub_x, torch.Tensor):
                sub_x = self.to_pil(sub_x)
            h = sub_x.height
            w = sub_x.width
            sub_x, file_size = self.compression_module(sub_x)
            if self.post_transform is not None:
                sub_x = self.post_transform(sub_x)
                bpps.append(file_size * 8 / (h * w))
            tmp_list.append(sub_x.unsqueeze(0))
        x = torch.hstack(tmp_list).to(self.device)
        if not skip_prediction:
            x = self.predictor(x)
        if return_avg_bpp:
            return x, sum(bpps) * 1.0 / len(bpps)

        return x

    def update(self, force=False) -> bool:
        if self.neural:
            updated = self.compression_module.update(force=force)
            self.compressor_updated = updated
        else:
            self.compressor_updated = True
        return self.compressor_updated

    @property
    def stride(self):
        return self.predictor.stride

    @property
    def names(self):
        return self.predictor.names


@register_model_class
class NetworkWithInputCompression2(nn.Module):
    """
    1. pass model to Ultralytics evaluator
        forward pass with BPG/PIL Module:
            recon, filesize
            recon = post_transform(recon) # to_tensor
            self.rate_analyzer.update(recon)
    2. get rate summary from analyzer from wrapper
    """

    def __init__(
        self,
        predictor_config: Dict[str, Any],
        codec_params: Dict[str, Any],
        detection_transforms: Optional[Dict[str, Any]] = None,
        skip_prediction: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.predictor = get_model(
            predictor_config["name"], **predictor_config["params"]
        )
        self.codec = build_transform(codec_params)
        self.detection_transforms = build_transform(detection_transforms) or transforms.ToTensor()
        self.skip_prediction = skip_prediction
        self.device = device
        self.to_pil = transforms.ToPILImage()
        self.rate_analyzer = RateAnalyzer(unit="byte")


    def forward(
        self,
        x: Tensor | Image,
        skip_prediction: bool = False,
    ):
        if isinstance(x, torch.Tensor):
            assert x.shape[0] == 1, "Set batch size =1"
            x = self.to_pil(x.squeeze(dim=0))
        h = x.height
        w = x.width
        recon_x, file_size = self.codec(x)
        self.rate_analyzer.update(file_size, img_shape=(1, 3, h, w))
        recon_x = self.detection_transforms(recon_x).unsqueeze(dim=0).to(next(self.predictor.parameters()).device)
        if skip_prediction:
            return recon_x
        res = self.predictor(recon_x)
        return res

    # def update(self, force=False) -> bool:
    #     if self.neural:
    #         updated = self.compression_module.update(force=force)
    #         self.compressor_updated = updated
    #     else:
    #         self.compressor_updated = True
    #     return self.compressor_updated

    @property
    def stride(self):
        return self.predictor.stride

    @property
    def names(self):
        return self.predictor.names


@register_model_func
def network_with_input_compressor(
    model_name: str,
    compression_module_config: Dict[str, Any],
    predictor_model_config: Dict[str, Any],
    post_transform_params: Dict[str, any],
    neural: bool,
    skip_prediction: bool = False,
):
    predictor = get_model(
        model_name=predictor_model_config["name"], **predictor_model_config["params"]
    )
    network = get_model(
        model_name=model_name,
        compressor_config=compression_module_config,
        predictor=predictor,
        post_transform_params=post_transform_params,
        neural=neural,
        skip_prediction=skip_prediction,
    )
    return network


@register_model_func
def network_with_bottleneck_injection(
    compression_module_config: Dict[str, Any],
    network_type: str,
    from_cai: bool = False,
    backbone_module_config: Dict[str, Any] = None,
    analyzers_config: Dict[str, Any] = None,
    *args,
    **kwargs
) -> NetworkWithCompressionModule:
    if from_cai:
        compression_module = get_compressai_model(
            compression_module_config["name"], **compression_module_config["params"]
        )
    else:
        compression_module = get_compression_module(
            compression_module_config["name"], **compression_module_config["params"]
        )

    if backbone_module_config:
        backbone_module = get_model(
            model_name=backbone_module_config["name"],
            **backbone_module_config["params"],
        )
    else:
        logger.info("Backbone is identity function..")
        backbone_module = nn.Identity()
    network = get_model(
        model_name=network_type,
        compressor=compression_module,
        backbone=backbone_module,
        analyzers_config=analyzers_config,
        *args,
        **kwargs,
    )
    return network
