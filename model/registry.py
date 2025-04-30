from typing import Any, Callable

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_factorized_relu,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
    ssf2020,
)
from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_class

SYNTHESIS_NETWORK_DICT = dict()
COMPRESSOR_DICT = dict()
ANALYSIS_NETWORK_DICT = dict()
HYBRID_CQNODE_DICT = dict()
HYPER_NETWORK_DICT = dict()
Q_CIRCUIT_DICT = dict()
CONTEXT_COMPONENT_DICT = dict()
CONTEXT_MODEL_DICT = dict()
CUSTOM_COMPRESSION_MODULE_DICT = dict()
BOTTLENECK_WRAPPE_DICT = dict()
# could just dynamically import and return method but w.e
COMPRESSAI_DICT = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_factorized_relu": bmshj2018_factorized_relu,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
    "mbt2018": mbt2018,
    "mbt2018_mean": mbt2018_mean,
    "cheng2020_anchor": cheng2020_anchor,
    "cheng2020_attn": cheng2020_attn,
    "ssf2020": ssf2020
}
logger = def_logger.getChild(__name__)


def get_compressai_model(name: str, **kwargs: Any) -> Callable:
    if name not in COMPRESSAI_DICT:
        raise ValueError(f"{name} is not in the CompressAI registry")
    return COMPRESSAI_DICT[name](**kwargs)


def register_bottleneck_wrapper(cls) -> Callable:
    BOTTLENECK_WRAPPE_DICT[cls.__name__] = cls
    return cls


def get_bottleneck_wrapper(wrapper_name, **kwargs):
    if wrapper_name not in BOTTLENECK_WRAPPE_DICT:
        raise ValueError(
            "Wrapper for bottleneck with name `{}` not registered".format(wrapper_name)
        )
    return BOTTLENECK_WRAPPE_DICT[wrapper_name](**kwargs)


def register_hyper_network(cls) -> Callable:
    HYPER_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_hyper_network(hyper_network_name, **kwargs):
    if hyper_network_name not in HYPER_NETWORK_DICT:
        raise ValueError(
            "hyper network with name `{}` not registered".format(hyper_network_name)
        )
    return HYPER_NETWORK_DICT[hyper_network_name](**kwargs)


def register_compression_module(
    cls: "model.compression_base CompressionModule",
) -> Callable:
    COMPRESSOR_DICT[cls.__name__] = cls
    return cls


def get_compression_module(
    compressor_name: str, **kwargs
) -> "model.compression_base CompressionModule":
    if compressor_name not in COMPRESSOR_DICT:
        raise ValueError(
            "compressor with name `{}` not registered".format(compressor_name)
        )
    return COMPRESSOR_DICT[compressor_name](**kwargs)


def register_synthesis_network(cls: "model.ntc.synthesis"):
    SYNTHESIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_synthesis_network(synthesis_network_name: str, **kwargs):
    if synthesis_network_name not in SYNTHESIS_NETWORK_DICT:
        raise ValueError(
            "synthesis network with name `{}` not registered".format(
                synthesis_network_name
            )
        )
    return SYNTHESIS_NETWORK_DICT[synthesis_network_name](**kwargs)


def register_analysis_network(cls):
    ANALYSIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_analysis_network(analysis_network_name, **kwargs):
    if analysis_network_name not in ANALYSIS_NETWORK_DICT:
        raise ValueError(
            "analysis network with name `{}` not registered".format(
                analysis_network_name
            )
        )

    return ANALYSIS_NETWORK_DICT[analysis_network_name](**kwargs)


def register_context_model(cls: nn.Module) -> nn.Module:
    CONTEXT_MODEL_DICT[cls.__name__] = cls
    return cls


def get_context_model(component_name: str, **kwargs) -> nn.Module:
    if component_name not in CONTEXT_MODEL_DICT:
        raise ValueError(
            "context model with name `{}` not registered".format(component_name)
        )
    return CONTEXT_MODEL_DICT[component_name](**kwargs)


def register_context_component(cls: nn.Module) -> nn.Module:
    CONTEXT_COMPONENT_DICT[cls.__name__] = cls
    return cls


def get_context_component(component_name: str, **kwargs) -> nn.Module:
    if component_name not in CONTEXT_COMPONENT_DICT:
        raise ValueError(
            "context component with name `{}` not registered".format(component_name)
        )
    return CONTEXT_COMPONENT_DICT[component_name](**kwargs)


@register_model_class
@register_analysis_network
@register_synthesis_network
@register_compression_module
class IdentityLayer(nn.Identity):
    """ """

    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__()
