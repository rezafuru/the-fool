from typing import Any, Mapping
from abc import abstractmethod

from torch import nn

from model.ntc.base import BaseNTC
from model.ntc.context.components import Conv1x1EntropyParameter, MaskedConv2d
from model.registry import get_context_component, register_context_model


class ContextModel(BaseNTC):
    def __init__(self):
        super(ContextModel, self).__init__()

    @property
    def entropy_parameters(self) -> nn.Module:
        return self.e_p

    @property
    def context_prediction(self) -> nn.Module:
        return self.c_p


@register_context_model
class BaselineContexModel(ContextModel):
    """
    Context Model from original JointAutoregressiveHierarchicalPriors model
    """

    def __init__(
        self,
        entropy_params_config: Mapping[str, Any],
        context_pred_config: Mapping[str, Any],
    ):
        super().__init__()
        self.e_p = Conv1x1EntropyParameter(**entropy_params_config)
        self.c_p = MaskedConv2d(**context_pred_config)
