from typing import Any, List

import torch
from torch import Tensor, nn

from torchdistill.common.module_util import freeze_module_params
from torchdistill.models.registry import get_model, register_model_class


@register_model_class
class ReconModel(nn.Module):
    def __init__(self, recon_model_name, recon_model_config):
        super(ReconModel, self).__init__()
        self.recon_model = get_model(recon_model_name, **recon_model_config)

    def forward(self, x: Tensor) -> Tensor:
        return self.recon_model(x)


@register_model_class
class SequenceReconModel(ReconModel):
    def __init__(self, seq_len, recon_model_name, recon_model_config):
        super().__init__(recon_model_name, recon_model_config)
        self.seq_len = seq_len

    def forward(self, x: List[Tensor] | Tensor) -> List[Tensor]:
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
            B, D, C, H, W = x.shape
            x = x.reshape(B * D, C, H, W)

        recon = self.recon_model(x)
        B_D, C, H, W = recon.shape
        B = B_D // self.seq_len
        recon = recon.reshape(B, self.seq_len, C, H, W)
        recon = [t.squeeze(dim=1) for t in recon.split(1, dim=1)]
        return recon


@register_model_class
class SequenceReconModelWithFeatureExtractor(SequenceReconModel):
    def __init__(self, seq_len, recon_model_name, recon_model_config, feature_extractor_config):
        super().__init__(
            seq_len,
            recon_model_name,
            recon_model_config,
        )
        self.feature_extractor = get_model(
            feature_extractor_config["name"], **feature_extractor_config["params"]
        )
        freeze_module_params(self.feature_extractor)

    def forward(self, x: Tensor) -> List[Tensor]:
        if isinstance(x, list):
            x = torch.stack(x, dim=1)
            B, D, C, H, W = x.shape
            x = x.reshape(B * D, C, H, W)
        x = self.feature_extractor(x)
        x = super().forward(x)
        return x

    def load_compressor_state_dict(self, state_dict: Any, strict=True, **kwargs):
        # compression_module_state_dict = OrderedDict()
        # for key in list(state_dict.keys()):
        #     if key.startswith("compression_module."):
        #         compression_module_state_dict[key.replace("compression_module.", '', 1)] = state_dict[key]

        self.feature_extractor.load_state_dict(state_dict, strict=strict)
        super().load_state_dict(state_dict, strict=strict)
