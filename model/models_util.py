import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torchinfo import ModelStatistics, summary

from model.network import (
    DiscriminativeModelWithBottleneck,
    NetworkWithCompressionModule,
    NetworkWithPerceptualRecovery,
)
from model.ntc.image.image_base import ModularImageCompressionModel
from model.ntc.recon.recon_model import ReconModel

from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import save_on_master
from torchdistill.models.registry import get_model

logger = def_logger.getChild(__name__)


def save_ckpt(
        model: DiscriminativeModelWithBottleneck,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        eval_metrics: Mapping[str, Any],
        output_file_path: str,
        store_backbone: bool = False,
):
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    lr_scheduler_state_dict = (
        lr_scheduler.state_dict() if lr_scheduler is not None else None
    )
    optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
    if isinstance(model, NetworkWithPerceptualRecovery):
        state_dict = {"model": {"gan": model.stoch_dec.state_dict()}}
    elif isinstance(model, NetworkWithCompressionModule):
        state_dict = {
            "model": {
                "compressor": model.compression_module.state_dict(),
            },
        }
    else:
        state_dict = {
            "model": model.state_dict(),
        }
    state_dict.update(
        {
            "optimizer": optimizer_state_dict,
            "lr_scheduler": lr_scheduler_state_dict,
            "eval_metrics": eval_metrics,
        }
    )
    if store_backbone:
        state_dict["backbone"] = model.backbone.state_dict()
    save_on_master(state_dict, output_file_path)


def load_modular_model_weights(
        ckpt_file_path: Optional[Union[str, bytes, os.PathLike]],
        model: NetworkWithCompressionModule,
        strict: bool = True,
        compressor_only: bool = True,
) -> Union[nn.Module, NetworkWithCompressionModule]:
    if os.path.exists(ckpt_file_path):
        ckpt = torch.load(ckpt_file_path, map_location="cpu")
        if compressor_only and not isinstance(model, ReconModel):
            logger.info("Loading only embedded compressor parameters...")
            model.load_compressor_state_dict(
                ckpt["model"]["compressor"]
                if not isinstance(model, NetworkWithPerceptualRecovery)
                else ckpt["model"]["gan"],
                strict=strict,
            )
        else:
            logger.info("Loading model parameters...")
            model.load_state_dict(ckpt["model"], strict=strict)
    else:
        logger.warning(f"ckpt not found in {ckpt_file_path}, not loading any weights")
    return model


def load_model(
        model_config: Mapping[str, Any],
        device: str,
        skip_ckpt: bool = False,
        compressor_only: bool = True,
) -> Union[nn.Module, DiscriminativeModelWithBottleneck]:
    repo_or_dir = model_config.get("repo_or_dir", None)
    model = get_model(model_config["name"], repo_or_dir, **model_config["params"])
    pretrained_weights_path = model_config.get("pretrained_weights")
    if not skip_ckpt:
        ckpt_file_path = os.path.expanduser(model_config.get("ckpt"))
        load_modular_model_weights(
            ckpt_file_path, model=model, strict=True, compressor_only=compressor_only
        )
    elif pretrained_weights_path is not None:
        pretrained_weights_path = os.path.expanduser(pretrained_weights_path)
        assert Path(pretrained_weights_path).exists()
        logger.info(f"Founding pretrained weights in {pretrained_weights_path}...")
        load_modular_model_weights(
            ckpt_file_path=pretrained_weights_path,
            model=model,
            strict=False,
            compressor_only=compressor_only,
        )
    else:
        logger.info("Skipping loading from checkpoint...")
    return model.to(device)


def calc_total_model_params(
        model: nn.Module, input_size: Optional[Union[Iterable[int], int]] = None
) -> ModelStatistics:
    if isinstance(input_size, int):
        input_size = (1, 3, input_size, input_size)
    if input_size is not None:
        return summary(
            model,
            depth=10,
            verbose=0,
            # input_size=input_size,
            # col_names=["num_params", "input_size", "output_size"],
        )
    return summary(model, col_names=["num_params"], verbose=0)


def summarize_model_and_calc_bottleneck_overhead(
        device: str,
        bnet_injected_model: Union[NetworkWithCompressionModule, nn.Module],
        base_model: Optional[nn.Module] = None,
        input_size: Optional[Union[Iterable[int], int]] = None,
) -> Tuple[str, Dict[str, int], Dict[str, ModelStatistics]]:
    full_summary = dict()
    full_summary_modified = calc_total_model_params(bnet_injected_model, input_size)
    full_summary["Bottleneck Injected Model"] = full_summary_modified
    params_mod = full_summary_modified.total_params
    if not isinstance(bnet_injected_model, NetworkWithCompressionModule):
        (
            summary_str,
            summary_params,
        ) = f"Model with no Compressor params: {params_mod}", {
            "Total Params": params_mod
        }
    else:
        summary_str = f"""
            Total params compression module: {summary(bnet_injected_model.compression_module, verbose=0).total_params}
        """
        summary_params = 0
        full_summary = ""

    return summary_str, summary_params, full_summary
