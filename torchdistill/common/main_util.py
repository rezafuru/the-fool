import builtins as __builtin__
import logging
import os
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import check_if_exists, make_parent_dirs
from torchdistill.common.module_util import check_if_wrapped

logger = def_logger.getChild(__name__)


def setup_for_distributed(is_master):
    """
    This function disables logging when not in master process
    """
    def_logger.setLevel(logging.INFO if is_master else logging.WARN)
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(seed):
    if not isinstance(seed, int):
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(world_size=1, dist_url="env://"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_id = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        device_id = rank % torch.cuda.device_count()
    else:
        logger.info("Not using distributed mode")
        return False, None

    torch.cuda.set_device(device_id)
    dist_backend = "nccl"
    logger.info("| distributed init (rank {}): {}".format(rank, dist_url))
    torch.distributed.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True, [device_id]


def load_ckpt(
    ckpt_path, model=None, optimizer=None, lr_scheduler=None, strict=True
) -> Mapping[str, Any]:
    if check_if_exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    elif isinstance(ckpt_path, str) and (
        ckpt_path.startswith("https://") or ckpt_path.startswith("http://")
    ):
        ckpt = torch.hub.load_state_dict_from_url(
            ckpt_path, map_location="cpu", progress=True
        )
    else:
        logger.info("ckpt file is not found at `{}`".format(ckpt_path))
        return None, None, None

    if model is not None:
        if "model" in ckpt:
            logger.info("Loading model parameters")
            if strict is None:
                model.load_state_dict(ckpt["model"], strict=strict)
            else:
                model.load_state_dict(ckpt["model"], strict=strict)
        elif optimizer is None and lr_scheduler is None:
            logger.info("Loading model parameters only")
            model.load_state_dict(ckpt, strict=strict)
        else:
            logger.info("No model parameters found")

    if optimizer is not None:
        if "optimizer" in ckpt:
            logger.info("Loading optimizer parameters")
            optimizer.load_state_dict(ckpt["optimizer"])
        elif model is None and lr_scheduler is None:
            logger.info("Loading optimizer parameters only")
            optimizer.load_state_dict(ckpt)
        else:
            logger.info("No optimizer parameters found")

    if lr_scheduler is not None:
        if "lr_scheduler" in ckpt:
            logger.info("Loading scheduler parameters")
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        elif model is None and optimizer is None:
            logger.info("Loading scheduler parameters only")
            lr_scheduler.load_state_dict(ckpt)
        else:
            logger.info("No scheduler parameters found")
    return ckpt


def save_ckpt(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    eval_metrics: Mapping[str, Any],
    output_file_path,
):
    make_parent_dirs(output_file_path)
    model_state_dict = (
        model.module.state_dict() if check_if_wrapped(model) else model.state_dict()
    )
    lr_scheduler_state_dict = (
        lr_scheduler.state_dict() if lr_scheduler is not None else None
    )
    save_on_master(
        {
            "model": model_state_dict,
            "optimizer": optimizer.state_dict(),
            "eval_metrics": eval_metrics,
            "lr_scheduler": lr_scheduler_state_dict,
        },
        output_file_path,
    )
