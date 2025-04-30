import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import torch
from compressai.models import CompressionModel
from torch import Tensor, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

from log_manager.log_utils import prepare_local_log_file
from log_manager.wandblogger import WandbLogger
from misc.eval import EvaluationMetric
from misc.util import make_parent_dirs, overwrite_config
from model.models_util import load_model
from model.ntc.image.image_base import ModularImageCompressionModel
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import load_ckpt, set_seed
from torchdistill.core.distillation import DistillationBox, get_distillation_box
from torchdistill.core.training import TrainingBox
from torchdistill.core.training import get_training_box as get_td_training_box

logger = def_logger.getChild(__name__)


def compute_and_apply_aux_loss(aux_loss_or_list: Union[List[Tensor], Tensor]):
    if not isinstance(aux_loss_or_list, list):
        aux_loss_or_list.backward()
        return aux_loss_or_list

    aux_loss_sum = 0
    for aux_loss in aux_loss_or_list:
        aux_loss_sum += aux_loss
        aux_loss.backward()

    return aux_loss_sum


def load_df(
        col_names: List[str], path: Optional[Union[str, bytes, os.PathLike]] = None
):
    """
    Creates pandas dataframe according to col_names. Loads existing rows if path is set (and exists)
    """
    make_parent_dirs(path)
    if path and os.path.isfile(path):
        logger.info(f"Loading existing results into dataframe from {path}")
        df = pd.read_csv(path, names=col_names, index_col=None, sep=",", skiprows=[0])
    else:
        logger.info("Creating empty dataframe")
        df = pd.DataFrame(columns=col_names, index=None)
    return df


def parse_col_values_from_conf(conf: Dict[str, Any], col_names):
    result_dict = dict()
    for col_name in col_names:
        result_dict[col_name] = conf[col_name]
    return result_dict


def wandb_rows_to_pandas_dict(
        col_names: List[str], data: List[List[Any]], summarize: bool = False
) -> Dict[str, List[Any]]:
    # assume that data is in correct oder, and col_names is a subset in the corresponding order

    result_dict = defaultdict(list)
    for idx, entries in enumerate(data):
        entry = data[idx]
        data[idx] = entry[: len(col_names)]
        for column_idx, value in enumerate(entry[: len(col_names)]):
            result_dict[col_names[column_idx]].append(value)
    if summarize:
        result_dict = {k: [sum(v) / len(v)] for k, v in result_dict.items()}
    return result_dict


def get_student_teacher(
        device: str,
        load_student_params: bool,
        student_model_config: Mapping[str, Any],
        teacher_model_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[ModularImageCompressionModel, Optional[nn.Module]]:
    """
    Constructs student (and teacher if present in config)
    """
    # todo: (D)DP wrapper?
    student: ModularImageCompressionModel = load_model(
        student_model_config,
        device=device,
        skip_ckpt=not load_student_params,
    )  # load params from checkpoint with args.resume
    teacher: nn.Module = (
        load_model(teacher_model_config, device=device, skip_ckpt=False)
        if teacher_model_config
        else None
    )
    return student, teacher


def get_no_stages(train_config) -> int:
    return sum(map(lambda x: "stage" in x, train_config.keys()))


# def get_eval_metrics(
#     train_config: Mapping[str, Any]
# ) -> List[Mapping[str, EvaluationMetric]]:
#     stages = get_no_stages(train_config)
#     eval_metrics = []
#     if stages == 0:
#         stage_eval_metrics = {}
#         metrics = train_config.get("eval_metrics")
#         for metric in metrics:
#             stage_eval_metrics[metric] = get_eval_metric(metric)
#         eval_metrics.append(stage_eval_metrics)
#     else:
#         for stage in range(stages):
#             stage_eval_metrics = {}
#             stage_metrics = train_config.get(f"stage{stage + 1}").get("eval_metrics")
#             for metric in stage_metrics:
#                 stage_eval_metrics[metric] = get_eval_metric(metric)
#             eval_metrics.append(stage_eval_metrics)
#     return eval_metrics


def common_setup(config: Mapping[str, Any], args: argparse.Namespace):
    prepare_local_log_file(
        test_only=args.test_only,
        log_file_path=args.log_path,
        config_path=args.config,
        start_epoch=args.start_epoch,
        overwrite=False,
    )
    cudnn.benchmark = True
    cudnn.deterministic = True
    set_seed(args.seed)
    logger.info(json.dumps(config))
    if args.json_overwrite:
        logger.info("Overwriting config")
        overwrite_config(config, json.loads(args.json_overwrite))


def common_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="yaml configuration file path")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--log_path", help="log file folder path")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed in random number generator"
    )
    parser.add_argument("--test_only", action="store_true", help="only test the models")
    parser.add_argument(
        "--test_teacher",
        action="store_true",
        help="Test and compare the teacher to the student",
    )
    parser.add_argument(
        "--main_metric",
        default="psnr",
        help="Which Validation metric should be favored when updating the training checkpoint",
    )
    parser.add_argument("--skip_ckpt", action="store_true", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--torchdistill_scheduling",
        action="store_true",
        help="Give back scheduling control to torchdistill. Use when not using ReduceLROnPlateau",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes training according to persisted artifacts",
    )
    parser.add_argument(
        "--wandb_id", default=None, help="Wandb id to resume training from"
    )
    parser.add_argument(
        "--profile_only", action="store_true", help="Will only run and log profilers"
    )
    parser.add_argument(
        "--log_model_summary",
        action="store_true",
        help="Log extensive summary of the model",
    )
    parser.add_argument("--load_last_after_train", action="store_true")
    parser.add_argument("--aux_loss_stage", default=1, type=int)
    parser.add_argument(
        "--load_best_after_stage",
        action="store_true",
        help="Load the best performing model after each stage in multi-stage training",
    )
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--adjust_lr",
        action="store_true",
        help="multiply learning rate by number of distributed processes (world_size)",
    )
    parser.add_argument("--result_file", required=False, help="result file path")
    parser.add_argument(
        "--json_overwrite",
        required=False,
        help="json object to overwrite the config file",
    )
    parser.add_argument(
        "--neural", action="store_true", help="use neural compressor network"
    )
    parser.add_argument(
        "--results_file", required=False, help="path to results file", type=str
    )
    parser.add_argument("--pre_eval_teacher", action="store_true")
    return parser


def get_argparser(description: str, task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser = common_args(parser)

    return parser


def load_params_metrics(
        training_box: Union[TrainingBox, DistillationBox],
        student_model: ModularImageCompressionModel,
        resume: bool,
        ckpt_path: str,
) -> Dict[str, Any]:
    """
    Load persisted training artifacts and return tracked metrics
    """
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if resume:
        ckpt = load_ckpt(
            model=student_model,
            optimizer=optimizer,
            ckpt_path=ckpt_path,
            lr_scheduler=lr_scheduler,
        )
        track_metrics = ckpt["eval_metrics"]
        logger.info(f"Loaded stored metrics {track_metrics}")
    else:
        track_metrics = defaultdict(
            float,
            {
                "metrics/mAP50(B)": float("0"),
                "metrics/precision(B)": float("0"),
                "metrics/recall(B)": float("0"),
                "metrics/mAP50-95(B)": float("0"),
                "fitness": float("0"),
                "psnr": float("-inf"),
                "ms-ssim": float(0),
                "bpp": float("inf"),
                "lpips": float("inf"),
                "epoch": 0,
            },
        )
    start_epoch = track_metrics["epoch"]
    logger.info(f"Starting training from epoch {start_epoch}...")
    training_box.current_epoch = start_epoch
    return track_metrics


def get_training_box(
        student_model: ModularImageCompressionModel,
        teacher_model: Optional[CompressionModel],
        dataset_dict: Mapping[str, Dataset],
        train_config: Mapping[str, Any],
        device: str,
) -> Union[TrainingBox, DistillationBox]:
    device = torch.device(device)
    if teacher_model:
        return get_distillation_box(
            teacher_model=teacher_model,
            student_model=student_model,
            data_loader_dict=dataset_dict,
            train_config=train_config,
            device=device,
            device_ids=None,
            distributed=False,
            lr_factor=1,
        )

    return get_td_training_box(
        model=student_model,
        data_loader_dict=dataset_dict,
        train_config=train_config,
        device=device,
        device_ids=None,
        distributed=False,
        lr_factor=1,
    )


def build_update_string(
        main_metric: str,
        track_metrics: Mapping[str, float],
        result_dict: Mapping[str, float],
) -> str:
    res = f"\nBest {main_metric}: {track_metrics[main_metric]:.4f} -> {result_dict[main_metric]:.4f} with \n"
    for metric in result_dict:
        if metric == main_metric or metric == "epoch" or "viz" in metric:
            continue
        res += f"{metric}: {track_metrics[metric]:.4f} -> {result_dict[metric]:.4f} \n"
    return res


# def validate_multiset(
#     model: NetworkWithCompressionModule,
#     loader: Optional[DataLoader | Dict[str, DataLoader]],
#     wdb_logger: WandbLogger,
#     device: str,
#     log_freq: int,
#     eval_metrics: Dict[str, EvaluationMetric],
#     epoch: Optional[int] = None,
# ) -> Dict[str, float]:
#     if not loader:
#         return dict()
#     if not isinstance(loader, dict):
#         loader = {"noname": loader}
#     if epoch is not None:
#         prefix_wandb = "validation"
#         result_dict = {"epoch": epoch}
#     else:
#         prefix_wandb = "test (estimation)"
#         result_dict = dict()
#     for ultralytics_path, set_loader in loader.items():
#         prefix_metric = f"{Path(ultralytics_path).stem}_" if ultralytics_path != "noname" else ""
#         for name, eval_metric in eval_metrics.items():
#             eval_func = partial(
#                 eval_metric.eval_func,
#                 data_loader=set_loader,
#                 device=device,
#                 log_freq=log_freq,
#                 epoch=epoch or "Test",
#                 wdb_logger=wdb_logger,
#             )
#             res = eval_func(model=model, title=f"Evalution {name} Student:")
#             if isinstance(res, dict):
#                 result_dict.update(res)
#             else:
#                 result_dict[name] = res
#
#     # wdb_logger.log(result_dict, step=step or 0, key=key, prefix=key)
#         log_dict = {
#             f"{prefix_wandb}/{prefix_metric}{k.replace('metrics/', '')}" if k != "epoch" else k: v for k, v in result_dict.items()
#         }
#         wdb_logger.log(log_dict)
#     return result_dict
def validate(
        model: nn.Module,
        loader: Optional[DataLoader],
        wdb_logger: WandbLogger,
        device: str,
        log_freq: int,
        eval_metrics: Dict[str, EvaluationMetric],
        epoch: Optional[int] = None,
) -> Dict[str, float]:
    if epoch is not None:
        prefix = "validation"
        result_dict = {"epoch": epoch}
    else:
        prefix = "test"
        result_dict = dict()
    if loader:
        for name, eval_metric in eval_metrics.items():
            res = eval_metric.eval_func(
                model=model,
                title=f"Evalution {name} Student:",
                data_loader=loader,
                device=device,
                log_freq=log_freq,
                epoch=epoch or "Test",
                wdb_logger=wdb_logger,
                test_mode=epoch is None,
            )
            if isinstance(res, dict):
                result_dict.update(res)
            else:
                result_dict[name] = res

    # wdb_logger.log(result_dict, step=step or 0, key=key, prefix=key)
    log_dict = {
        f"{prefix}/{k.replace('metrics/', '')}" if k != "epoch" else k: v
        for k, v in result_dict.items()
    }
    wdb_logger.log(log_dict)
    return result_dict
