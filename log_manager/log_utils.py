import os
from logging import FileHandler, Formatter
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional

from log_manager.wandb_metric_logger import WandBMetricLogger
from misc.util import make_parent_dirs, mkdir, uniquify
from torchdistill.common.constant import LOGGING_FORMAT, def_logger


def setup_log_file(log_file_path, mode="w"):
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode=mode)
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)


def prepare_local_log_file(
    test_only, log_file_path, config_path, start_epoch, overwrite=False
):
    eval_file = "_eval" if test_only else ""
    if log_file_path:
        log_file_path = (
            f"{os.path.join(log_file_path, Path(config_path).stem)}{eval_file}.log"
        )
    else:
        log_file_path = (
            f"{config_path.replace('config', 'logs', 1)}{eval_file}".replace(
                ".yaml", ".log", 1
            )
        )
    qubits = os.environ.get("QUBITS")
    if qubits:
        log_file_path = (
            Path(log_file_path).parent / f"qubits_{qubits}" / Path(log_file_path).name
        )
    if start_epoch == 0 or overwrite:
        log_file_path = uniquify(log_file_path)
        mode = "w"
    else:
        mode = "a"
    setup_log_file(os.path.expanduser(log_file_path), mode=mode)


def setup_wandb_and_metric_logger(
    wandb_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    include_top_levels: Optional[Iterable[str]]=None,
    resume: bool = False,
    wandb_run_id: Optional[str] = None,
) -> WandBMetricLogger:
    """
    If WandBConfig.enabled is set, it initializes a WandB run. Consequently, calls to WandBMetricLoggerWrapper
    will relay logs to wandb. If wandb is not enabled, the calls are simple no-ops
    TODO: Ideally we'd just log everything locally (including visualization) if not enabled
    """

    if include_top_levels:
        included_experiment_config = {
            top_level: experiment_config[top_level] for top_level in include_top_levels
        }
    else:
        included_experiment_config = experiment_config
    metric_logger = WandBMetricLogger(
        wandb_config=({**wandb_config, "resume": resume, "id": wandb_run_id}),
        experiment_config=included_experiment_config,
    )
    metric_logger.wandblogger.init()
    metric_logger.wandblogger.define_metric(
        prefix_path="validation/*", metric_name="epoch"
    )
    return metric_logger
