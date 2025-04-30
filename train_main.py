import datetime
import os
import time
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from log_manager.wandb_metric_logger import WandBMetricLogger
from misc.eval import get_eval_metrics
from misc.loss import LPIPSLoss, CharbonnierLoss
from model.models_util import (
    save_ckpt,
    summarize_model_and_calc_bottleneck_overhead,
)
from model.ntc.image.analysis import SequencedAnalysisWith3DAttention
from model.ntc.image.keypoint_catt import SideinfoWithKPCATT
from model.ntc.image.synthesis import SequencedSynthesisWith3DAttention
from model.ntc.recon.swinir import SwinIR

from log_manager.log_utils import setup_wandb_and_metric_logger
from model.network import NetworkWithCompressionModule

from torchdistill.common import module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import (
    is_main_process,
    load_ckpt,
    set_seed,
)
from torchdistill.core.distillation import DistillationBox
from torchdistill.core.training import TrainingBox
from torchdistill.datasets import util as dataset_util
from torchdistill.datasets.util import build_data_loader
from torchdistill.misc.log import SmoothedValue
from train.train_util import (
    build_update_string,
    common_setup,
    compute_and_apply_aux_loss,
    get_training_box,
    load_params_metrics,
    get_argparser,
    get_student_teacher,
    validate,
)

logger = def_logger.getChild(__name__)

torch.autograd.set_detect_anomaly(True)


def train_one_epoch(
        training_box: Union[TrainingBox, DistillationBox],
        metric_logger: WandBMetricLogger,
        device: str,
        epoch: int,
        log_freq: int,
        apply_aux_loss: bool = False,
):
    # uncomment when you get funky gradient computation errors
    # torch.autograd.set_detect_anomaly(True)
    model: NetworkWithCompressionModule = (
        training_box.student_model
        if hasattr(training_box, "student_model")
        else training_box.model
    )
    model_without_ddp = model.module if module_util.check_if_wrapped(model) else model
    if isinstance(model_without_ddp, nn.Sequential):
        model_without_ddp = model_without_ddp[0]
    model.to(device)
    model.train()
    metric_logger.add_meter(
        name=f"lr", meter=SmoothedValue(window_size=1, fmt="{value}")
    )
    metric_logger.add_meter(
        name="img/s", meter=SmoothedValue(window_size=10, fmt="{value:.2f}")
    )
    header = f"Epoch: [{epoch}]"
    for batch in metric_logger.log_every(
            training_box.train_data_loader, log_freq, header
    ):
        start_time = time.time()
        if len(batch) == 2:
            (samples, targets), supp_dict = batch
        else:
            samples, targets, supp_dict = batch
        if isinstance(samples, list):
            batch_size = samples[0].shape[0]
            samples = [s.to(device) for s in samples]
            fac = len(samples)
        else:
            batch_size = samples.shape[0]
            samples = samples.to(device)
            fac = 1

        targets = (
            [t.to(device) for t in targets]
            if isinstance(targets, list)
            else targets.to(device)
        )
        # configure LazyGeneralizedCustomLoss to get individual components of the loss
        loss, io_dict = training_box(samples, targets, supp_dict, return_io_dict=True)
        scalars = dict()
        if isinstance(loss, dict):
            total_loss = 0
            for loss_name, loss_val in loss.items():
                total_loss += loss_val
                scalars[loss_name] = loss_val.detach().item()
                # scalars[loss_name] = loss_val
            scalars["total_loss"] = total_loss
            loss = total_loss
        else:
            scalars["loss"] = loss.item()

        scalars["lr"] = training_box.optimizer.param_groups[0]["lr"]
        training_box.update_params(loss)
        aux_loss = None
        if apply_aux_loss:
            aux_loss = compute_and_apply_aux_loss(model_without_ddp.aux_loss())
        if aux_loss:
            scalars["aux_loss"] = aux_loss.item()
        metric_logger.update(scalars=scalars, io_dict=io_dict)
        metric_logger.meters["img/s"].update(
            (batch_size * fac) / (time.time() - start_time)
        )
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError("Detected faulty loss = {}".format(loss))

    metric_logger.clear()


def train(
        student_model: nn.Module,
        teacher_model: Optional[nn.Module],
        train_config: Dict[str, Any],
        training_box: TrainingBox,
        ckpt_path: os.PathLike,
        device: str,
        metric_logger: WandBMetricLogger,
        resume: bool,
        main_metric: str,
        pre_eval_teacher: bool = False,
) -> NetworkWithCompressionModule:
    log_freq = train_config["log_freq"]
    aux_loss_epochs = train_config["aux_loss_epochs"]

    # todo: eval_metrics by stage for multistage training

    eval_metrics = get_eval_metrics(
        train_config["stage1"]["eval_metrics"]
        if "stage1" in train_config
        else train_config["eval_metrics"]
    )
    if pre_eval_teacher:
        #                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        # DOTAv2.0 tiled:  all      16022      85793      0.646      0.458      0.485      0.299
        logger.info("Evaluating teacher before training...")
        map_eval = eval_metrics["map"]
        results = map_eval.eval_func(model=teacher_model, device=device)
        logger.info(f"Teacher results: {results}")

    student_model_without_ddp = (
        student_model.module
        if module_util.check_if_wrapped(student_model)
        else student_model
    )
    track_metrics = load_params_metrics(
        training_box=training_box,
        student_model=student_model,
        resume=resume,
        ckpt_path=ckpt_path,
    )
    start_epoch = training_box.current_epoch
    curr_stage = (
        training_box.current_stage if hasattr(training_box, "current_stage") else 1
    )
    half_after = train_config.get("half_after")
    start_time = time.time()
    for epoch in range(start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        if (
                hasattr(training_box, "current_stage")
                and training_box.current_stage != curr_stage
        ):
            curr_stage = training_box.current_stage
            eval_metrics = get_eval_metrics(
                training_box.train_config[f"stage{curr_stage}"]["eval_metrics"]
            )
        if epoch == half_after:
            # easier to setup than implementing multiple scheduler support
            logger.info("Halving learning rate manually")
            for g in training_box.optimizer.param_groups:
                g['lr'] /= 2
        train_one_epoch(
            training_box=training_box,
            metric_logger=metric_logger,
            device=device,
            epoch=epoch,
            log_freq=log_freq,
            apply_aux_loss=epoch < aux_loss_epochs,
        )

        for eval_metric in eval_metrics.values():
            eval_metric.viz_on_best(
                model=student_model_without_ddp,
                device=device,
                wdb_logger=metric_logger.wandblogger,
                epoch=epoch,
            )
        # todo optional visualization and logging to wandb
        if (epoch + 1) % train_config.get("eval_frequency", 1) == 0:
            result_dict = validate(
                model=student_model_without_ddp,
                loader=training_box.val_data_loader,
                wdb_logger=metric_logger.wandblogger,
                device=device,
                log_freq=log_freq,
                eval_metrics=eval_metrics,
                epoch=epoch + 1,
            )
            update = result_dict[main_metric] < track_metrics[main_metric] if "lpips" in main_metric.lower() else \
                result_dict[main_metric] > track_metrics[main_metric]
            if update:
                #            if comparator(
                #                main_metric,
                #                curr_best=track_metrics[main_metric],
                #                result=result_dict[main_metric],
                #            ):
                logger.info(
                    build_update_string(
                        main_metric=main_metric,
                        track_metrics=track_metrics,
                        result_dict=result_dict,
                    )
                )
                # logger.info(f"Best {main_metric}: {track_metrics[main_metric]:.4f} -> {result_dict[main_metric]:.4f} with bpp {track_metrics['bpp']:.4f} -> {result_dict['bpp']:.4f}")
                track_metrics = result_dict
                logger.info("Updating ckpt at {}".format(ckpt_path))
                save_ckpt(
                    model=student_model_without_ddp,
                    optimizer=training_box.optimizer,
                    lr_scheduler=training_box.lr_scheduler,
                    eval_metrics=track_metrics,
                    output_file_path=ckpt_path,
                )
                logger.info("Visualizing output of new best model...")
            for eval_metric in eval_metrics.values():
                eval_metric.viz_on_best(
                    model=student_model_without_ddp,
                    device=device,
                    wdb_logger=metric_logger.wandblogger,
                    epoch=epoch,
                )
        training_box.post_process(metrics=track_metrics[main_metric])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Finished Training.. total time {}".format(total_time_str))
    training_box.clean_modules()


def test(
        test_config: Dict[str, Any],
        test_loader: Optional[DataLoader],
        metric_logger: WandBMetricLogger,
        device: str,
        student_model: NetworkWithCompressionModule,
        teacher_model: nn.Module = None,
):
    cudnn.deterministic = True
    log_freq = test_config.get("log_freq", 1000)
    eval_metrics = get_eval_metrics(test_config["eval_metrics"])
    logger.info(
        "Updating Student Model => quantizing latent instead of sampling noise"
    )
    student_model.update()
    student_model.use_compression = True
    test_results = validate(
        model=student_model,
        loader=test_loader,
        wdb_logger=metric_logger.wandblogger,
        device=device,
        log_freq=log_freq,
        eval_metrics=eval_metrics,
    )

    return test_results


def main(args: Any):
    set_seed(args.seed)
    cudnn.deterministic = True
    ckpt_path = args.config.replace("config", "resources/weights", 1).replace(
        ".yaml", ".pt", 1
    )

    config = yaml_util.load_yaml_file(args.config)
    common_setup(config, args)
    models_config = config["model"]
    student_model_config = models_config.get("student", models_config)
    student_model_config["ckpt"] = ckpt_path
    teacher_model_config = models_config.get("teacher", None)
    logger.info(
        f"\n{120 * '-'}\n"
        f"{'Testing' if args.test_only else 'Profiling' if args.profile_only else 'Starting run'}"
        f"with config {args.config}\n"
        f" {120 * '-'}"
    )

    # Load params here if args.te  st_only, otherwise load all persisted artefacts in train
    student, teacher = get_student_teacher(
        student_model_config=student_model_config,
        teacher_model_config=teacher_model_config,
        device=args.device,
        load_student_params=args.test_only,
    )
    (
        summary_str,
        summary_params,
        model_statistics,
    ) = summarize_model_and_calc_bottleneck_overhead(
        device=args.device,
        bnet_injected_model=student,
        base_model=teacher,
        input_size=config.get("input_spatial_dim"),
    )
    logger.info(summary_str)
    if args.profile_only:
        # logger.info(f"Model Statistics:\n{model_statistics}")
        # logger.info(f"Summary Params:\n{summary_params}")
        return
    wandb_config = config.get("wandb")
    metric_logger = setup_wandb_and_metric_logger(
        wandb_config=wandb_config,
        experiment_config=config,
        resume=args.resume,
        wandb_run_id=args.wandb_id,
    )
    datasets_dict = dataset_util.get_all_datasets(config["datasets"])
    train_config = config["train"]
    training_box = get_training_box(
        student_model=student,
        teacher_model=teacher,
        dataset_dict=datasets_dict,
        train_config=train_config,
        device=args.device,
    )

    if not args.test_only:
        train(
            teacher_model=teacher,
            student_model=student,
            training_box=training_box,
            ckpt_path=ckpt_path,
            device=args.device,
            train_config=train_config,
            metric_logger=metric_logger,
            resume=args.resume,
            main_metric=student_model_config.get("main_metric", "map"),
            pre_eval_teacher=args.pre_eval_teacher,
        )
    else:
        torch.cuda.synchronize()
        logger.info("Loading best performing model")
        test_config = config["test"]
        test_loader_config = test_config.get("test_data_loaders")
        if test_loader_config is None:
            logger.info("Finishing without Testing best performing model")
            return

        test_loaders = {
            path: build_data_loader(
                datasets_dict[loader_config["dataset_id"]],
                loader_config,
                False,
                None,
            )
            for path, loader_config in test_loader_config.items()
        }
        # test_loaders = build_data_loader(
        #     datasets_dict[test_loader_config["dataset_id"]],
        #     test_loader_config,
        #     False,
        #     None,
        # )
        # test_loaders = {
        #     path: build_data_loader(
        #         datasets_dict[loader_config["dataset_id"]],
        #         loader_config,
        #         False,
        #         None,
        #     )
        #     for path, loader_config in test_loader_config.items()
        # } if isinstance(test_loader_config, dict) else build_data_loader(
        #     datasets_dict[test_loader_config["dataset_id"]],
        #     test_loader_config,
        #     False,
        #     None,
        # )
        load_ckpt(model=student, ckpt_path=ckpt_path)
        test(
            student_model=student,
            teacher_model=teacher if args.test_teacher and teacher else None,
            device=args.device,
            test_config=test_config,
            test_loader=test_loaders,
            metric_logger=metric_logger,
        )
    metric_logger.finish()


if __name__ == "__main__":
    main(
        get_argparser(
            description="", task="leo-compress"
        ).parse_args()
    )
