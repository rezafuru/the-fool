import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from kornia_rs import Tensor

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, converter
from ultralytics.utils import ops
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.ops import Profile
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import (
    de_parallel,
    select_device,
    smart_inference_mode,
)

from misc.frames_ultralytics_dataset import (
    build_frames_yolo_dataset,
    build_sequenced_yolo_dataset,
)
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)
def fwd_frames_simplified(self, im: List[Tensor], augment=False, visualize=False):
    """
    """
    if hasattr(self, "fp16") and self.fp16 and im[0].dtype != torch.float16:
        im = [i.half() for i in im]  # to FP16
    if hasattr(self, "nhwc") and self.nhwc:
        im = [i.permute(0, 2, 3, 1) for i in im]  # torch BCHW to numpy BHWC shape(1,320,192,3)

    assert self.pt or self.nn_module, "Only PyTorchonica :)"
    y = self.model(im)

    if isinstance(y, (list, tuple)):
        return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
    else:
        return self.from_numpy(y)




class FramesBaseValidator:
    """
    BaseValidator

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.model = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(
            parents=True, exist_ok=True
        )
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        patched_aa = AutoBackend
        patched_aa.forward = fwd_frames_simplified
        self.patched_aa = patched_aa

    @torch.no_grad()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        assert not self.training and not augment, "Only use this validator for eval"

        callbacks.add_integration_callbacks(self)
        self.run_callbacks("on_val_start")
        model = self.patched_aa(
            model or self.args.model,
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1  # export.py models default to batch-size 1
            logger.info(
                f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models"
            )

        if isinstance(self.args.data, str) and self.args.data.split(".")[-1] in (
            "yaml",
            "yml",
        ):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == "classify":
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(
                emojis(
                    f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                )
            )

        if self.device.type in ("cpu", "mps"):
            self.args.workers = (
                0  # faster CPU val as time dominated by inference, not dataloading
            )
        if not pt:
            self.args.rect = False
        self.dataloader = self.dataloader or self.get_dataloader(
            # self.data.get(self.args.split), self.args.batch
            self.data.get(self.args.split), self.args.batch
        )

        model.eval()
        # model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for frames_i, frames in enumerate(bar):
            if len(frames) == 1:
                continue
            self.run_callbacks("on_val_batch_start")
            self.batch_i = frames_i
            # Preprocess
            with dt[0]:
                frames = [self.preprocess(f) for f in frames]

            # pass frames as list at once, but evaluate predictions iteratively
            with dt[1]:
                preds = model([frame["img"] for frame in frames], augment=augment)

            # Loss
            # with dt[2]:
            #     if self.training:
            #         self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = [self.postprocess(list(p)) for p in preds]
            for frame_idx, pred_frame in enumerate(preds):
                self.update_metrics(pred_frame, frames[frame_idx])
                if self.args.plots and frame_idx < 3:
                    self.plot_val_samples(frames[frame_idx], frame_idx)
                    self.plot_predictions(frames[frame_idx], pred_frame, frame_idx)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(
                    self.loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            logger.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    logger.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(
                    iou >= threshold
                )  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[
                            iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        ]
                        matches = matches[
                            np.unique(matches[:, 1], return_index=True)[1]
                        ]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[
                            np.unique(matches[:, 0], return_index=True)[1]
                        ]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError(
            "get_dataloader function not implemented for this validator"
        )

    def build_dataset(self, img_path):
        """Build dataset"""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass


class DetectionValidatorSequenced(FramesBaseValidator):
    """
    """

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor(
                (width, height, width, height), device=self.device
            )
            self.lb = (
                [
                    torch.cat(
                        [
                            batch["cls"][batch["batch_idx"] == i],
                            bboxes[batch["batch_idx"] == i],
                        ],
                        dim=-1,
                    )
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    # def preprocess(self, batch):
    #     """Preprocesses batch of images for YOLO training."""
    #     frames_in_batch = []
    #     for frame_in_batch in batch:
    #         frame_in_batch["img"] = frame_in_batch["img"].to(self.device, non_blocking=True)
    #         frame_in_batch["img"] = (
    #             frame_in_batch["img"].half()
    #             if self.args.half
    #             else frame_in_batch["img"].float()
    #         ) / 255
    #         for k in ["batch_idx", "cls", "bboxes"]:
    #             frame_in_batch[k] = frame_in_batch[k].to(self.device)
    #
    #         if self.args.save_hybrid:
    #             height, width = frame_in_batch["img"].shape[2:]
    #             nb = len(frame_in_batch["img"])
    #             bboxes = frame_in_batch["bboxes"] * torch.tensor(
    #                 (width, height, width, height), device=self.device
    #             )
    #             self.lb = (
    #                 [
    #                     torch.cat(
    #                         [
    #                             frame_in_batch["cls"][frame_in_batch["batch_idx"] == i],
    #                             bboxes[frame_in_batch["batch_idx"] == i],
    #                         ],
    #                         dim=-1,
    #                     )
    #                     for i in range(nb)
    #                 ]
    #                 if self.args.save_hybrid
    #                 else []
    #             )  # for autolabelling
    #
    #         frames_in_batch.append(frame_in_batch)
    #     return frames_in_batch

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and val.endswith(f"{os.sep}val2017.txt")
        )  # is COCO
        self.class_map = (
            converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        )
        self.args.save_json |= (
            self.is_coco and not self.training
        )  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            correct_bboxes = torch.zeros(
                npr, self.niou, dtype=torch.bool, device=self.device
            )  # init
            self.seen += 1

            if npr == 0:
                if nl:  #
                    self.stats.append(
                        (
                            correct_bboxes,
                            *torch.zeros((2, 0), device=self.device),
                            cls.squeeze(-1),
                        )
                    )
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, labels=cls.squeeze(-1)
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(
                batch["img"][si].shape[1:],
                predn[:, :4],
                shape,
                ratio_pad=batch["ratio_pad"][si],
            )  # native-space pred

            # Evaluate
            if nl:
                height, width = batch["img"].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device
                )  # target boxes
                ops.scale_boxes(
                    batch["img"][si].shape[1:],
                    tbox,
                    shape,
                    ratio_pad=batch["ratio_pad"][si],
                )  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append(
                (correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1))
            )  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = (
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                )
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(
            stats[-1].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        logger.info(
            pf
            % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            logger.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                logger.info(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        self.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_sequenced_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=gs
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1
        )  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (
                (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            )  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = (
                self.data["path"] / "annotations/instances_val2017.json"
            )  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            logger.info(
                f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}..."
            )
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(
                    str(pred_json)
                )  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, "bbox")
                if self.is_coco:
                    eval.params.imgIds = [
                        int(Path(x).stem) for x in self.dataloader.dataset.im_files
                    ]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[
                    :2
                ]  # update mAP50-95 and mAP50
            except Exception as e:
                logger.warning(f"pycocotools unable to run: {e}")
        return stats


class DetectionValidatorFrames(FramesBaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (
            batch["img"].half() if self.args.half else batch["img"].float()
        ) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor(
                (width, height, width, height), device=self.device
            )
            self.lb = (
                [
                    torch.cat(
                        [
                            batch["cls"][batch["batch_idx"] == i],
                            bboxes[batch["batch_idx"] == i],
                        ],
                        dim=-1,
                    )
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    # def preprocess(self, batch):
    #     """Preprocesses batch of images for YOLO training."""
    #     frames_in_batch = []
    #     for frame_in_batch in batch:
    #         frame_in_batch["img"] = frame_in_batch["img"].to(self.device, non_blocking=True)
    #         frame_in_batch["img"] = (
    #             frame_in_batch["img"].half()
    #             if self.args.half
    #             else frame_in_batch["img"].float()
    #         ) / 255
    #         for k in ["batch_idx", "cls", "bboxes"]:
    #             frame_in_batch[k] = frame_in_batch[k].to(self.device)
    #
    #         if self.args.save_hybrid:
    #             height, width = frame_in_batch["img"].shape[2:]
    #             nb = len(frame_in_batch["img"])
    #             bboxes = frame_in_batch["bboxes"] * torch.tensor(
    #                 (width, height, width, height), device=self.device
    #             )
    #             self.lb = (
    #                 [
    #                     torch.cat(
    #                         [
    #                             frame_in_batch["cls"][frame_in_batch["batch_idx"] == i],
    #                             bboxes[frame_in_batch["batch_idx"] == i],
    #                         ],
    #                         dim=-1,
    #                     )
    #                     for i in range(nb)
    #                 ]
    #                 if self.args.save_hybrid
    #                 else []
    #             )  # for autolabelling
    #
    #         frames_in_batch.append(frame_in_batch)
    #     return frames_in_batch

    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and val.endswith(f"{os.sep}val2017.txt")
        )  # is COCO
        self.class_map = (
            converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        )
        self.args.save_json |= (
            self.is_coco and not self.training
        )  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            correct_bboxes = torch.zeros(
                npr, self.niou, dtype=torch.bool, device=self.device
            )  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append(
                        (
                            correct_bboxes,
                            *torch.zeros((2, 0), device=self.device),
                            cls.squeeze(-1),
                        )
                    )
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, labels=cls.squeeze(-1)
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(
                batch["img"][si].shape[1:],
                predn[:, :4],
                shape,
                ratio_pad=batch["ratio_pad"][si],
            )  # native-space pred

            # Evaluate
            if nl:
                height, width = batch["img"].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device
                )  # target boxes
                ops.scale_boxes(
                    batch["img"][si].shape[1:],
                    tbox,
                    shape,
                    ratio_pad=batch["ratio_pad"][si],
                )  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append(
                (correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1))
            )  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = (
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                )
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(
            stats[-1].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        logger.info(
            pf
            % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            logger.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                logger.info(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        self.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.names.values(),
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_frames_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=gs
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1
        )  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (
                (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            )  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = (
                self.data["path"] / "annotations/instances_val2017.json"
            )  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            logger.info(
                f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}..."
            )
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(
                    str(pred_json)
                )  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, "bbox")
                if self.is_coco:
                    eval.params.imgIds = [
                        int(Path(x).stem) for x in self.dataloader.dataset.im_files
                    ]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[
                    :2
                ]  # update mAP50-95 and mAP50
            except Exception as e:
                logger.warning(f"pycocotools unable to run: {e}")
        return stats
