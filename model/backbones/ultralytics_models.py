import types
from collections.abc import Iterable
from typing import List, Optional, Union

import torch.hub
from torch import Tensor, nn
from torchinfo import summary
from ultralytics import YOLO
from ultralytics.nn import DetectionModel
from ultralytics.nn.autobackend import AutoBackend

from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_func

logger = def_logger.getChild(__name__)

old_init = AutoBackend.__init__


def new_init(
    self,
    weights="yolov8n.pt",
    device=torch.device("cpu"),
    dnn=False,
    data=None,
    fp16=False,
    fuse=False,
    verbose=True,
):
    old_init(self, weights, device, dnn, data, fp16, fuse, verbose)


logger.info("patching AutoBackend __init__...")
AutoBackend.__init__ = new_init


def predict(self, x, profile=False, *args, **kwargs):
    return self._predict_once(x, *args, **kwargs)


def _predict_once_frames_stacked(
    self, frames: List[Tensor], *, append_intermediate: bool = False, **kwargs
) -> List[Tensor]:
    """
    If frames were not passed through a compressor, result should be identical as evaluating with _predict_once
    """
    if isinstance(frames, list):
        frames = torch.stack(frames, dim=1)
        B, D, C, H, W = frames.shape
        frames = frames.reshape(B * D, C, H, W)
    x = frames
    y, self.dt = [None for _ in range(self.split_idx)], []  # outputs
    if append_intermediate:
        y[self.split_idx - 1] = x
    layers = self.model if self.pruned else self.model[self.split_idx :]
    for m in layers:
        if m.f != -1:  # if not from previous layer
            x = (
                y[m.f]
                if isinstance(m.f, int)
                else [x if j == -1 else y[j] for j in m.f]
            )  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
    return x


def _predict_once_frames(
    self, frames: List[Tensor], *, append_intermediate: bool = False, **kwargs
) -> List[Tensor]:
    """
    If frames were not passed through a compressor, result should be identical as evaluating with _predict_once
    """
    # x = torch.stack(frames, dim=1).view(B * D, H, C, W)
    layers = self.model if self.pruned else self.model[self.split_idx :]
    res = []
    for x in frames:
        y, self.dt = [None for _ in range(self.split_idx)], []  # outputs
        if append_intermediate:
            y[self.split_idx - 1] = x
        for m in layers:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        res.append(x)
    # x = [f_m.squeeze() for f_m in torch.chunk(x, chunks=D)]
    return res


def _predict_once(
    self,
    x: Tensor | List[Tensor],
    *,
    confidence_filter: Optional[Tensor] = None,
    profile: bool = False,
    visualize: bool = False,
    append_intermediate: bool = False,
    drop_idx: Optional[Tensor] = None,
):
    # todo: just add number of skipped of layers when indexing
    if not isinstance(x, Tensor):
        x = torch.cat(x, dim=0)
    y, self.dt = [None for _ in range(self.split_idx)], []  # outputs
    if append_intermediate:
        y[self.split_idx - 1] = x
    layers = self.model if self.pruned else self.model[self.split_idx :]
    for m in layers:
        if m.f != -1:  # if not from previous layer
            x = (
                y[m.f]
                if isinstance(m.f, int)
                else [x if j == -1 else y[j] for j in m.f]
            )  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            logger.warning(
                "Attempted to visualize patched ultralytics model -> use LogVisualizers"
            )
    if drop_idx is not None:
        # todo: Vectorize
        # for each index zero out predictions
        for idx in drop_idx:
            x[0][idx] = torch.zeros_like(x[idx][0])
            x[1] = [torch.zeros_like(p[idx]) for p in x[1]]
        return torch.zeros_like(x)
    return x


class StackFrames(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: List[Tensor] | Tensor) -> Tensor:
        if isinstance(frames, list):
            frames = torch.stack(frames, dim=1)
            B, D, C, H, W = frames.shape
            frames = frames.reshape(B * D, C, H, W)
        return frames


def patch_ultralytics_model(
    yolo_instances: DetectionModel,
    split_idx: int = 0,
    predict_frames: bool = False,
    stack_frames: bool = False,
) -> DetectionModel:
    yolo_instances.split_idx = split_idx
    if predict_frames:
        yolo_instances._predict_once = types.MethodType(
            _predict_once_frames, yolo_instances
        )
    elif stack_frames:
        yolo_instances._predict_once = types.MethodType(
            _predict_once_frames_stacked, yolo_instances
        )
        yolo_instances.stack_frames = StackFrames()
    else:
        yolo_instances._predict_once = types.MethodType(_predict_once, yolo_instances)
    yolo_instances.predict = types.MethodType(predict, yolo_instances)
    return yolo_instances


@register_model_func
def get_ultralytics_model(
    name: str,
    pretrained: bool = True,
    autoshape: bool = False,
    weights_path: Optional[str] = None,
    split_idx: Optional[int] = None,
    prune_tail: bool = False,
    compile: bool = False,
    predict_frames: bool = False,
    layers_only: bool = False,
    stack_frames: bool = False,
) -> Union[DetectionModel, nn.Sequential]:
    assert not (predict_frames and stack_frames), "Choose stack or predict frames"

    if ".pt" not in name and not weights_path:
        if "8" in name:
            raise NotImplementedError  # load with YOLO class
        name = f"{name}.pt"
        model: DetectionModel = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model=name,
            autoshape=autoshape,
            pretrained=pretrained,
        ).model  # unwrap from Multibackbone Wrapper
    else:
        model = YOLO(name, task="detect").model
    model.pruned = prune_tail
    if split_idx is not None:
        if prune_tail:
            logger.info(
                f"Returning the first {split_idx} prunned layers of ultralytics {name}"
            )
            model.model = model.model[:split_idx]
        else:
            logger.info(
                f"Patching ultralytics model to run with shallow bottleneck injection at index {split_idx}"
            )
            logger.info(
                f"Replacing {summary(model.model[:split_idx], verbose=0, col_names=['num_params']).total_params} params..."
            )
            for idx in range(split_idx):
                model.model[idx] = nn.Identity()
    else:
        split_idx = 0
    model = patch_ultralytics_model(
        model, split_idx, predict_frames=predict_frames, stack_frames=stack_frames
    )
    if compile:
        model = torch.compile(model)
    if layers_only:
        model = model.model
    return model
