import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import kornia
import torch
from lpips import lpips
from pytorch_msssim import MS_SSIM
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from misc.util import separate_keyframes_and_stack_frames
from torchdistill.common.module_util import freeze_module_params
from torchdistill.losses.single import (
    get_single_loss,
    register_org_loss,
    register_single_loss, get_loss,
)


@register_single_loss
class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "vgg", reduction: str = "mean"):
        super().__init__()
        assert reduction in ["sum", "mean"]
        self.l = lpips.LPIPS(eval_mode=True, pretrained=True, pnet_tune=False, net=net)
        freeze_module_params(self.l)
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        l = self.l(inputs, targets)
        if self.reduction == "mean":
            return l.mean()
        else:
            return l.sum()


@register_single_loss
class MS_SSIMLoss(MS_SSIM):
    """
    pytorcH_ssim MS_SSIM loss wrapper

    alpha * MS_SSIM + (1-alpha) L1 Loss
    """

    def __init__(
            self,
            data_range: float = 255,
            size_average: bool = True,
            win_size: int = 11,
            win_sigma: float = 1.5,
            channel: int = 3,
            spatial_dims: int = 2,
            weights: Optional[List[float]] = None,
            K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
            reduction: str = "mean",
    ):
        super().__init__(
            data_range,
            size_average,
            win_size,
            win_sigma,
            channel,
            spatial_dims,
            weights,
            K,
        )
        assert reduction in ["sum", "mean"]
        self.reduction = reduction

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        l = super().forward(X, Y)
        if self.reduction == "mean":
            return l.mean()
        else:
            return l.sum()


@register_single_loss
class MS_SSIM_L1Loss(kornia.losses.MS_SSIMLoss):
    """
    Kornia MS_SSIM loss wrapper

    alpha * MS_SSIM + (1-alpha) L1 Loss
    """

    def __init__(
            self,
            sigmas: list[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
            data_range: float = 1.0,
            # scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get NaNs or negative loss.
            K: tuple[float, float] = (0.01, 0.03),
            alpha: float = 0.025,
            compensation: float = 200.0,
            reduction: str = "mean",
    ):
        super().__init__(
            sigmas=sigmas,
            data_range=data_range,
            K=K,
            reduction=reduction,
            alpha=alpha,
            compensation=compensation,
        )


@register_single_loss
class CharbonnierLossKornia(kornia.losses.CharbonnierLoss):
    """
    Kornia Charbonnier loss wrapper
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)


@register_single_loss
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"]
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        l = torch.sqrt(diff * diff + self.epsilon ** 2)

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()
        return l


# Example usage
if __name__ == "__main__":
    criterion = CharbonnierLoss()
    predictions = torch.randn(5, 3, requires_grad=True)
    targets = torch.randn(5, 3)
    loss = criterion(predictions, targets)
    print(loss)


@register_single_loss
class CosineDissimilarityLoss(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super().__init__()
        assert reduction in ["sum", "batchmean", "mean"]
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs, _ = inputs
        targets, _ = targets
        b = inputs.shape[0]
        l = F.cosine_similarity(inputs.reshape(b, -1), targets.reshape(b, -1)) + 1
        if self.reduction == "sum":
            return l.sum()
        elif self.reduction == "batchmean":
            return l.sum() / b
        else:
            return l.mean()


@register_single_loss
class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, box_pred_path, use_bbox_weight: bool = False, *args, **kwargs):
        super().__init__()
        if "pos_weight" in kwargs:
            kwargs["pos_weight"] = torch.tensor(kwargs["pos_weight"])
        self.bce = BCEWithLogitsLoss(*args, **kwargs)
        self.box_pred_path = box_pred_path
        self.use_bbox_weight = use_bbox_weight

    def forward(self, student_io_dict, teacher_io_dict, targets, **kwargs):
        if self.use_bbox_weight:
            return (
                    self.bce(
                        student_io_dict[self.box_pred_path]["output"].squeeze(),
                        targets.clamp(max=1).float(),
                    )
                    * targets.detach().float().mean()
            )
        return self.bce(
            student_io_dict[self.box_pred_path]["output"].squeeze(), targets.float()
        )


@register_single_loss
class CustomBCEWithLogitsLoss2(nn.Module):
    def __init__(
            self,
            box_pred_path,
            use_bbox_weight: bool = False,
            weights: Optional[Tuple[float, float]] = (1, 1),
            reduction="sum",
            *args,
            **kwargs,
    ):
        super().__init__()
        assert (len(weights)) == 2, "use BCEWithLogitsLoss instead"
        self.box_pred_path = box_pred_path
        self.use_bbox_weight = use_bbox_weight
        self.weights = weights
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, targets, **kwargs):
        targets = targets.float()
        if self.use_bbox_weight:
            b_factor = targets.detach().mean().clamp(min=1.0)
            targets = targets.clamp(max=1.0)
        else:
            b_factor = 1
        input = (
            F.sigmoid(student_io_dict[self.box_pred_path]["output"])
            .squeeze()
            .clamp(min=1e-7, max=1 - 1e-7)
        )
        l = -self.weights[1] * targets * torch.log(input) - (
                1 - targets
        ) * self.weights[0] * torch.log(1 - input)
        if self.reduction == "sum":
            return l.sum() * b_factor
        else:
            return l.mean() * b_factor


@register_single_loss
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, box_pred_path, *args, **kwargs):
        super().__init__()
        if "weight" in kwargs:
            kwargs["weight"] = torch.tensor(kwargs["weight"])
        self.ce = CrossEntropyLoss(*args, **kwargs)

        self.box_pred_path = box_pred_path

    def forward(self, student_io_dict, teacher_io_dict, targets, **kwargs):
        return self.ce(student_io_dict[self.box_pred_path]["output"], targets)


@register_single_loss
class VideoRateDistortionLoss(nn.Module):
    """
    Slightly altered from compressai
    """

    def __init__(
            self,
            distortion: Dict[str, Any],
            reduction: str = "mean",
            return_details: bool = False,
            bitdepth: int = 8,
    ):
        distortion["params"]["reduction"] = reduction
        super().__init__()
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.distortion = get_single_loss(distortion)
        self._scaling_functions = lambda x: (2 ** bitdepth - 1) ** 2 * x
        self.return_details = return_details

    def forward(self, output, target, *args, **kwargs):
        # assert isinstance(target, type(output["x_hat"]))
        # assert len(output["x_hat"]) == len(target)
        # assert self._check_tensors_list(target)
        # assert self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W if self.reduction == "mean" else 1

        # Get scaled and raw loss distortions for each frame
        distortion_loss = 0
        distortions_details = dict()
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            # scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)
            distortion_frame = self.distortion(x_hat, x)
            distortion_loss += distortion_frame
            distortions_details[f"frame{i}.distortion"] = distortion_frame
        # aggregate (over batch and frame dimensions).

        # average scaled_distortions accros the frames
        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_details = self.collect_likelihoods_list(
            likelihoods_list, num_pixels, num_frames
        )

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.

        out["distortion_loss"] = distortion_loss / num_frames
        out["bpp_loss"] = bpp_loss
        if self.return_details:
            out["details"] = {"distortion": distortions_details, "bpp": bpp_details}
        return out

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        assert len(x) == len(target), f"len(x)={len(x)} != len(target)={len(target)})"

        nC = x.size(1)
        assert nC == target.size(
            1
        ), "number of channels mismatches while computing distortion"

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.distortion(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def collect_likelihoods_list(likelihoods_list, num_pixels: int, num_frames: int):
        bpp_info_dict = defaultdict(int)
        bpp_loss = 0

        for i, frame_likelihoods in enumerate(likelihoods_list):
            frame_bpp = 0
            for label, likelihoods in frame_likelihoods.items():
                label_bpp = 0
                for field, v in likelihoods.items():
                    bpp = -v.log2().sum() / num_pixels

                    bpp_loss += bpp
                    frame_bpp += bpp
                    label_bpp += bpp

                    bpp_info_dict[f"bpp_loss.{label}"] += bpp
                    bpp_info_dict[f"bpp_loss.{label}.frame={i}.{field}"] = bpp
                bpp_info_dict[f"bpp_loss.{label}.frame={i}"] = label_bpp
            bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp
        bpp_loss /= num_frames
        return bpp_loss, bpp_info_dict

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
                isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst) -> bool:
        return not (
                not isinstance(lst, (tuple, list))
                or len(lst) < 1
                or any(not cls._check_tensor(x) for x in lst)
        )


@register_single_loss
class VideoRateDistortionLoss2(VideoRateDistortionLoss):
    """ """

    def __init__(
            self,
            distortion: Dict[str, Any],
            reduction: str = "mean",
            return_details: bool = False,
            bitdepth: int = 8,
            bpp_lmbda: float = 1.0,
    ):
        distortion["params"]["reduction"] = reduction
        super().__init__(distortion, reduction, return_details, bitdepth)
        # self.mse = nn.MSELoss(reduction=reduction)
        assert reduction in ["mean", "sum"]
        delattr(self, "distortion")
        self.distortion = BatchedFramesDistortionLoss(loss_params=distortion)
        self.bpp_lmbda = bpp_lmbda

    def forward(
            self,
            outputs: Dict[
                str, Union[List[Dict[str, Dict[str, Tensor]]], List[Tensor], Tensor]
            ],
            targets: List[Tensor],
            *args,
            **kwargs,
    ):
        # assert isinstance(target, type(output["x_hat"]))
        # assert len(output["x_hat"]) == len(target)
        # assert self._check_tensors_list(target)
        # assert self._check_tensors_list(output["x_hat"])

        if isinstance(targets, Tensor):
            targets = targets.chunk(len(outputs["x_hat"]), dim=0)
        _, _, H, W = targets[0].size()
        num_frames = len(targets)
        out = {}
        num_pixels = H * W if self.reduction == "mean" else 1

        keyframe_idx = outputs["keyframe_indexes"]

        distortion_loss = self.distortion(keyframe_idx, outputs["x_hat"], targets)

        assert isinstance(outputs["likelihoods"], list)
        likelihoods_list = outputs.pop("likelihoods")

        bpp_loss, bpp_details = self.collect_likelihoods_list(
            likelihoods_list, num_pixels, num_frames
        )
        out["distortion_loss"] = distortion_loss / num_frames
        out["bpp_loss"] = bpp_loss * self.bpp_lmbda
        return out


@register_single_loss
class BppLoss(nn.Module):
    """ """

    def __init__(
            self,
            entropy_module_path: str,
            reduction: str = "mean",
    ):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction

    def forward(self, model_io_dict, *args, **kwargs):
        entropy_module_dict = model_io_dict[self.entropy_module_path]
        _, likelihoods = entropy_module_dict["output"]
        n, _, h, w = likelihoods.shape
        if self.reduction == "mean":
            # note: assuming latent spatial dim == input spatial dim of split layer
            bpp = -likelihoods.log2().sum() / (n * h * w)
        elif self.reduction == "sum":
            bpp = -likelihoods.log2().sum()
        elif self.reduction == "batchmean":
            bpp = -likelihoods.log2().sum() / n
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp


@register_single_loss
class IdxTargetLossWrapper(nn.Module):
    def __init__(self, unwrap_idx, loss_name, loss_params):
        super().__init__()
        self.unwrap_idx = unwrap_idx
        self.loss = get_loss(loss_name, **loss_params)

    def forward(self, input: Tensor, target: Tuple) -> Tensor:
        return self.loss(input, target[self.unwrap_idx])


class BatchedFramesDistortionLoss(nn.Module):
    """
    Note: keyframe_idx=0 as convention for OUTPUTS (Still need to extract keyframe from targets)
    """

    def __init__(self, loss_params: Dict[str, Any]):
        super().__init__()
        self.distortion = get_single_loss(loss_params)

    def forward(
            self, keyframe_idx: Tensor, outputs: List[Tensor], targets: List[Tensor]
    ):
        targets_keyframes, targets_other_frames = separate_keyframes_and_stack_frames(
            keyframe_idx, targets
        )
        targets_other_frames = [
            t.squeeze(dim=1)
            for t in targets_other_frames.chunk(dim=1, chunks=len(targets))
        ]
        d = self.distortion(outputs[0], targets_keyframes)
        for output, target in zip(outputs[1:], targets_other_frames):
            d += self.distortion(output, target)

        return d


@register_single_loss
class SeqLossWrapper(nn.Module):
    def __init__(self, loss_name: str, loss_params: Dict[str, Any]):
        super().__init__()
        self.loss = get_loss(loss_name, **loss_params)

    def forward(self, outputs: List[Tensor], targets: List[Tensor]):
        l = 0
        for output, target in zip(outputs, targets):
            l += self.loss(output, target)
        return l


@register_single_loss
class SeqMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
            self, inputs: List[Tensor], targets: List[Tensor], *args, **kwargs
    ) -> Tensor:
        if isinstance(inputs, (tuple, list)):
            inputs = torch.stack(inputs, dim=1)

        l = F.mse_loss(
            input=inputs, target=torch.stack(targets, dim=1), reduction=self.reduction
        )
        return l


@register_single_loss
class SeqMSELoss2(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
            self, inputs: List[Tensor], targets: List[Tensor], *args, **kwargs
    ) -> Tensor:
        l = 0
        for i, t in zip(inputs, targets):
            l += F.mse_loss(input=i, target=t, reduction=self.reduction)
        return l


@register_single_loss
class RDLoss(nn.Module):
    def __init__(
            self,
            rate_loss_config: Dict[str, Any],
            distortion_loss_config: Dict[str, Any],
    ):
        super().__init__()
        self.rate_lmbda = rate_loss_config["params"].get("lmbda", 1.0)
        self.distortion_lmbda = distortion_loss_config["params"].get("lmbda", 1.0)
        self.rate_type = rate_loss_config["type"]
        self.distortion_type = distortion_loss_config["type"]
        self.rate_loss = get_single_loss(rate_loss_config)
        self.distortion_loss = get_single_loss(distortion_loss_config)

    def forward(self, *args, **kwargs) -> Tensor:
        rate = self.rate_loss(inputs, targets) * self.rate_lmbda
        distortion = self.distortion_loss(inputs, targets) * self.distortion_lmbda
        return {
            f" {self.rate_lmbda} * {self.rate_type}": rate,
            f" {self.distortion_lmbda} * {self.distortion_type}": distortion,
        }


@register_single_loss
class ScaledMSE(nn.MSELoss):
    def __init__(
            self,
            lmbda: float,
            reduction="mean",
            # assuming 8 bit images
            bitdepth: int = 8,
            alpha: float = 1.0,
    ):
        super().__init__(reduction=reduction)
        self.lmbda = lmbda
        self.power = (2 ** bitdepth - 1) ** 2
        self.alpha = alpha

    def forward(self, inputs, targets, *args, **kwargs) -> Tensor:
        return super().forward(input=inputs, target=targets) * self.power * self.lmbda


@register_single_loss
class WrappedScaledMSE(ScaledMSE):
    def __init__(
            self,
            lmbda: float,
            reduction="mean",
            # assuming 8 bit images
            bitdepth: int = 8,
            alpha: float = 1.0,
            output_path: str = "x_hat",
    ):
        super().__init__(lmbda, reduction, bitdepth, alpha)
        self.output_path = output_path

    def forward(self, inputs, targets, *args, **kwargs) -> Tensor:
        return (
                super().forward(inputs=inputs[self.output_path], targets=targets)
                * self.power
                * self.lmbda
        )


@register_single_loss
class BaselineRDLoss(nn.Module):
    def __init__(
            self,
            lmbda: float,
            input_dims: Tuple[int, int],
            reduction="mean",
            max_intensity: int = 255,
            alpha: float = 1.0,
    ):
        super().__init__()
        self.rate = BppLoss(input_dims=input_dims)
        self.distortion = ScaledMSE(reduction=reduction, bitdepth=max_intensity)
        self.lmbda = lmbda
        self.alpha = alpha

    def forward(self, outputs, targets, **kwargs) -> Tensor:
        rate = self.rate(outputs["likelihoods"])
        distortion = self.distortion(outputs["x_hat"], targets) * self.lmbda
        return rate + distortion


@register_org_loss
class NaiveDistortionKDLoss(nn.Module):
    # I don't think temperature makes sense here
    def __init__(
            self,
            alpha: float,
            beta: Optional[float] = None,
            reduction="sum",
            max_intensity: int = 255,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = (
            1 - alpha if beta is None else beta
        )  # when we, for some reason, want a, b > 1
        self.distortion = ScaledMSE(reduction=reduction, bitdepth=max_intensity)

    def forward(
            self,
            student_io_dict: Mapping[str, Any],
            teacher_io_dict: Mapping[str, Any],
            *args,
            **kwargs,
    ) -> Tensor:
        # reminder: io is passed as I want it to
        return torch.tensor(0)  # TODO


@register_single_loss
class TilesRateDistortionLoss(nn.Module):
    """ """

    def __init__(
            self,
            reduction: str = "mean",
            return_details: bool = False,
            bitdepth: int = 8,
            bpp_lmbda: float = 1.0,
    ):
        # super().__init__(distortion, reduction, return_details, bitdepth)
        super().__init__()
        # self.mse = nn.MSELoss(reduction=reduction)
        assert reduction in ["mean", "sum"]
        self.distortion = SeqMSELoss2(reduction=reduction)
        self.bpp_lmbda = bpp_lmbda
        self.reduction = reduction

    @staticmethod
    def collect_likelihoods_list(likelihoods_list, num_pixels: int, num_tiles: int):
        bpp_info_dict = defaultdict(int)
        bpp_loss = 0
        for tile_no, likelihoods_dict in enumerate(likelihoods_list):
            tile_bpp = 0
            for label, likelihoods in likelihoods_dict.items():
                label_bpp = 0
                bpp = -likelihoods.log2().sum() / num_pixels

                bpp_loss += bpp
                tile_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss:{label}"] += bpp
                bpp_info_dict[f"bpp_loss:tile={tile_no}:{label}"] = label_bpp
            bpp_info_dict[f"bpp_loss:{tile_no}"] = tile_bpp
        bpp_loss /= num_tiles
        return bpp_loss, bpp_info_dict

    def forward(
            self,
            outputs: Dict[
                str, Union[List[Dict[str, Dict[str, Tensor]]], List[Tensor], Tensor]
            ],
            targets: List[Tensor],
            *args,
            **kwargs,
    ):
        # assert isinstance(target, type(output["x_hat"]))
        # assert len(output["x_hat"]) == len(target)
        # assert self._check_tensors_list(target)
        # assert self._check_tensors_list(output["x_hat"])

        if isinstance(targets, Tensor):
            targets = targets.chunk(len(outputs["x_hat"]), dim=0)
        _, _, H, W = targets[0].size()
        num_tiles = len(targets)
        out = {}
        num_pixels = H * W if self.reduction == "mean" else 1

        distortion_loss = self.distortion(outputs["x_hat"], targets)

        assert isinstance(outputs["likelihoods"], list)
        likelihoods_list = outputs.pop("likelihoods")

        bpp_loss, bpp_details = self.collect_likelihoods_list(
            likelihoods_list, num_pixels, num_tiles
        )
        out["distortion_loss"] = distortion_loss / num_tiles
        out["bpp_loss"] = (
                bpp_loss * self.bpp_lmbda
        )  # already divided by / num_tiles in collect_likelihoods_list
        return out


@register_single_loss
class EdgeLoss(nn.Module):
    """
        Multi-Stage Progressive Image Restoration by Zamir et al.
    """

    def __init__(self,
                 reduction: str):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.soe = CharbonnierLoss(reduction=reduction)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel.to(img.device), groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x: Tensor, y: Tensor):
        loss = self.soe(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
