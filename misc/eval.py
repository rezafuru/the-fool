import copy
import functools
import math
import os.path
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from PIL import Image
from compressai.models import CompressionModel
from compressai.models.video import ScaleSpaceFlow
from compressai.utils.eval_model.__main__ import compute_metrics
from lpips import lpips
from pytorch_msssim import ms_ssim as compute_ms_ssim
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import IterableSimpleNamespace, yaml_load

from log_manager.wandblogger import WandbLogger

# from misc.frames_ultralytics_validator import DetectionValidatorSequenced
from misc.util import (
    concat_frames_h,
    concat_frames_v,
    concat_images_h,
    concat_images_h_caption_metric,
    concat_images_v,
    concat_images_v2,
    separate_keyframes_and_stack_frames, mkdir,
)
from model.network import (
    DiscriminativeModelWithBottleneck,
    DiscriminativeModelWithBottleneckAndNeuralFilter,
    NetworkWithPerceptualRecovery,
    NetworkWithFeatureReconstruction,
    NetworkWithInputCompression,
)
from model.ntc.image.image_base import ModularImageCompressionModel
from model.ntc.recon.recon_model import ReconModel
# from model.ntc.residual.residual_base import ModularResidualImageCompressionModel

from torchdistill.common.constant import def_logger
from torchdistill.common.module_util import freeze_module_params, unfreeze_module_params
from torchdistill.misc.log import MetricLogger

logger = def_logger.getChild(__name__)

_EVAL_METRICS_REGISTRY = dict()

METRIC_VAL_MAP = {
    "psnr": 0,
    "ms-ssim": 1,
    "recon_img": -1,  # always last one in the list
}


def compute_lpips(image1: Tensor, image2: Tensor, lpips_model) -> Tensor:
    """ """

    lpips_metric = lpips_model.to(image1.device)(image1, image2)

    return lpips_metric.mean()


def compute_psnr_pil(img1, img2, max_value=255):
    """ "Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean(
        (np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2
    )
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def compute_psnr(org_imgs, recon_imgs, max_val: int = 255) -> torch.FloatTensor:
    return 20 * math.log10(max_val) - 10 * torch.log10(
        (org_imgs - recon_imgs).pow(2).mean()
    )


def read_image(filepath: Path) -> Image:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return img


@torch.inference_mode()
def compute_one_vs_all_classification_metrics(
    logits: Tensor, labels: Tensor, filter_threshold: Optional[float] = None
):
    """
    Compute accuracy and sensitivity for a particular class.

    Args:
    - logits (torch.Tensor): The raw output from the neural network. Shape: [batch_size, num_classes]
    - true_labels (torch.Tensor): The true labels. Shape: [batch_size]
    - class_index (int): The index of the class for which metrics are to be computed.

    Returns:
    - accuracy (float): Accuracy of the given class against the rest.
    - sensitivity (float): Sensitivity of the given class.
    """
    if filter_threshold:
        predictions = (~(F.sigmoid(logits) > filter_threshold)).int()
    else:
        _, predictions = logits.topk(1, 1, True, True)
    predictions = predictions.t()
    y_pred_binary = (predictions == 0).int()
    y_true_binary = (labels == 0).int()

    # Compute the confusion matrix components
    tn = torch.sum((y_true_binary == 0) & (y_pred_binary == 0)).item()
    fp = torch.sum((y_true_binary == 0) & (y_pred_binary == 1)).item()
    tp = torch.sum((y_true_binary == 1) & (y_pred_binary == 1)).item()
    fn = torch.sum((y_true_binary == 1) & (y_pred_binary == 0)).item()

    # Compute specificity and sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy * 100, sensitivity * 100, specificity * 100


# Example


@torch.inference_mode()
def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])
        result_list = []
        for k in topk:
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            result_list.append(correct_k * (100.0 / batch_size))
        return result_list


def register_eval_metric(_func: Callable = None, *, name: Optional[str] = None):
    def decorator_register(cls):
        @functools.wraps(cls)
        def wrapper_register():
            cls_name = name or cls.__name__
            _EVAL_METRICS_REGISTRY[cls_name] = cls

        wrapper_register()
        return cls

    if _func is None:
        return decorator_register
    else:
        return decorator_register(_func)


class EvaluationMetric(ABC):
    """
    """

    def __init__(
        self,
        eval_args: Optional[Dict[str, Any]] = None,
        viz_on_best_args: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        if eval_args:
            self.eval_func = partial(self.eval_func, **eval_args)
        if viz_on_best_args:
            self.viz_on_best = partial(self.viz_on_best, **viz_on_best_args)

    @abstractmethod
    def comparator(self, val1: float, val2: float) -> bool:
        raise NotImplementedError

    # Let's throw any understanding of OOP out of the window
    @staticmethod
    @abstractmethod
    def eval_func(
        model: nn.Module,
        data_loader: DataLoader,
        device: str,
        title: Optional[str],
        header: str = "Validation:",
        *args,
        **kwargs,
    ) -> Union[float, Dict[str, float]]:
        raise NotImplementedError

    def reset(self):
        self.val = self.init_best_val

    @property
    def best_val(self) -> float:
        return self.best_val

    @best_val.setter
    def best_val(self, val: float):
        self._best_val = val

    @property
    @abstractmethod
    def init_best_val(self) -> float:
        raise NotImplementedError

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    def viz_on_best(self, *args, **kwargs):
        pass


@register_eval_metric(name="map")
class UltralyticsmAPEvaluation(EvaluationMetric):
    def __init__(self, eval_args: Optional[Dict[str, Any]] = None):
        super().__init__(eval_args)
        self.init_best_val = 0

    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return 0

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    # for some f*ing reason, when using inference_mode and using cuda:1, it throws an exception during first train pass
    # @torch.no_grad()
    @torch.inference_mode()
    def eval_func(
        model: DiscriminativeModelWithBottleneck,
        # Attention: Dataset config path must be nested in the validator config
        ultralytics_default_overrides: Dict[str, Any],
        device: str,
        title: Optional[str] = None,
        wdb_logger: Optional[WandbLogger] = None,
        use_kp_weights: bool = False,
        log_detection_viz_remotely: bool = False,
        empty_cache: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        # model = copy.deepcopy(model)

        if use_kp_weights:
            model.compression_module.use_kp_weights = use_kp_weights
        if title is not None:
            logger.info(title)
        validator_args = IterableSimpleNamespace(
            **yaml_load("external/ultralytics_yolo/cfg/default.yaml")
        ).__dict__
        eval_data = ultralytics_default_overrides["data"]
        model = model.to(device)
        freeze_module_params(model)
        ultralytics_default_overrides["device"] = device
        ultralytics_default_overrides["data"] = os.path.expanduser(
            ultralytics_default_overrides["data"]
        )
        ultralytics_default_overrides["verbose"] = False

        model.eval()
        if title is not None:
            logger.info(title)

        validator_args.update(**ultralytics_default_overrides)
        # if hasattr(model, 'compression_module') and isinstance(model.compression_module, ModularVideoLikeCompressionModel):
        #     validator = DetectionValidatorSequenced(
        #         args=IterableSimpleNamespace(**validator_args)
        #     )
        # else:
        #     validator = DetectionValidator(
        #         args=IterableSimpleNamespace(**validator_args)
        #     )
        validator = DetectionValidator(args=IterableSimpleNamespace(**validator_args))
        stats = validator(model=model)

        def _log_info(*l_args, **l_kwargs):
            logger.info(*l_args, **l_kwargs)

        # monkeh patch print to log summary printed by ultralytics
        # builtin_print = __builtin__.print
        # __builtin__.print = _log_info
        # __builtin__.print = builtin_print

        # we don't care about the inference time of the detector.
        #  We only care about the execution time of our compression model
        if log_detection_viz_remotely:
            stats["viz"] = wandb.Image(
                Image.open("resources/eval-ultralytics-yolo/val_batch0_labels.jpg")
            )
        if use_kp_weights:
            model.compression_module.use_kp_weights = False
        # del model

        if hasattr(model, "compression_module"):
            unfreeze_module_params(model.compression_module)
            if hasattr(model.compression_module, "feature_extractor"):
                freeze_module_params(model.compression_module.feature_extractor)
        if empty_cache:
            torch.cuda.empty_cache()
        return stats


@register_eval_metric(name="recon-feature")
class FeatureToImageReconEvaluation(EvaluationMetric):
    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return float("-inf")

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    @torch.inference_mode()
    def eval_func(
        model: ReconModel,
        data_loader: Union[DataLoader, Dict[str, DataLoader]],
        device: str,
        log_freq=1000,
        title: Optional[str] = None,
        header: str = "Validation:",
        max_val: float = 1.0,
        **kwargs,
    ):
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        lpips_model = lpips.LPIPS(net="vgg", verbose=False, pretrained=True, eval_mode=True)
        freeze_module_params(lpips_model)
        if isinstance(data_loader, dict):
            data_loader = next(iter(data_loader.values()))  # todo: multi dataset
        for features, targets in metric_logger.log_every(data_loader, log_freq, header):
            features = [f.to(device, non_blocking=True) for f in features]
            targets = [t.to(device, non_blocking=True) for t in targets]
            output = model(features)
            output = torch.stack(output, dim=1)
            B, D, C, H, W = output.shape
            output = output.reshape(B * D, C, H, W)
            output = torch.clamp(output, min=0., max=1.)
            targets = torch.stack(targets, dim=1).reshape(B * D, C, H, W)
            psnr = compute_psnr(org_imgs=targets, recon_imgs=output, max_val=max_val)
            ssim = compute_ms_ssim(targets, output, data_range=max_val)

            l_pips = compute_lpips(targets, output, lpips_model=lpips_model)
            metric_logger.meters["psnr"].update(psnr.item())
            metric_logger.meters["ssim"].update(ssim.item())
            metric_logger.meters["lpips"].update(l_pips.item())
        del lpips_model
        psnr = metric_logger.psnr.global_avg
        ms_ssim = metric_logger.ssim.global_avg
        l_pips = metric_logger.lpips.global_avg
        # Convert to dB scale for better interpretability between models
        ms_ssim_db = -10 * math.log10(1 - ms_ssim)
        logger.info(" * PSNR {:.4f}".format(psnr))
        logger.info(" * SSIM (db) {:.4f}".format(ms_ssim_db))
        logger.info(" * LPIPS {:.4f}".format(l_pips))
        return {"psnr": psnr, "ssim": ms_ssim_db, "lpips": l_pips}

    @torch.inference_mode()
    def viz_on_best(
        self,
        model: ReconModel,
        device: str,
        wdb_logger: WandbLogger,
        viz_frames_root: str,
        caption_metric: Optional[str] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        epoch: Optional[int] = None,
        output_viz_root: Optional[Path] = None,
        file_prefix: Optional[str] = None,
        max_val: float = 1.0,
        downsample_factor_result: Optional[int] = None,
        **kwargs,
    ):
        """
        Root folder, list of groups
        1 row is 1 group
        """
        if viz_frames_root is None:
            return
        columns = [
            "psnr (per frame)",
            "ms-ssim (per frame)",
            "imgs",
        ]
        if epoch is not None:
            columns = ["epoch"] + columns
        model.eval()
        model = model.to(device)
        logger.info(
            f"Evaluating metrics and visualizing reconstruction for frames in {viz_frames_root}"
        )
        to_tensor = transforms.ToTensor()
        to_img = transforms.ToPILImage()
        to_size = (
            transforms.Resize(resize_to, antialias=True) if resize_to else lambda x: x
        )


        metric_logger = MetricLogger(delimiter="  ")
        assert os.path.isdir(viz_frames_root), "Image folder for eval not found"
        inputs_root = Path(viz_frames_root)
        result_list = list()
        recon_rows = []
        orig_rows = []
        for frames_paths in inputs_root.iterdir():
            frames_path = sorted(f for f in frames_paths.iterdir() if f.is_file())
            frames_orig = [
                np.asarray(to_size(Image.open(p).convert("RGB"))) for p in frames_path
            ]
            frame_list = [
                t.unsqueeze(dim=0).to(device)
                for t in torch.chunk(
                    to_tensor(
                        np.concatenate(
                            frames_orig,
                            axis=-1,
                        )
                    ),
                    len(frames_orig),
                )
            ]
            D = len(frames_orig)
            B, C, H, W = frame_list[0].shape
            recons = model(frame_list)
            # recons_tensor = torch.stack(recons, dim=1).reshape(B * D, C, H, W)
            # frames_tensor = torch.stack(frame_list, dim=1).reshape(B* D, C, H, W)
            # psnr = compute_psnr(
            #     org_imgs=frames_tensor, recon_imgs=recons_tensor, max_val=max_val
            # )
            # ssim = compute_ms_ssim(frames_tensor, recons_tensor, data_range=max_val)
            # l_pips = compute_lpips(recons_tensor, recons_tensor)
            recon_rows.append(
                concat_images_h(frames=[to_img(torch.clamp(f, min=0, max=1.).squeeze(dim=0)) for f in recons])
            )
            orig_rows.append(
                concat_images_h(frames=[to_img(f.squeeze(dim=0)) for f in frame_list])
            )

        full_recon_img = concat_images_v2(recon_rows)
        full_orig_img = concat_images_v2(orig_rows)
        comparison = concat_images_h([full_recon_img, full_orig_img], margin=10)
        if downsample_factor_result is not None:
            to_size_result = transforms.Resize(size=(comparison.height // downsample_factor_result, comparison.width // downsample_factor_result),
                              antialias=True,
                              interpolation=InterpolationMode.BICUBIC)
            comparison = to_size_result(comparison)
        wdb_logger.log_img(key="Recon Comparisons", img=comparison, commit=False)
        # wdb_logger.log_table(
        #     columns=columns,
        #     data=result_list,
        #     key=f"Recon Qualitative {'validation' if epoch is not None else 'test'}",
        # )
        if output_viz_root is not None:
            mkdir(output_viz_root)
            file_name = file_prefix if file_prefix is not None else "recon_vs_orig"
            if epoch is not None:
                file_name = f"{file_name}_epoch={epoch}.png"
            else:
                file_name = f"{file_name}.png"
            file_path = Path(output_viz_root) / file_name
            logger.info(f"Saving visualization locally to {file_path}")
            comparison.save(file_path, 'PNG', compress_level=9)


@register_eval_metric(name="recon")
class ReconEvaluation(EvaluationMetric):
    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return float("-inf")

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    def eval_func(
        model: Union[DiscriminativeModelWithBottleneck, CompressionModel],
        data_loader: Union[DataLoader, Dict[str, DataLoader]],
        device: str,
        log_freq=1000,
        title: Optional[str] = None,
        header: str = "Validation:",
        **kwargs,
    ):
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        with torch.no_grad():
            for image, _ in metric_logger.log_every(data_loader, log_freq, header):
                image = image.to(device, non_blocking=True)
                recon = model(image)
                if isinstance(recon, dict):
                    recon = recon["x_hat"]
                recon = torch.clamp(recon, min=0, max=1)
                # we don't convert to pil image
                psnr = compute_psnr(org_imgs=image, recon_imgs=recon, max_val=1)
                # Convert to dB scale for readability
                ms_ssim = compute_ms_ssim(image, recon, data_range=1)
                batch_size = image.shape[0]
                metric_logger.meters["psnr"].update(psnr.item(), n=batch_size)
                metric_logger.meters["ms_ssim"].update(ms_ssim.item(), n=batch_size)

            psnr = metric_logger.psnr.global_avg
            ms_ssim = metric_logger.ms_ssim.global_avg
            ms_ssim_db = -10 * math.log10(1 - ms_ssim)
            logger.info(" * PSNR {:.4f}".format(psnr))
            logger.info(" * MS-SSIM (db) {:.4f}".format(ms_ssim_db))
        return {"psnr": metric_logger.psnr.global_avg, "ms-ssim": ms_ssim_db}


@register_eval_metric(name="recon-quali")
class QualitativeReconEvaluation(EvaluationMetric):
    def __init__(self, eval_args: Optional[Dict[str, Any]] = None):
        super().__init__(eval_args)
        self.init_best_val = 0

    @staticmethod
    @torch.no_grad()
    def inference_recon(
        model: Union[
            NetworkWithFeatureReconstruction, NetworkWithPerceptualRecovery
        ],
        x: Tensor,
        max_val=255,
    ) -> Tuple[List[Union[float, wandb.Image]], Tensor]:
        """ """
        x = x.unsqueeze(0)
        out_net = torch.clamp(model.forward(x), min=0.0, max=1.0)
        # assume input images as 8bit RGB
        metrics = compute_metrics(x, out_net, max_val=max_val)
        result = [
            metrics[f"psnr"],
            metrics[f"ms-ssim"],
        ]
        return result, out_net

    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return float("-inf")

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    @torch.no_grad()
    def eval_func(
        viz_imgs_root: str,
        model: Union[DiscriminativeModelWithBottleneck, CompressionModel],
        device: str,
        wdb_logger: Optional[WandbLogger] = None,
        output_viz_root: Optional[Path] = None,
        caption_metric: str = "psnr",
        include_summary: bool = False,
        add_caption: bool = False,
        epoch: Optional[int] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        # *args,
        **kwargs: Any,
    ) -> Union[List[float], Tuple[List[float], Dict[str, float]]]:
        model.eval()
        logger.info(
            f"Evaluating metrics and visualizing reconstruction for {viz_imgs_root}"
        )
        to_tensor = transforms.ToTensor()
        to_img = transforms.ToPILImage()
        to_size = (
            transforms.Resize(resize_to, antialias=True) if resize_to else lambda x: x
        )

        assert os.path.isdir(viz_imgs_root), "Image folder for eval not found"
        input_root = Path(viz_imgs_root)
        result_list = list()
        # if not entropy_estimation and hasattr(model, 'context_model'):
        #     logger.info("Detected context model.. will use cpu for compression evaluation")
        #     device = 'cpu'
        #     model = model.to(device)
        # else:
        #     device = next(model.parameters()).device
        for img_file in tqdm(os.listdir(input_root)):
            img_file_path = input_root / img_file
            orig_img = to_size(read_image(img_file_path))
            x = to_tensor(orig_img).to(device)
            img_metrics, recon_img = QualitativeReconEvaluation.inference_recon(
                model=model, x=x
            )
            orig_vs_recon_img = concat_images_h_caption_metric(
                img_a=orig_img,
                img_b=(
                    to_img(recon_img.squeeze()),
                    img_metrics[METRIC_VAL_MAP[caption_metric]],
                ),
                metric=caption_metric if add_caption else None,
            )
            img_metrics.append(wandb.Image(orig_vs_recon_img))
            if output_viz_root:
                orig_vs_recon_img.save(output_viz_root / img_file)

            result_list.append(img_metrics)
        wdb_logger.log_table(
            columns=[
                "psnr",
                "ms-ssim",
                "imgs",
            ],
            data=result_list,
            key=f"Recon Qualitative {'validated' if epoch is not None else 'test'}",
        )
        return dict()
        #     return result_list, {
        #         "epoch": epoch if epoch is not None else "test",
        #         "psnr": avg_psnr,
        #         "bpp": avg_bpp,
        #         "ms-ssim": avg_ssim,
        #         "ms-ssim (dB)": -10 * math.log10(1 - avg_ssim),
        #     }
        # return result_list


@register_eval_metric(name="bpp")
class BppEvaluation(EvaluationMetric):
    def __init__(self, eval_args: Optional[Dict[str, Any]] = None):
        super().__init__(eval_args)
        self.init_best_val = float("inf")

    def comparator(self, val1: float, val2: float) -> bool:
        return val1 < val2

    @property
    def init_best_val(self) -> float:
        return 0

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    @torch.inference_mode()
    def _eval_inference(
        model: CompressionModel, images: Tensor, metric_logger: MetricLogger
    ):
        if isinstance(images, list):
            b, _, h, w = images[0].size()
            b *= len(images)
        else:
            b, _, h, w = images.size()
        num_pixels = b * h * w
        bpp = (
            sum(len(s[0]) for s in model.compress(images)["strings"]) * 8.0 / num_pixels
        )
        metric_logger.meters["bpp"].update(bpp, n=b)

    @staticmethod
    @torch.no_grad()
    def _eval_estimation(
        model: CompressionModel, images: Tensor, metric_logger: MetricLogger
    ):
        if isinstance(images, list):
            b, _, h, w = images[0].size()
            b *= len(images)
        else:
            b, _, h, w = images.size()
        num_pixels = b * h * w
        bpp = 0
        if isinstance(model, ModularImageCompressionModel):
            likelihoods_latents = model.forward(images, return_likelihoods=True)[
                "likelihoods"
            ]
        else:
            likelihoods_latents = model(images)["likelihoods"]
        for latent, likelihoods in likelihoods_latents.items():
            bpp_latent = -likelihoods.log2().sum() / num_pixels
            metric_logger.meters[f"bpp_{latent}"].update(bpp_latent.item(), n=b)
            bpp += bpp_latent
        metric_logger.meters["bpp"].update(bpp, n=b)

    @staticmethod
    @torch.inference_mode()
    def _eval_pillow_codec(
        model: NetworkWithInputCompression, images: List, metric_logger
    ):
        _, bpp = model(images, return_avg_bpp=True, skip_prediction=True)
        metric_logger.meters["bpp"].update(bpp, n=len(images))

    @staticmethod
    @torch.no_grad()
    def eval_func(
        model: Union[DiscriminativeModelWithBottleneck, CompressionModel],
        data_loader: Union[DataLoader, Dict[str, DataLoader]],
        device: str,
        log_freq=1000,
        title: Optional[str] = None,
        header: str = "Validation:",
        test_mode: bool = False,
        **kwargs,
    ) -> float:
        if title is not None:
            logger.info(title)

        model = model.to(device)
        model.eval()
        if test_mode:
            assert (
                model.compressor_updated
            ), "Compressor must be updated to evaluate test time file sizes"

        if not isinstance(data_loader, dict):
            data_loader = {"": data_loader}

        # for evaluating bbp of compressai models
        if isinstance(model, CompressionModel) or isinstance(
            model, NetworkWithInputCompression
        ):
            compressor = model
        else:
            compressor = model.compression_module


        if isinstance(model, NetworkWithInputCompression):
            eval_fun = BppEvaluation._eval_pillow_codec
        # elif isinstance(compressor, FeatureParallelizedScaleSpaceFlow):
        #     eval_fun = VideoRDEvaluation._eval_rate_estimation
        elif test_mode:
            eval_fun = BppEvaluation._eval_inference
        else:
            eval_fun = BppEvaluation._eval_estimation
        metric_logger = MetricLogger(delimiter="  ")

        result_dict = dict()
        for key, data_loader in data_loader.items():
            prefix = f"{Path(key).stem}" if key != "" else key
            for images, _ in metric_logger.log_every(data_loader, log_freq, f"{header} {prefix}"):
                if isinstance(images, Tensor):
                    images = images.to(device, non_blocking=True)
                elif isinstance(images, list):
                    images = [img.to(device, non_blocking=True) for img in images]
                eval_fun(compressor, images, metric_logger)
            metric_logger.synchronize_between_processes()

            for attr in metric_logger.meters.keys():
                if "bpp" in attr:
                    res = metric_logger.meters[attr].global_avg
                    logger.info(f" * {attr} {res:.5f}\n")
            result_dict[f"{prefix}_bpp"] = metric_logger.meters["bpp"].global_avg
        return result_dict




@register_eval_metric(name="accuracy")
class AccuracyEvaluation(EvaluationMetric):
    def __init__(self, eval_args: Optional[Dict[str, Any]] = None):
        super().__init__(eval_args)
        self.init_best_val = 0

    def comparator(self, val1: float, val2: float) -> bool:
        return val1 > val2

    @property
    def init_best_val(self) -> float:
        return 0

    @init_best_val.setter
    def init_best_val(self, val: float):
        self._init_best_val = val

    @staticmethod
    @torch.inference_mode()
    def eval_func(
        model: nn.Module,
        data_loader: DataLoader,
        device: str,
        log_freq=1000,
        title: Optional[str] = None,
        header: str = "Validation:",
        include_ce: Optional[bool] = False,
        # from_submodule: Optional[str] = None,
        from_filter: bool = False,
        **kwargs,
    ) -> Union[float, Dict[str, float]]:
        model.eval()
        # if from_filter:
        #     if isinstance(model.compression_module, NeuralFilterWrapper):
        #         model = partial(model.compression_module, return_filter_logits=True)
        #     else:
        #         model = partial(model.compression_module, return_likelihoods=True)

        # if from_submodule:
        #     model = getattr(model, from_submodule)

        if not isinstance(data_loader, dict):
            data_loader = {"": data_loader}

        metric_logger = MetricLogger(delimiter="  ")
        result_dict = dict()
        for key, data_loader in data_loader.items():
            prefix = f"{Path(key).stem}" if key != "" else key
            for image, target in metric_logger.log_every(data_loader, log_freq, header):
                batch_size = image.shape[0]
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                if isinstance(output, tuple):
                    output = output[1]
                else:
                    output = output["filter_logits"]
                acc1 = compute_accuracy(output, target, topk=(1,))
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.synchronize_between_processes()
            result_dict[f"{prefix}_acc@1"] = metric_logger.meters["acc1"].global_avg
        top1_accuracy = metric_logger.acc1.global_avg
        # top5_accuracy = metric_logger.acc5.global_avg
        logger.info(" * Acc@1 {:.4f}\n".format(top1_accuracy))
        return result_dict




# todo: Support for multiple metrics in Mapping
def get_eval_metric(metric_name: str, **kwargs) -> EvaluationMetric:
    if metric_name not in _EVAL_METRICS_REGISTRY:
        raise ValueError(f"Evaluation metric `{metric_name}` not registered")
    return _EVAL_METRICS_REGISTRY[metric_name](**kwargs)


def get_eval_metrics(
    metric_configs: List[Dict[str, Any]]
) -> Dict[str, EvaluationMetric]:
    eval_metrics = dict()
    for metric in metric_configs:
        metric_name = metric["name"]
        eval_metrics[metric_name] = get_eval_metric(metric_name, **metric["params"])
    return eval_metrics
