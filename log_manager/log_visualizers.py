import abc
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import wandb
from PIL import Image
from torchvision import transforms
from collections.abc import Iterable, Container

from misc.util import concat_images_h, show_att_map_on_image

LOG_VISUALIZER_DICT = dict()


def register_log_visualizer(cls: Callable) -> Callable:
    LOG_VISUALIZER_DICT[cls.__name__] = cls
    return cls


class LogVisualizer(abc.ABC):
    """
    ATM Its a bit crappy and we select the first image in the batch,
        which is basically random since we shuffle the input
    """

    def __init__(self):
        self.to_img = transforms.ToPILImage()

    @abc.abstractmethod
    def create_viz_img(
        self, io_dict: Dict[str, Dict[str, Tensor]], *args, **kwargs
    ) -> Image.Image:
        raise NotImplemented()


@register_log_visualizer
class OrigVsReconLogVisualizer(LogVisualizer):
    """
    Compare original and reconstructed input
    """

    def __init__(
        self,
        path_io_orig: Tuple[str, str],
        path_io_recon: Tuple[str, str],
        from_models: Union[Tuple[str, str]],
    ):
        """
        path_io are the path and io to in the passed io_dict (e.g., 'network.g_s', 'output')

        from_models is a tuple if orig is from another model (e.g., teacher)
        """
        super().__init__()
        self.orig_module_path, self.orig_module_io = path_io_orig
        self.recon_module_path, self.recon_module_io = path_io_recon
        if isinstance(from_models, str):
            self.from_model_orig = self.from_model_recon = from_models
        else:
            self.from_model_orig, self.from_model_recon = from_models

    def create_viz_img(
        self, io_dict: Dict[str, Dict[str, Tensor]], *args, **kwargs
    ) -> Image.Image:
        # todo move caption to WandbLogger
        return concat_images_h(
            self.to_img(
                io_dict[self.from_model_orig][self.orig_module_path][
                    self.orig_module_io
                ][0]
                .detach()
                .squeeze()
            ),
            self.to_img(
                io_dict[self.from_model_recon][self.recon_module_path][
                    self.recon_module_io
                ][0]
                .detach()
                .squeeze()
            ),
        )


@register_log_visualizer
class AttentionMapsVisualizer(LogVisualizer):
    """
    Overaly a variable number of attention maps on top of an input image
    """

    def __init__(
        self,
        path_io_orig: Tuple[str, str],
        paths_io_features: List[Tuple[str, str]],
        att_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.orig_module_path, self.orig_module_io = path_io_orig
        self.path_io_features = paths_io_features
        self.to_pil = transforms.ToPILImage()
        if att_config:
            strategy = att_config["strategy"]
            params = att_config["params"]
            if strategy == "sum_of_absolutes":
                self.comp_att_maps = partial(self._sum_of_absolute_p, **params)
            elif strategy == "max_of_absolutes":
                self.comp_att_maps = partial(self._max_of_absolute_p, **params)
            elif strategy == "mean_of_absolutes":
                self.comp_att_maps = partial(self._mean_of_absolute_p, **params)
            else:
                raise ValueError(f"{strategy} not implemented")
        else:
            self.comp_att_maps = nn.Identity()

    @staticmethod
    def _sum_of_absolute_p(features: Tensor, p: int, *args, **kwargs) -> Tensor:
        """
        b x C x H x W  --> b x 1 x H x W
        """
        return torch.pow(torch.abs(features), exponent=p).sum(dim=1, keepdim=True)
        # return F.normalize(torch.pow(torch.abs(features), exponent=p).sum(dim=1).flatten(start_dim=1), p=5.0)

    @staticmethod
    def _mean_of_absolute_p(features: Tensor, p: int, *args, **kwargs) -> Tensor:
        """
        b x C x H x W  --> b x 1 x H x W
        """
        return torch.pow(torch.abs(features), exponent=p).mean(dim=1, keepdim=True)
        # return F.normalize(torch.pow(torch.abs(features), exponent=p).mean(dim=1).flatten(start_dim=1), p=5.0)

    @staticmethod
    def _max_of_absolute_p(features: Tensor, p: int, *args, **kwargs) -> Tensor:
        """
        b x C x H x W  --> b x 1 x H x W
        """
        return torch.pow(
            torch.max(torch.abs(features), dim=1, keepdim=True)[0], exponent=p
        )
        # return F.normalize(torch.pow(torch.max(torch.abs(features), dim=1)[0], exponent=p).flatten(start_dim=1), p=5.0)

    def create_viz_img(
        self, io_dict: Dict[str, Dict[str, Tensor]], *args, **kwargs
    ) -> Image.Image:
        orig_input = io_dict[self.orig_module_path][self.orig_module_io].detach()
        h, w = orig_input.size()[2:]
        orig_img = self.to_pil(orig_input.squeeze())
        rgb_orig_img = np.float32(np.array(orig_img) / 255)
        res = orig_img
        # res = Image.new('RGB', (0, 0))
        for feature_module_path, feature_module_io in self.path_io_features:
            # features = F.normalize(io_dict[feature_module_path][feature_module_io].detach())
            features = io_dict[feature_module_path][feature_module_io].detach()
            f_h, f_w = features.size()[2:]
            att_map = self.comp_att_maps(features=features).reshape(-1, 1, f_h, f_w)
            att_map = F.interpolate(att_map, size=(h, w), mode="bilinear").squeeze()
            masked_img = show_att_map_on_image(rgb_orig_img, att_map, use_rgb=True)
            res = concat_images_h(
                Image.fromarray(masked_img), res
            )  # TODO: Add feature_module_path as description above/below img
        return res


# import torchvision
# from torchvision.models import ResNet101_Weights, resnet101
# m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
# t = transforms.ToTensor()(Image.open('ILSVRC2012_val_00018162.JPEG')).unsqueeze(dim=0)
# t = F.interpolate(t, size=(224, 224))
# m.eval()
# f0, f1, f2, f3, f4 = m(t)
# d = {
#     "orig": {"input": t},
#     "f0": {"output": f0},
#     "f1": {"output": f1},
#     "f2": {"output": f2},
#     "f3": {"output": f3},
#     "f4": {"output": f4},
# }
# v_soa = partial(AttentionMapsVisualizer, ("orig", "input"),
#                             [
#                                 ("f0", "output"),
#                                 ("f1", "output"),
#                                 ("f2", "output"),
#                                 ("f3", "output"),
#                                 ("f4", "output"),
#                              ])
#
# v_soa_p1 = v_soa(att_config={"strategy": "sum_of_absolutes", "params": {"p": 3}})
# v_soa_p2 = v_soa(att_config={"strategy": "mean_of_absolutes", "params": {"p": 3}})
# v_soa_p3 = v_soa(att_config={"strategy": "max_of_absolutes", "params": {"p": 3}})
#
# img_soa_p1 = v_soa_p1.create_viz_img(d)
# img_soa_p2 = v_soa_p2.create_viz_img(d)
# img_soa_p3 = v_soa_p3.create_viz_img(d)
# img_soa_p1.save('img_soa_p1.png')
# img_soa_p2.save('img_soa_p2.png')
# img_soa_p3.save('img_soa_p3.png')
# img_soa_p4.save('img_soa_p4.png')


@register_log_visualizer
class LatentLogVisualizer(LogVisualizer):
    """
    TODO: Which channel(s) to choose?
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        paths_io: Tuple[str, str, str],
        paths_io_comparison: Optional[Tuple[str, str, str]] = (None, None, None),
    ):
        super().__init__()
        self.from_model, self.module_path, self.module_io = paths_io
        (
            self.from_model_comparison,
            self.module_path_comparison,
            self.module_io_comparison,
        ) = paths_io_comparison
        self.resolution = resolution

    def create_viz_img(self, io_dict: Dict[str, Any], *args, **kwargs) -> Image.Image:
        latent = (
            io_dict[self.from_model][self.module_path][self.module_io][0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        img = self.to_img(F.upsample(latent, size=self.resolution))
        if self.from_model_comparison:
            latent_comparison = (
                io_dict[self.from_model_comparison][self.module_path_comparison][
                    self.module_io_comparison
                ][0]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            latent_comparison = self.to_img(
                F.upsample(latent_comparison, size=self.resolution)
            )
            img = concat_images_h(img, latent_comparison)
        return img


@register_log_visualizer
class EntropyMapLogVisualizer(LogVisualizer):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


def get_log_visualizers(
    visualizers_config: List[Mapping[str, Any]]
) -> List[LogVisualizer]:
    visualizers = list()
    missing = []
    for item in visualizers_config:
        [_, vis_name], [_, vis_config] = item["entry"].items()
        if vis_name not in LOG_VISUALIZER_DICT:
            missing.append(vis_name)
        else:
            visualizers.append(LOG_VISUALIZER_DICT[vis_name](**vis_config))
    if missing:
        raise ValueError(f"Could not find visualizers: {','.join(missing)}")
    return visualizers
