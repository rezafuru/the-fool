import numpy as np
from torch import nn

from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size

logger = def_logger.getChild(__name__)


class RateAnalyzer(nn.Module):
    UNIT_DICT = {"B": 1, "KB": 1024, "MB": 1024 * 1024}

    def __init__(self, unit="KB", **kwargs):
        # should be byte for BPGModule
        super().__init__()
        self.unit = unit
        self.unit_size = self.UNIT_DICT[unit]
        self.kwargs = kwargs
        self.file_size_list = list()
        self.bpp_list = list()

    def update(self, compressed_obj, img_shape):
        b, _, h, w = img_shape
        if self.discard_shape:
            compressed_obj = compressed_obj[:-1]
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        file_size_unit = file_size * self.unit_size
        self.file_size_list.append(file_size)
        bpp = (file_size_unit * 8.0) / (b * h * w)
        self.bpp_list.append(bpp)

    def summarize(self):
        file_sizes = np.array(self.file_size_list)
        logger.info(
            "Bottleneck size [{}]: mean {} std {} for {} samples".format(
                self.unit, file_sizes.mean(), file_sizes.std(), len(file_sizes)
            )
        )
        bpp_sizes = np.array(self.bpp_list)
        logger.info(
            "Bpp size: mean {:.4f} std {:.4f} for {} samples".format(
                bpp_sizes.mean(), bpp_sizes.std(), len(bpp_sizes)
            )
        )

    def clear(self):
        self.file_size_list.clear()
        self.bpp_list.clear()
