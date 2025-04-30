import datetime
import time
from collections import defaultdict
from typing import Any, Mapping, Optional

import torch
import wandb
from torch import Tensor
from PIL import Image

from log_manager.log_visualizers import get_log_visualizers
from log_manager.wandblogger import WandbLogger
from misc.dclasses import WandBConfig
from misc.util import concat_images_v
from torchdistill.common.constant import def_logger
from torchdistill.misc.log import SmoothedValue

py_logger = def_logger.getChild(__name__)


class WandBMetricLogger:
    def __init__(
        self,
        wandb_config: Mapping[str, Any],
        experiment_config: Mapping[str, Any],
        delimiter="\t",
    ):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        wandb_remote_config = wandb_config["remote"]
        wandb_remote_config["resume"] = wandb_config["resume"]
        wandb_remote_config["id"] = wandb_config["id"]
        wandb_remote_config = WandBConfig(**wandb_config["remote"])
        self.wandblogger = WandbLogger(
            wandb_config=wandb_remote_config, init_config=experiment_config
        )
        self.step = 0
        self.scalar_freq = wandb_config["scalar_freq"]
        self.viz_freq = wandb_config["viz_freq"]
        self.visualizers = get_log_visualizers(wandb_config.get("visualizers", []))

    def update(
        self, scalars: Mapping[str, Tensor], io_dict: Optional[Mapping[str, Any]] = None
    ):
        for k, v in scalars.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().item()

            assert isinstance(
                v, (float, int)
            ), f"`{k}` ({v}) should be either float or int"
            self.meters[k].update(v)
        # del wandb_log_dict["aux_loss"]
        # reminder: this is crap and your ancestors would be ashamed
        if (self.step + 1) % self.scalar_freq == 0:
            self.wandblogger.log(log_dict=scalars, step=self.step, prefix="train")
        if io_dict and self.step % self.viz_freq == 0:
            img = Image.new("RGB", (0, 0))
            for log_visualizer in self.visualizers:
                # create single image to log to reduce roundtrips
                img = concat_images_v(img, log_visualizer.create_viz_img(io_dict))
            # self.wandblogger.log_img(key=f"orig_vs_recon_train)", img=img,
            #                          step=self.step,
            #                          commit=True,
            #                          caption="Orig vs Recon")

    def clear(self):
        """ """
        self.step = 0
        del self.meters
        self.meters = defaultdict(SmoothedValue)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def finish(self):
        py_logger.info("Finishing wandb run...")
        wandb.finish(quiet=False)
        time.sleep(1)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq: int = None, header: str = None):
        i = 0
        if not header:
            header = ""

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            self.step += 1
            if i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    py_logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        py_logger.info("{} Total time: {}".format(header, total_time_str))
