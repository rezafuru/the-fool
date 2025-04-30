import os
import time
from collections import defaultdict
from typing import List, Optional

import torch
import wandb
from PIL import Image
from matplotlib.pyplot import clf

from misc.dclasses import WandBConfig
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)


# TODO: unify regular and Wandb logger.


class WandbLogger:
    def __init__(
        self, wandb_config: WandBConfig, init_config: dict, retries_init: int = 3
    ) -> None:
        self.config = wandb_config
        self.init_config = {
            "entity": self.config.entity,
            "project": self.config.project_name,
            "notes": self.config.notes,
            "tags": self.config.tags,
            "config": init_config,
            "resume": self.config.resume,
            "id": self.config.id,
            "name": self.config.run_name,
        }
        self.entries = dict()
        self.retries_init = retries_init
        self._enabled = False

    @property
    def enabled(self):
        return self.config.enabled and self._enabled

    @enabled.setter
    def enabled(self, enable: bool):
        self._enabled = enable

    def define_metric(self, prefix_path: str, metric_name: str):
        """
        Add custom metric for the x-axis on the dashboard plots to the prefix path to override the default "step"
        """
        if self.enabled:
            wandb.define_metric(prefix_path, step_metric=metric_name)

    def init(self):
        if self.config.enabled:
            for _ in range(self.retries_init):
                try:
                    wandb.init(**self.init_config)
                    self.enabled = True
                    break
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize wandb with error: {e}, retrying in 5 seconds..."
                    )
                    time.sleep(5)
            else:
                raise RuntimeError("Failed to initialize wandb")

        # wandb.run.name = wandb.run.id

    def watch_model(self, model: torch.nn.Module):
        if self.enabled:
            wandb.watch(model, log="all", log_freq=100, log_graph=False)

    def log(
        self,
        log_dict: dict,
        commit: bool = True,
        step=None,
        key: str = "",
        prefix: Optional[str] = None,
    ):
        if self.enabled:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in log_dict.items()}
                if prefix
                else log_dict
            )

    def set_config_value(self, key: str, value):

        if self.enabled:
            wandb.config[key] = value

    def log_img(
        self,
        key: str,
        img: Image.Image,
        commit: bool = True,
        step=None,
        caption: str = "",
    ):

        if self.enabled:
            wandb.log({key: wandb.Image(img, caption=caption)})

    def log_table(
        self, key: str, columns: List[str], data, commit: bool = True, step=None
    ):
        if self.enabled:
            table = wandb.Table(columns=columns)
            entries = self.entries.get(key, list())
            entries.append(data)
            # this sucks so f***ing bad, but atm you cannot append to wandb tables
            for entry in entries:
                for data in entry:
                    table.add_data(*data)
            self.entries[key] = entries
            wandb.log(
                {key: table},
            )

    def save_bestmodel(self, state):
        if self.enabled:
            torch.save(state, os.path.join(wandb.run.dir, "model_best.pth"))

    def finish(self):
        if self.enabled:
            wandb.finish()
