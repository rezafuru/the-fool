from typing import Optional

import torch
from torch.cuda.amp import GradScaler


class GradScaleMockWrapper:
    def __init__(self, scaler: Optional[GradScaler] = None):
        self.scaler = scaler
        self.enabled = scaler is not None

    def scale(self, loss):
        if self.scaler:
            return self.scaler.scale(loss)
        else:
            return loss

    def step(self, optim: torch.optim.Optimizer):
        if self.scaler:
            self.scaler.step(optim)
        else:
            optim.step()

    def unscale_(self, optimizer: torch.optim.Optimizer):
        if self.scaler:
            self.scaler.unscale_(optimizer)

    def update(self):
        if self.scaler:
            self.scaler.update()
