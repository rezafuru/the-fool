from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torchvision import transforms

from misc.dclasses import DataVizConfig


# todo: implement visualizers


class DataVisualizer:
    def __init__(self, config: DataVizConfig) -> None:
        self.config = config
        # todo: Fix that dataset.root becomes a dict object and not a Path object
        self.config.dataset.root = Path(self.config.dataset.root["root"])
        self._init_dataset(self.config.dataset.root)

        self.config.outputpath = self.config.outputpath / Path(
            self.config.dataset.root.name
        )
        self.config.outputpath.mkdir(exist_ok=True, parents=True)

    def _init_dataset(self, path: Path):
        self.dataset = LabeledAWGNoisySignalDataset(
            root=path,
            transform=transforms.Compose([transforms.ToTensor()]),
            db_target="22.5",
        )

    @staticmethod
    def plot_gt_vs_noise_vs_recon(
        gt_np: np.ndarray,
        noisy_np: np.ndarray,
        recon_np: np.ndarray,
        noise_label: str = "noise",
    ) -> Figure:
        plot = WaveformComparisionPlot(
            None,
            "",
            [gt_np, noisy_np, recon_np],
            ["ground truth", noise_label, "reconstruction"],
        )
        plot.createPlot()
        return plt.gcf()

    def plot_gt_vs_noise(self, noise_label: str = "noise", sample_idx: int = 0):

        gt, noisy, label = self.dataset[sample_idx]
        gt_np = gt.detach().cpu().numpy()
        noisy_np = noisy.detach().cpu().numpy()
        diff_np = np.abs(gt_np - noisy_np)

        filename = f"Waveform gt vs. {noise_label} sample_{sample_idx}"
        plot = WaveformComparisionPlot(
            self.config.outputpath / Path(filename + ".png"),
            filename,
            [gt_np, noisy_np, diff_np],
            ["ground truth", noise_label, "difference"],
        )
        plot.generatePlotFile()

    def plot_gt_vs_all_noise_variants(self, sample_idx: int = 0):
        """
        iterates through all noise variants and plots them
        """

        for type_folder in self.config.dataset.root.iterdir():
            if type_folder.is_dir():
                noise_type = type_folder.name
                for noise_amount in type_folder.iterdir():
                    if noise_amount.is_dir():
                        self._init_dataset(noise_amount)
                        self.plot_gt_vs_noise(
                            noise_label=f"{noise_type}_{noise_amount.name}",
                            sample_idx=sample_idx,
                        )
