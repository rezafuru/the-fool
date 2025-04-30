from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from train.viz.baseplot import BasePlot

# todo: Implement useful train plots (PSN/BPP progression per X iter/epoch (?))


class WaveformPlot(BasePlot):
    """
    class to plot ECG signal in waveform
    """

    def __init__(
        self,
        filepath: Path,
        caption: str,
        ax: matplotlib.axes,
        wavedata: np.ndarray,
        overlaydata: np.ndarray = None,
        dim0_idx: int = 0,
        xlabel: str = "Time",
        ylabel: str = "Amplitude",
    ):
        super().__init__(filepath, caption, ax)
        self.wavedata = wavedata

        if len(self.wavedata.shape) > 0:
            self.wavedata = self.wavedata[dim0_idx]

        self.overlaydata = overlaydata
        self.xlabel = xlabel
        self.ylabel = ylabel

    def createPlot(self):
        tick = np.array(range(1, len(self.wavedata) + 1))
        self.ax.plot(tick, self.wavedata)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.caption)


class WaveformComparisionPlot(BasePlot):
    """
    compares n waveforms rowise
    """

    def __init__(
        self,
        filepath: Path,
        caption: str,
        records: List[np.ndarray],
        labels: List[str],
        min_fig_size: int = 5,
        ax: matplotlib.axes = None,
    ):
        super().__init__(filepath, caption, ax)

        assert len(records) == len(labels)
        num_plots = len(records)
        self.min_fig_size = min_fig_size
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(self.min_fig_size * 2, (self.min_fig_size) * num_plots),
        )
        axes = axes.flatten()
        self.subplots = []

        for (wavedata, label, ax) in zip(records, labels, axes):
            self.subplots.append(WaveformPlot(filepath, label, ax, wavedata))

    def _align_axes(self):
        """
        aligns all given axes to the same axis limits
        """

        (xlim_min, xlim_max) = self.subplots[0].ax.get_xlim()
        (ylim_min, ylim_max) = self.subplots[0].ax.get_ylim()

        for plot in self.subplots:
            if "difference" not in plot.caption:  # todo fix this uggly solution
                xlim = plot.ax.get_xlim()
                xlim_min = xlim[0] if xlim[0] < xlim_min else xlim_min
                xlim_max = xlim[1] if xlim[1] > xlim_max else xlim_max
                ylim = plot.ax.get_ylim()
                ylim_min = ylim[0] if ylim[0] < ylim_min else ylim_min
                ylim_max = ylim[1] if ylim[1] > ylim_max else ylim_max

        for plot in self.subplots:
            if "difference" not in plot.caption:  # todo fix this uggly solution
                plot.ax.set_xlim((xlim_min, xlim_max))
                plot.ax.set_ylim((ylim_min, ylim_max))

    def createPlot(self):
        for subplot in self.subplots:
            subplot.createPlot()

        # self._align_axes()
