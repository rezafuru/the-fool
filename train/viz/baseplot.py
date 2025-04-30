import abc
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from log_manager.manager import LoggingManager


class BasePlot(abc.ABC):
    def __init__(self, filepath: Path, caption: str, ax: matplotlib.axes = plt.gca()):
        self.filepath = filepath
        self.caption = caption
        self.ax = ax
        matplotlib.rcParams["axes.linewidth"] = 2

    def showPlot(self):
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def savePlot(self):
        plt.savefig(self.filepath)
        plt.cla()
        plt.clf()
        plt.close()
        msg = f"successfully saved plot with caption '{self.caption}' to path: '{self.filepath}'"
        LoggingManager.get_default_logger().info(msg)
        print(msg)

    @abc.abstractmethod
    def createPlot(self):
        pass

    def generatePlotFile(self):
        self.createPlot()
        self.savePlot()
