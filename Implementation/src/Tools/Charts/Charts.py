from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

def createBarChart(data: List, 
                     seriesLabels: List[str], 
                     categoryLabels: List[str] = None, 
                     showValues: bool = False, 
                     valueFormat: str = "{}", 
                     title: str = None, 
                     yLabel: str = None,
                     grid: bool = False,
                     legendPosition: str = "lower left",
                     figSize: Tuple = (14.5, 8),
                     barWidth: float = 0.88,
                     labelOffset: float = 0.014,
                     labelPrefix: str = "",
                     colors = [["steelblue", "darkolivegreen", "darkgoldenrod", "orangered"], 
                               ["skyblue", "darkseagreen", "goldenrod", "coral"],
                               ["steelblue", "darkolivegreen", "darkgoldenrod", "orangered"], 
                               ["skyblue", "darkseagreen", "goldenrod", "coral"]],
                     savePath: Path = None):

    axes = []
    plt.figure(figsize = figSize)
    ax = plt.gca()
    for outerIndex, subData in enumerate(data):
        for innerIndex, row_data in enumerate(subData):
            indexList = list(range(1, len(row_data) + 1))
            axes.append(plt.bar([(x - (barWidth / len(subData) * 2.0)) + innerIndex * (barWidth / len(subData)) for x in indexList], 
                       row_data, 
                       label = seriesLabels[outerIndex][innerIndex], 
                       edgecolor = "black",
                       color = colors[outerIndex][innerIndex] if colors is not None else None, 
                       width = barWidth / len(subData)))

    if categoryLabels:
        indexList = list(range(1, len(data[0][0]) + 1))
        plt.xticks([(x - (barWidth / len(data[0]) * 2.0)) + ((len(data[0]) / 2.0) - 0.5) * (barWidth / len(data[0])) for x in indexList], categoryLabels)

    if title:
        plt.title(title)

    if yLabel:
        plt.ylabel(yLabel)

    if seriesLabels[0][0] != "":
        plt.legend(loc = legendPosition, ncol = 2)
    
    plt.tight_layout(rect = [0, 0, 1, 1])

    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    bars = []

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")

    if showValues:
        for axis in axes:
            for bar in axis:
                drawLabel = True
                for oldBar in bars:
                    if oldBar.get_x() == bar.get_x():
                        if (oldBar.get_y() + oldBar.get_height()) - (bar.get_y() + bar.get_height()) < 0.021:
                            drawLabel = False

                if drawLabel:
                    w, h = bar.get_width(), bar.get_height()
                    plt.text(bar.get_x() + w / 2, 
                            bar.get_y() + h - labelOffset,
                            labelPrefix + valueFormat.format(h), 
                            ha = "center", 
                            va = "center",
                            fontsize = 9)
                    bars.append(bar)
    
    if savePath != None:
        plt.savefig(savePath, dpi = 300)