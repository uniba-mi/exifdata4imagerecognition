from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    
    for outerIndex, subData in enumerate(data):
        for innerIndex, rowData in enumerate(subData):
            indexList = list(range(1, len(rowData) + 1))
            axes.append(plt.bar([(x - (barWidth / len(subData) * 2.0)) + innerIndex * (barWidth / len(subData)) for x in indexList], 
                       rowData, 
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
    
    ax = plt.gca()
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")

    if showValues:
        bars = []
        for axis in axes:
            for bar in axis:
                drawLabel = True
                for oldBar in bars:
                    if oldBar.get_x() == bar.get_x():
                        if (oldBar.get_y() + oldBar.get_height()) - (bar.get_y() + bar.get_height()) < 0.021:
                            drawLabel = False
                
                if drawLabel:
                    w, h = bar.get_width(), bar.get_height()
                    if h > 0.0000:
                        plt.text(bar.get_x() + w / 2, 
                                bar.get_y() + h - labelOffset,
                                labelPrefix + valueFormat.format(h), 
                                ha = "center", 
                                va = "center",
                                fontsize = 9)
                        bars.append(bar)
    
    plt.tight_layout()
    
    if savePath != None:
        plt.savefig(savePath, dpi = 300)
    
def createSingleBarChart(data: List, 
                         seriesLabels: List[str], 
                         categoryLabels: List[str] = None, 
                         showValues: bool = False, 
                         valueFormat: str = "{}", 
                         title: str = None, 
                         yLabel: str = None,
                         grid: bool = False,
                         legendPosition: str = "lower left",
                         figSize: Tuple = (14.5, 8),
                         barWidth: float = 0.9,
                         labelOffset: float = 0.005,
                         yLabelFormatter: ticker.FuncFormatter = None,
                         yLimit: float = None,
                         sc: bool = False,
                         labelPrefix: str = "",
                         colors = ["skyblue", "darkseagreen", "goldenrod", "coral", 
                                   "steelblue", "darkolivegreen", "darkgoldenrod", "orangered"],
                         savePath: Path = None):
    plt.figure(figsize = figSize)
    indexList = list(range(0, len(data)))
    axes = []
    for innerIndex, rowData in enumerate(data):
        axes.append(plt.bar(indexList[innerIndex],
                    rowData, 
                    label = seriesLabels[innerIndex] if len(seriesLabels) - 1 >= innerIndex else "", 
                    edgecolor = "black",
                    color = colors[0] if sc else colors[innerIndex] if len(colors) - 1 >= innerIndex else colors[0], 
                    width = barWidth))

    plt.xticks(list(range(0, len(categoryLabels))), categoryLabels, rotation = "horizontal")

    ax = plt.gca()
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    
    if len(seriesLabels) > 0:
        plt.legend(loc = legendPosition, ncol = 2)

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")
    
    if title:
        plt.title(title)

    if yLabel:
        plt.ylabel(yLabel)

    if yLabelFormatter != None:
        ax.yaxis.set_major_formatter(yLabelFormatter)
    
    if yLimit != None:
        ax.set_ylim([0, yLimit])
    
    if showValues:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w / 2, 
                         bar.get_y() + h - labelOffset,
                         labelPrefix + valueFormat.format(h), 
                         ha = "center", 
                         va = "center",
                         fontsize = 9)
    plt.tight_layout()
            
    if savePath != None:
        plt.savefig(savePath, dpi = 300)


def createSingleBarChartHorizontal(data: List,
                                   seriesLabels: List[str], 
                                   categoryLabels: List[str] = None, 
                                   showValues: bool = False, 
                                   valueFormat: str = "{}",
                                   grid: bool = False,
                                   title: str = None, 
                                   figSize: Tuple = (14.5, 8),
                                   barHeight: float = 0.9,
                                   labelOffset: float = 0.007,
                                   sc: bool = False,
                                   xLimit: float = None,
                                   labelPrefix: str = "",
                                   colors = ["skyblue", "darkseagreen", "goldenrod", "coral", 
                                              "steelblue", "darkolivegreen", "darkgoldenrod", "orangered"],
                                   savePath: Path = None):
    
    plt.figure(figsize = figSize)
    indexList = list(range(0, len(data)))
    axes = []
    for innerIndex, rowData in enumerate(data):
        axes.append(plt.barh(indexList[innerIndex],
                    rowData, 
                    align = "center",
                    label = seriesLabels[innerIndex] if len(seriesLabels) - 1 >= innerIndex else "", 
                    edgecolor = "black",
                    color = colors[0] if sc else colors[innerIndex] if len(colors) - 1 >= innerIndex else colors[0], 
                    height = barHeight))
    ax = plt.gca()
    ax.set_yticks(indexList, labels = categoryLabels) 
    ax.invert_yaxis()
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    if title:
        plt.title(title)

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")
    
    if xLimit != None:
        ax.set_xlim([0, xLimit])
    
    if showValues:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                if w > 0.0000:
                    plt.text(bar.get_x() + w + labelOffset, 
                            bar.get_y() + h / 2,
                            labelPrefix + valueFormat.format(w), 
                            ha = "center", 
                            va = "center",
                            fontsize = 9)

    plt.tight_layout()

    if savePath != None:
        plt.savefig(savePath, dpi = 300)
        