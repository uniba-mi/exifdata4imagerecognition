from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from collections import OrderedDict
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve
from itertools import cycle

def createBarChart(data: List, 
                   seriesLabels: List[str], 
                   categoryLabels: List[str] = None, 
                   showValues: bool = False, 
                   showNegativeValues: bool = False,
                   valueFormat: str = "{}", 
                   title: str = None, 
                   yLabel: str = None,
                   grid: bool = False,
                   legendPosition: str = "lower left",
                   figSize: Tuple = (14.5, 8),
                   yLabelFormatter: ticker.FuncFormatter = None,
                   barWidth: float = 0.88,
                   labelOffset: float = 0.014,
                   yLimit: float = None,
                   labelPrefix: str = "",
                   labelPostfix: str = "",
                   colors = [["steelblue", "darkolivegreen", "darkgoldenrod", "orangered"], 
                             ["skyblue", "darkseagreen", "goldenrod", "coral"],
                             ["skyblue", "darkseagreen", "goldenrod", "coral"], 
                             ["skyblue", "darkseagreen", "goldenrod", "coral"]],
                   additionalLines: List[float] = None,
                   bigFont: bool = False,
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
                       linewidth = 1.0,
                       alpha = 0.75 if outerIndex == 1 else 1.0,
                       color = (colors[outerIndex][innerIndex] if colors is not None else None), 
                       width = barWidth / len(subData)))

    #font = {'fontname':'Times New Roman'}

    if categoryLabels:
        indexList = list(range(1, len(data[0][0]) + 1))
        labelPos = [(x - (barWidth / len(data[0]) * 2.0)) + ((len(data[0]) / 2.0) - 0.5) * (barWidth / len(data[0])) for x in indexList]
        if bigFont:
            plt.xticks(labelPos, categoryLabels, fontsize = 18, rotation = 60) 
        else:
            plt.xticks(labelPos, categoryLabels)
    
    if bigFont:
        params = {'mathtext.default': 'it', 'font.size': 18 }          
        plt.rcParams.update(params)

    if title:
        plt.title(title)

    if yLabel:
        if bigFont:
            plt.ylabel(yLabel, fontsize = 18, labelpad = 10)
        else:
            plt.ylabel(yLabel)
    
    if seriesLabels[0][0] != "":
        handles, labels = plt.gca().get_legend_handles_labels()
        if not additionalLines is None:
            line = Line2D([0], [0], label = "EXIF-Only", color = "black", linestyle = "solid")
            dummyLine = Line2D([0], [0], label = "", color = "w", alpha = 0)
            handles.extend([dummyLine])
            handles.extend([dummyLine])
            handles.extend([dummyLine])
            handles.extend([line])
        duplicatesRemovedLabels = OrderedDict(zip(labels, handles))
        plt.legend(handles = handles, loc = legendPosition, ncol = 3 if not additionalLines is None else 2)
    
    ax = plt.gca()
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    if not additionalLines is None:
        count = 1
        centers = [(x - (barWidth / len(data[0]) * 2.0)) + ((len(data[0]) / 2.0) - 0.5) * (barWidth / len(data[0])) for x in indexList]
        for val in additionalLines:
            ax.plot([centers[count - 1] - (len(data[0]) / 2.0) * (barWidth / len(data[0])), centers[count - 1] + (len(data[0]) / 2.0) * (barWidth / len(data[0]))], [val, val],
                    linestyle = "solid", linewidth = 2, color = "black")
            count += 1

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")
    
    if yLabelFormatter != None:
        if yLimit != None:
            ax.yaxis.set_ticks(np.arange(ax.get_ylim()[0], yLimit, 0.05))
        ax.yaxis.set_major_formatter(yLabelFormatter)

    if yLimit != None:
        ax.set_ylim([0.0, yLimit]) #ax.get_ylim()[0]
    
    if bigFont:
        plt.yticks(fontsize = 18)

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
                    if h > 0.0000 or showNegativeValues:
                        plt.text(bar.get_x() + w / 2, 
                                bar.get_y() + h - labelOffset,
                                labelPrefix + valueFormat.format(h) + labelPostfix, 
                                ha = "center", 
                                va = "center",
                                fontsize = 14)
                        bars.append(bar)
    
    plt.tight_layout()
    
    if savePath != None:
        plt.savefig(savePath, dpi = 300, transparent = True)
    
def createSingleBarChart(data: List, 
                         seriesLabels: List[str], 
                         categoryLabels: List[str] = None, 
                         showValues: bool = False, 
                         valueFormat: str = "{}", 
                         title: str = None, 
                         yLabel: str = None,
                         grid: bool = False,
                         bigFont: bool = False,
                         legendPosition: str = "lower left",
                         figSize: Tuple = (14.5, 8),
                         barWidth: float = 0.9,
                         labelOffset: float = 0.005,
                         yLabelFormatter: ticker.FuncFormatter = None,
                         valueFormatFract: int = 1,
                         yLimit: float = None,
                         sc: bool = False,
                         labelPrefix: str = "",
                         labelPostFix: str = "",
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

    if bigFont:
        plt.xticks(list(range(0, len(categoryLabels))), categoryLabels, rotation = 60, fontsize = 20)
    else:
        plt.xticks(list(range(0, len(categoryLabels))), categoryLabels, rotation = "horizontal")

    ax = plt.gca()
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    if bigFont:
        params = {'mathtext.default': 'it', 'font.size': 18 }          
        plt.rcParams.update(params)
    
    if len(seriesLabels) > 0:
        plt.legend(loc = legendPosition, ncol = 2, fontsize = 18)

    if grid:
        plt.grid(color = "gray", linestyle = '--', linewidth = 0.5, which = "both")
    
    if title:
        plt.title(title)

    if yLabel:
        if bigFont:
            plt.ylabel(yLabel, fontsize = 20, labelpad = 10)
        else:
            plt.ylabel(yLabel)

    if yLabelFormatter != None:
        ax.yaxis.set_major_formatter(yLabelFormatter)
    
    if yLimit != None:
        ax.set_ylim([0, yLimit])

    ax.yaxis.set_ticks(np.arange(0, yLimit, 5e6))
    
    if showValues:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w / 2, 
                         bar.get_y() + h - labelOffset,
                         labelPrefix + valueFormat.format(h / valueFormatFract) + labelPostFix, 
                         ha = "center", 
                         va = "center",
                         fontsize = 18)

    if bigFont:
        plt.yticks(fontsize = 18)

    plt.tight_layout()
            
    if savePath != None:
        plt.savefig(savePath, dpi = 300, transparent = True)


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
                                   labelPostFix: str = "",
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
                            labelPrefix + valueFormat.format(w) + labelPostFix, 
                            ha = "center", 
                            va = "center",
                            fontsize = 9)

    plt.tight_layout()

    if savePath != None:
        plt.savefig(savePath, dpi = 300, transparent = True)
        
def createTrainingAccuracyLossChart(dataFrames: List[pd.DataFrame], 
                                    dataIndexKey: str, 
                                    valDataIndexKey: str, 
                                    figSize: Tuple = (14, 7.5), 
                                    labels: List = [], 
                                    title: str = "",
                                    targetEpochPatience: int = 0,
                                    startFineTuningEpoch: int = 0,
                                    savePath: Path = None):

    fig, axs = plt.subplots(math.ceil(len(dataFrames) / 2.0), 2 if len(dataFrames) > 1 else 1, figsize = figSize)
    row = 0
    index = 0
    count = 0

    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])

    for df in dataFrames:
        metric = df.loc[[dataIndexKey]]
        valMetric = df.loc[[valDataIndexKey]]
        x = metric.columns.to_numpy()
        y = metric.values[0]
        xVal = valMetric.columns.to_numpy()
        yVal = valMetric.values[0]

        ax = axs[row, index] if isinstance(axs, np.ndarray) else axs
        ax.plot(x, y, color = "deepskyblue")
        ax.plot(xVal, yVal, color = "darkseagreen")
        ax.set_title(labels[count], fontsize = 18)

        start, end = ax.get_ylim()
        end = np.min([end + 0.1, 1.0])

        if dataIndexKey == "loss":
            start = np.max([0, start - 0.15])
            end = max(yVal) + 0.15

        ax.set_ylim([start, end])
        ax.yaxis.set_ticks(np.arange(start, end, (end - start) / 8.0))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(0, end, math.ceil(max(x) / 10.0)))
        ax.tick_params(axis = "both", which = "major", labelsize = 18)

        if targetEpochPatience > 0:
            ax.axvline(x = max(x) - targetEpochPatience, color = "black")
        
        if startFineTuningEpoch > 0:
            ax.axvline(x = startFineTuningEpoch, color = "orange")

        count += 1
        if index < 1:
            index += 1
        else:
            index = 0
            row += 1

    legendLines = [Line2D([0], [0], color = "deepskyblue", lw = 2),
                   Line2D([0], [0], color = "darkseagreen", lw = 2),
                   Line2D([0], [0], color = "black", lw = 2),
                   Line2D([0], [0], color = "orange", lw = 2)]
    
    legendLabels = ["training set " + (dataIndexKey if dataIndexKey == "loss" else "accuracy"), 
                    "validation set " + (dataIndexKey if dataIndexKey == "loss" else "accuracy"), 
                    "target epoch",
                    "start fine-tuning"]
    
    if count == 1:
        legendLines.pop()
        legendLabels.pop()
    
    fig.legend(legendLines, 
               legendLabels,
               loc = "upper left", 
               fontsize = 19, 
               bbox_to_anchor = (0.18, 1), ncol = 2) #if count > 1 else 0.75)

    #fig.suptitle(title)
    fig.tight_layout()
    make_space_above(axs, topmargin = 1.5)  

    if savePath != None:
        plt.savefig(savePath, dpi = 300, transparent = True)

def make_space_above(axes, topmargin = 1):
    """ increases figure size to add topmargin (in inches) space for titles, without changing the axes sizes 
    adapted from: https://stackoverflow.com/questions/25068384/bbox-to-anchor-and-loc-in-matplotlib """
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def createImageOverviewChart(images: List[Tuple], figSize: Tuple = (9, 6), imagesPerRow: int = 6, savePath: Path = None):
    rows = math.ceil(len(images) / imagesPerRow)
    fig = plt.figure(figsize = figSize)
    for index, image in enumerate(images):
        ax = plt.subplot(rows, imagesPerRow, index + 1)
        imageRGB = mpimg.imread(image[1])
        plt.imshow(imageRGB)
        plt.title(image[0], fontsize = 25)
        ax.axis("off")
    
    plt.axis("off")
    fig.tight_layout()

    if savePath != None:
        plt.savefig(savePath, dpi = 300, transparent = True)

def createPrecisionRecallGraph(yTrue, yPred, classes: List[str], storagePath: Path, name: str, multilabel: bool):
    """ creates a precision-recall graph for the given true / predicted labels, using the given class names.
    adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html"""

    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()
    for i in range(len(classes)):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(yTrue[:, i], yPred[:, i])
        average_precision[i] = average_precision_score(yTrue[:, i], yPred[:, i])

    precision["micro"], recall["micro"], thresholds["micro"] = precision_recall_curve(yTrue.ravel(), yPred.ravel())   
    f1 = (2 * precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"])
    average_precision["micro"] = average_precision_score(yTrue, yPred, average="micro")

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal", "cyan", "green", "blue", "yellow", "brown", "purple", "olive", "gray", "indigo"])
    _, ax = plt.subplots(figsize = (7, 8))
    f_scores = np.linspace(0.1, 0.8, num = 7)
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color = "gray", alpha = 0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy = (0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(recall = recall["micro"], 
                                     precision = precision["micro"], 
                                     average_precision = average_precision["micro"])
    bestF1Index = np.argmax(f1)
    bestF1 = f1[bestF1Index]
    display.plot(ax = ax, name="average", color="gold")
    plt.scatter(recall["micro"][bestF1Index], precision["micro"][bestF1Index], marker = "x", color=  "red", zorder = 10)
    ax.annotate(f"f1={bestF1:.2f}", (recall["micro"][bestF1Index], precision["micro"][bestF1Index] + 0.02), zorder = 11, weight = "bold")

    if multilabel:
        for i, color in zip(range(len(classes)), colors):
            display = PrecisionRecallDisplay(recall = recall[i],
                                         precision = precision[i],
                                         average_precision = average_precision[i])
            display.plot(ax = ax, name=f"{classes[i]}", color=color)

    # add the legend (iso-f1)
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])

    # add the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles = handles, labels = labels, loc = "upper left", bbox_to_anchor = (1.02, 1))
    display.figure_.savefig(storagePath.joinpath(name + "_precision_recall_graph_single.png"), dpi = 300, bbox_inches = "tight")
