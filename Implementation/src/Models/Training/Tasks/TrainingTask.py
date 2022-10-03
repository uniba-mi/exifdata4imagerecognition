from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
from Models.Classifiers.Classifier import Classifier
from keras.models import Model
from keras.callbacks import History
from Models.Training.Generators.BatchGeneratorProvider import BatchGeneratorProvider
from Models.Training.Generators.BatchGenerators import BatchGenerator
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, multilabel_confusion_matrix, PrecisionRecallDisplay, average_precision_score, precision_recall_curve
import time
import shutil
import csv
from keras.utils.vis_utils import plot_model
import math
from itertools import cycle

def createPrecisionRecallGraph(yTrue, yPred, classes: List[str], storagePath: Path, name: str, multilabel: bool):
    """ creates a precision-recall graph for the given true / predicted labels, using the given class names.
    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html"""

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

def createEvaluationFiles(dataGenerator: BatchGenerator, model: Model, name: str, classNames: List[str], storagePath: Path, omitAxisLabels: bool = False, multiLabel: bool = False):
    """ Creates evaluation files for the given model, evaluating the given data generator. """

    # predict the data generators data and get predictionResulttrue labels
    predictionResult = dataGenerator.predictOn(model = model, multiLabel = multiLabel, batchSize = 1)

    # get ids
    ids = dataGenerator.ids(len(predictionResult.predictionY))

    trueClassNames = [[classNames[i] for i in range(len(sub)) if sub[i] == 1.0] for sub in predictionResult.trueY]
    predictedClassNames = [[classNames[i] for i in range(len(sub)) if sub[i] == 1.0] for sub in predictionResult.predictionYRounded]

    labels = list(zip(ids, trueClassNames, predictionResult.trueY, predictedClassNames, predictionResult.predictionYRounded, predictionResult.predictionY))
    labels = sorted(labels, key = lambda x: x[0])
    with open(storagePath.joinpath(name + "_labels.csv"), "w") as labelFile:
        writer = csv.writer(labelFile) 
        writer.writerow(["id",  "true_name", "true", "predicted_name", "predicted_round", "predicted_prob"]) 
        writer.writerows(labels)
    
    # save ids of misclassified examples
    with open(storagePath.joinpath(name + "_misclassified_ids.csv"), "w") as idFile:
        writer = csv.writer(idFile) 
        for identifier, true, predicted in zip(ids, predictionResult.trueY, predictionResult.predictionYRounded):
            if not np.array_equal(true, predicted):
                writer.writerow([identifier.tolist()])

    # create classification report
    classificationReport = classification_report(predictionResult.trueY, predictionResult.predictionYRounded, digits = 3, output_dict = True, target_names = classNames) 
    reportDataFrame = pd.DataFrame(data = classificationReport).transpose().round(3)
    reportDataFrame.to_csv(storagePath.joinpath(name + "_classification_report.csv"))

    # create confusion matrix
    confusionMatrixSavePath = storagePath.joinpath(name + "_confusion_matrix" + str(dataGenerator.size) + ".png")
    if multiLabel:
        cf = multilabel_confusion_matrix(predictionResult.trueY, predictionResult.predictionYRounded)
        columns = 3
        rows = math.ceil(len(classNames) / columns)
        f, axes = plt.subplots(rows, columns, figsize = (14, 14))
        axes = axes.ravel()
        for i in range(len(classNames)):
            plot = ConfusionMatrixDisplay(confusion_matrix = cf[i], 
                                          display_labels = ["not " + classNames[i], classNames[i]])
            plot.plot(ax = axes[i], cmap = "Blues")
            plot.ax_.set_title(" ")
            plot.im_.colorbar.remove()
            plot.ax_.set_xlabel("")
            plot.ax_.set_ylabel("")

        # remove empty sub-plots
        for i in range(len(classNames), rows * columns):
            axes[i].set_axis_off()

        plt.subplots_adjust(wspace = 0.5, hspace = 0.2)
        f.savefig(confusionMatrixSavePath, dpi = 300)
    else:
        plot = ConfusionMatrixDisplay.from_predictions(predictionResult.trueYMax, 
                                                       predictionResult.predictionYMax, 
                                                       display_labels = classNames, 
                                                       cmap = "Blues", 
                                                       ax = plt.subplots(figsize = (12, 12))[1])

        if not omitAxisLabels:
            plt.rcParams["axes.titlepad"] = 25
            plt.title("Confusion Matrix", fontsize = 25)
            plt.xlabel("Predicted Label", fontsize = 20)
            plt.ylabel("True Label", fontsize = 20)
        else:
            plt.title("")
            plt.xlabel("")
            plt.ylabel("")
    
        plt.xticks(rotation = 90, fontsize = 15)
        plt.yticks(fontsize = 15)
        plot.figure_.savefig(confusionMatrixSavePath, dpi = 300)
    
    # create p-r graph
    createPrecisionRecallGraph(predictionResult.trueY, 
                               predictionResult.predictionY, 
                               classes = classNames, 
                               storagePath = storagePath, 
                               name = name,
                               multilabel = multiLabel)

    # save prediction time to txt
    with open(storagePath.joinpath(name + "_prediction_time.txt"), "w") as timeFile:
        timeFile.write(str(predictionResult.predictionTime))
    
def createTrainingFiles(history: History, duration: float, storagePath: Path, earlyStoppingPatience: int = 0):
    """ Generates plots and statistics for the given training history and training duration. """

    # identify accuracy metric key
    accuracyKey = None
    for key in history.history.keys():
        if "accuracy" in key and not "val" in key:
            accuracyKey = key
            break

    # plot training 
    accuracy = history.history[accuracyKey]
    validationAccuracy = history.history["val_" + accuracyKey]
    loss = history.history["loss"]
    validationLoss = history.history["val_loss"]
    targetEpoch = len(history.history["val_loss"]) - earlyStoppingPatience - 1 if earlyStoppingPatience > 0 else 0

    # create plot for accuracy
    plt.figure(figsize = (9, 5))
    plt.plot(accuracy, label = "training", color = "deepskyblue")
    plt.plot(validationAccuracy, label = "validation", color = "darkseagreen")
    plt.ylim([min(plt.ylim()), 1])
    if targetEpoch > 0:
        plt.axvline(x = targetEpoch, color = "black", label = "target epoch")
    plt.legend(fontsize = "medium", loc = "upper left", bbox_to_anchor = (1.01, 1))
    plt.xlabel("epoch", fontsize = 12)
    plt.ylabel('accuracy', fontsize = 12)
    plt.savefig(storagePath.joinpath("training_accuracy.png"), bbox_inches = "tight", dpi = 300)

    # create plot for loss
    plt.figure(figsize = (9, 5))
    plt.plot(loss, label = "training", color = "deepskyblue")
    plt.plot(validationLoss, label = "validation", color = "darkseagreen")
    plt.ylim([0, max(validationLoss) + 0.5])
    if targetEpoch > 0:
        plt.axvline(x = targetEpoch, color = "black", label = "target epoch")
    plt.legend(fontsize = "medium", loc = "upper left", bbox_to_anchor = (1.01, 1))
    plt.xlabel("epoch", fontsize = 12)
    plt.ylabel("loss", fontsize = 12)
    plt.savefig(storagePath.joinpath("training_loss.png"), bbox_inches = "tight", dpi = 300)

    # save history to csv
    historyDataframe = pd.DataFrame(history.history).round(3)
    historyDataframe.to_csv(storagePath.joinpath("training_history.csv"))

    # save training time to txt
    with open(storagePath.joinpath("training_time.txt"), "w") as timeFile:
        timeFile.write(str(duration))

class TrainingTask(object):
    """ A training task trains and evaluates the given classifier. The evaluation / training / model files will be
    stored at the given storage path. The given batch generator provider is used to obtain the data generators which
    are used for training / evaluating the classifier. """

    classifier: Classifier
    modelStoragePath: Path
    logStoragePath: Path
    dataProvider: BatchGeneratorProvider
    disableEvaluationAxisLabels: bool

    def __init__(self, classifier: Classifier, storagePath: str, provider: BatchGeneratorProvider, disableEvaluationAxisLabels: bool = False):
        self.classifier = classifier
        self.modelStoragePath = Path(storagePath).joinpath(classifier.name)
        self.logStoragePath = self.modelStoragePath.joinpath("tensorboard_log")
        self.dataProvider = provider
        self.disableEvaluationAxisLabels = disableEvaluationAxisLabels

    def run(self, epochs: int, earlyStoppingPatience: int, optimize: str = "accuracy", verbose: int = 1):
        """ Runs the training task and evaluates the model. Additionally persists the model as well as evaluation metrices. """
        if verbose > 0:
            print("starting training task for " + self.classifier.name + " ...")

        # create output directory for the model / evaluation files (delete directory if it is already there)
        if self.modelStoragePath.exists():
            shutil.rmtree(self.modelStoragePath)
        self.modelStoragePath.mkdir(parents = True, exist_ok = True)
        self.logStoragePath.mkdir(parents = False, exist_ok = True)

        # create data generators and apply shapes to the model builder
        trainGenerator, testGenerator, valGenerator = self.dataProvider.createGenerators()
        self.classifier.inputShape = self.dataProvider.inputShape
        self.classifier.outputShape = self.dataProvider.outputShape
        self.classifier.multiLabel = self.dataProvider.multiLabel
        
        if verbose > 0:
            print("starting model training ...")
        start = time.time()

        # start training
        history = self.classifier.train(trainDataGenerator = trainGenerator,
                                          testDataGenerator = testGenerator,
                                          epochs = epochs,
                                          optimize = optimize,
                                          earlyStoppingPatience = earlyStoppingPatience,
                                          logDir = self.logStoragePath,
                                          verbose = verbose)
        # caluclate duration in seconds                             
        duration = round(time.time() - start, 3)

        if verbose > 0:
            print("finished model training, took " + str(duration) + " seconds ...")
            print("creating model evaluation files ...")
        
        # create training log
        createTrainingFiles(history = history, 
                           duration = duration, 
                           storagePath = self.modelStoragePath,
                           earlyStoppingPatience = self.classifier.lastEarlyStoppingPatience)
        
        # evaluate model and create evaluation files
        createEvaluationFiles(dataGenerator = trainGenerator, 
                              model = self.classifier.model, 
                              name = "training",
                              classNames = self.dataProvider.classes,
                              storagePath = self.modelStoragePath,
                              omitAxisLabels = self.disableEvaluationAxisLabels,
                              multiLabel = self.dataProvider.multiLabel)
        
        createEvaluationFiles(dataGenerator = testGenerator, 
                              model = self.classifier.model, 
                              name = "test",
                              classNames = self.dataProvider.classes,
                              storagePath = self.modelStoragePath,
                              omitAxisLabels = self.disableEvaluationAxisLabels,
                              multiLabel = self.dataProvider.multiLabel)

        createEvaluationFiles(dataGenerator = valGenerator, 
                              model = self.classifier.model, 
                              name = "validation",
                              classNames = self.dataProvider.classes,
                              storagePath = self.modelStoragePath,
                              omitAxisLabels = self.disableEvaluationAxisLabels,
                              multiLabel = self.dataProvider.multiLabel)

        if verbose > 0:
            print("saving model file to hard disk ...")

        # save model file
        try:
            self.classifier.model.save(self.modelStoragePath.joinpath("model.keras"))
        except Exception as exception:
            print(exception)
            print("failed to save model, try to save model ...")

        # save model parameters
        with open(self.modelStoragePath.joinpath("model_parameters.txt"), "w") as parameterFile:
            self.classifier.model.summary(print_fn = lambda line: parameterFile.write(line + '\n'))
        
        # save class names
        with open(self.modelStoragePath.joinpath("class_names.txt"), "w") as classFile:
            classFile.write(",".join(self.dataProvider.classes))

        # save model graph
        plot_model(self.classifier.model, 
                   to_file = self.modelStoragePath.joinpath("model.png"), 
                   show_shapes = True, 
                   show_dtype = True,
                   show_layer_names = True, 
                   rankdir = "TB", 
                   expand_nested = True, 
                   dpi = 300)

        if verbose > 0:
            print("training task finished for " + self.classifier.name)
    