from Models.Training.Tasks.TrainingTask import TrainingTask
from Models.Training.Generators.BatchGeneratorProvider import BatchGeneratorProvider
from Models.Classifiers.MLP import MLPClassifier
from Models.Classifiers.CNN import CNNClassifier
from Models.Classifiers.MixedModel import MixedModel
from typing import Callable
from keras import Model

class ExifImageMixedTrainingTask(object):
    """ A training task for training a mixed model (exif data + image data). The MLP used to classify exif
    data is pre-trained, before the training of the mixed model starts. """

    modelStoragePath: str
    dataProvider: BatchGeneratorProvider
    cnnModelFunction: Callable[[], Model]
    modelName: str
    fineTuneLayersCount: int
    fineTuneEpochsCount: int

    def __init__(self, 
                 storagePath: str, 
                 provider: BatchGeneratorProvider, 
                 modelName: str, 
                 cnnModelFunction: Callable[[], Model],
                 fineTuneLayersCount: int,
                 fineTuneEpochsCount: int):
        self.modelStoragePath = storagePath
        self.dataProvider = provider
        self.modelName = modelName
        self.cnnModelFunction = cnnModelFunction
        self.fineTuneEpochsCount = fineTuneEpochsCount
        self.fineTuneLayersCount = fineTuneLayersCount

    def run(self, epochs: int, earlyStoppingPatience: int, optimize: str = "accuracy", verbose: int = 1):
        batchSize = self.dataProvider.batchSize
        self.dataProvider.exifOnly = True
        self.dataProvider.batchSize = 128
        
        mlp = MLPClassifier(name = "MLP")

        # create training task for MLP (pre-training)
        trainingTask = TrainingTask(classifier = mlp, 
                                    storagePath = None,
                                    provider = self.dataProvider)
        trainingTask.run(epochs = 500, earlyStoppingPatience = 50, optimize = optimize, verbose = verbose)

        # create training task for mixed model
        self.dataProvider.batchSize = batchSize
        self.dataProvider.exifOnly = False

        trainingTask = TrainingTask(classifier = MixedModel(name = self.modelName, 
                                                            classifier1 = CNNClassifier(name = "CNN", modelFunction = self.cnnModelFunction), 
                                                            classifier2 = mlp, 
                                                            fineTuneLayersCount = self.fineTuneLayersCount, 
                                                            fineTuneEpochs = self.fineTuneEpochsCount), 
                                    storagePath = self.modelStoragePath, 
                                    provider = self.dataProvider)
        
        trainingTask.run(epochs = epochs, earlyStoppingPatience = earlyStoppingPatience, optimize = optimize, verbose = verbose)