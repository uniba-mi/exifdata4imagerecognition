import unittest
import tensorflow
from Models.Classifiers.MLP import MLPClassifier
from Models.Classifiers.CNN import CNNClassifier
from Models.Classifiers.MixedModel import MixedModel
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.TrainingTask import TrainingTask
from Tests.Training.TestTrainingConstants import TestTrainingConstants
from keras.applications.efficientnet import EfficientNetB0
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats

class MLPTrainingTests(unittest.TestCase):

    def testTrainIndoorOutdoorMixedModel(self):
        tensorflow.get_logger().setLevel("ERROR")
        #dataSetPaths = { TestTrainingConstants.datasetFilePath : ExifImageTrainingDataFormats.Flickr,
        #                 TestTrainingConstants.mirDatasetFilePath : ExifImageTrainingDataFormats.MirFlickr }
        
        dataSetPaths = { TestTrainingConstants.indoorOutdoorFilePath : ExifImageTrainingDataFormats.Flickr }

        dataProvider = ExifImageProvider(dataSetsPaths = dataSetPaths,
                                        cachePath = TestTrainingConstants.cacheStoragePath,
                                        trainSize = 0.7,
                                        testSize = 0.2,
                                        validationSize = 0.1,
                                        batchSize = 128,
                                        exifOnly = True,
                                        useSuperConcepts = False,
                                        imageSize = (150, 150),
                                        seed = 131)
        
        mlp = MLPClassifier(name = "MLP")

        # create training task
        trainingTask = TrainingTask(classifier = mlp, 
                                    storagePath = None,
                                    provider = dataProvider)
        trainingTask.run(epochs = 1, earlyStoppingPatience = 50, optimize = "loss")

        # create training task for mixed model
        dataProvider.batchSize = 32
        dataProvider.exifOnly = False

        task = TrainingTask(classifier = MixedModel(name = "MixedIndoorOutdoor", 
                                                      classifier1 = CNNClassifier(name = "CNN", modelFunction = EfficientNetB0), 
                                                      classifier2 = mlp, 
                                                      fineTuneLayersCount = 10, 
                                                      fineTuneEpochs = 1), 
                            storagePath = TestTrainingConstants.modelDirectory, 
                            provider = dataProvider)

        task.run(epochs = 2, earlyStoppingPatience = 10, optimize = "loss")


if __name__ == '__main__':
    unittest.main()
