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
        
        # create training task
        trainingTask = TrainingTask(classifier = MLPClassifier(name = "MLP"), 
                                    storagePath = None,
                                    provider = dataProvider)
        trainingTask.run(epochs = 300, earlyStoppingPatience = 50, optimize = "loss")

if __name__ == '__main__':
    unittest.main()
