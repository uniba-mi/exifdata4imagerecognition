import unittest
import tensorflow
from Models.Classifiers.CNN import CNNClassifier
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.TrainingTask import TrainingTask
from Tests.Training.TestTrainingConstants import TestTrainingConstants
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats

class CNNTrainingTests(unittest.TestCase):

    def testTrainIndoorOutdoorCNN(self):
        tensorflow.get_logger().setLevel("ERROR")
        #dataSetPaths = { TestTrainingConstants.datasetFilePath : ExifImageTrainingDataFormats.Flickr,
        #                 TestTrainingConstants.mirDatasetFilePath : ExifImageTrainingDataFormats.MirFlickr }

        dataSetPaths = { TestTrainingConstants.landscapesObjectsFilePath : ExifImageTrainingDataFormats.Flickr }

        dataProvider = ExifImageProvider(dataSetsPaths = dataSetPaths,
                                        cachePath = TestTrainingConstants.cacheStoragePath,
                                        trainSize = 0.7,
                                        validationSize = 0.29,
                                        testSize = 0.01,
                                        batchSize = 32,
                                        useSuperConcepts = False,
                                        imageSize = (150, 150),
                                        imageOnly = True,
                                        seed = 131)

        task = TrainingTask(classifier = CNNClassifier(name = "CNNMobileNetV2IndoorOutdoor",
                                                         modelFunction = MobileNetV2,
                                                         fineTuneLayersCount = 10,
                                                         fineTuneEpochs = 1),
                            storagePath = TestTrainingConstants.modelDirectory, 
                            provider = dataProvider)

        task.run(epochs = 2, earlyStoppingPatience = 10, optimize = "loss")

if __name__ == '__main__':
    unittest.main()
