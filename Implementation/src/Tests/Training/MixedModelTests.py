import unittest
import tensorflow
from Models.Classifiers.MLP import MLPClassifier
from Models.Classifiers.CNN import CNNClassifier
from Models.Classifiers.MixedModel import MixedModel
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.TrainingTask import TrainingTask
from Tests.Training.TestTrainingConstants import TestTrainingConstants
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats

class MLPTrainingTests(unittest.TestCase):

    def testTrainIndoorOutdoorMixedModel(self):
        tensorflow.get_logger().setLevel("ERROR")
        
        dataSetPaths = { TestTrainingConstants.movingStaticFilePath : ExifImageTrainingDataFormats.Flickr }

        dataProvider = ExifImageProvider(dataSetsPaths = dataSetPaths,
                                        cachePath = TestTrainingConstants.cacheStoragePath,
                                        trainSize = 0.7,
                                        validationSize = 0.2,
                                        testSize = 0.1,
                                        batchSize = 32,
                                        exifOnly = False,
                                        useSuperConcepts = True,
                                        imageSize = (150, 150),
                                        seed = 131)
                                        
        task = TrainingTask(classifier = MixedModel(name = "MixedMobileNetV2MovingStatic", 
                                                      classifier1 = CNNClassifier(name = "CNN", modelFunction = MobileNetV2), 
                                                      classifier2 = MLPClassifier(name = "MLP"), 
                                                      fineTuneLayersCount = 10, 
                                                      fineTuneEpochs = 1), 
                            storagePath = TestTrainingConstants.modelDirectory, 
                            provider = dataProvider)

        task.run(epochs = 2, earlyStoppingPatience = 10, optimize = "loss")


if __name__ == '__main__':
    unittest.main()
