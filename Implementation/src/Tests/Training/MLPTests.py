import unittest
from Models.Classifiers.MLP import MLPClassifier
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.PermutationTask import ExifPermutationTask
from Models.Training.Tasks.TrainingTask import TrainingTask
from Tests.Training.TestTrainingConstants import TestTrainingConstants
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats

class MLPTrainingTests(unittest.TestCase):

    def testTrainIndoorOutdoorExifOnly(self):
        #dataSetPaths = { TestTrainingConstants.datasetFilePath : ExifImageTrainingDataFormats.Flickr,
        #                 TestTrainingConstants.mirFlickrRelevantFilePath : ExifImageTrainingDataFormats.MirFlickr }

        dataSetPaths = { TestTrainingConstants.landscapesObjectsFilePath : ExifImageTrainingDataFormats.Flickr }

        dataProvider = ExifImageProvider(dataSetsPaths = dataSetPaths,
                                        cachePath = TestTrainingConstants.cacheStoragePath,
                                        trainSize = 0.7,
                                        validationSize = 0.2,
                                        testSize = 0.1,
                                        batchSize = 128,
                                        useSuperConcepts = False,
                                        exifOnly = True,
                                        seed = 46)
        
        # create training task
        trainingTask = TrainingTask(classifier = MLPClassifier(name = "MovingStatic"), 
                                    storagePath = TestTrainingConstants.modelDirectory, 
                                    provider = dataProvider)
        trainingTask.run(epochs = 10, earlyStoppingPatience = 50, optimize = "loss")

        # create permutation task to assess the feature importance of individual exif features
        permutationTask = ExifPermutationTask(permutations = 1, storagePath = TestTrainingConstants.modelDirectory)
        permutationTask.run(classifier = trainingTask.classifier, dataGeneratorProvider = dataProvider)


if __name__ == '__main__':
    unittest.main()
