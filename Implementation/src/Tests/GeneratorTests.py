import unittest
import numpy as np
from DataSource.TrainingDataSource import StaticTrainingDataSource
from DomainModel.TrainingConcept import TrainingConcept
from DomainModel.TrainingData import EXIFImageTrainingData
from Models.Training.Generators.BatchGenerators import EXIFGenerator, EXIFImageGenerator, ImageGenerator
from Transformers.DataSource.DataSourceInstanceTransformer import DataSourceInstanceTransformer
from Transformers.DataSource.DataSourceToDataFrameTransformer import ExiImageDataSourceToDataFrameTransformer
from Transformers.Exif.ExifTagFilter import ExifTagFilter
from Transformers.Exif.ExifTagTransformer import ExifTagTransformer
from Transformers.Preprocessing.DataFrameLabelEncoder import DataFrameLabelEncoder
from Transformers.Preprocessing.DataFrameTrainingSplitter import DataFrameTrainingSplitter
from Transformers.Preprocessing.DataFramePreProcessor import TabularDataFramePreProcessor
from Transformers.Transformer import CompositeTransformer
from DomainModel.ExifData import ExifData
from DomainModel.Image import Image

class GeneratorTests(unittest.TestCase):

    # Note: To successfully run all generator tests, the mi indoor / outdoor data set must be extracted within the resource folder

    imageFolderPath = "resources/mi_indoor_outdoor_multilabel/indoor/indoor,bathroom/images/"

    @classmethod
    def setUpClass(self):
        self.exifData1 = ExifData(tags = {
            "ExposureTime" : "10",
            "FNumber" : "2.3",
            "FocalLength" : "4.23 mm",
            "Orientation" : "mirror vertical"
        }, id = "1")

        self.exifData2 = ExifData(tags = {
            "ExposureTime" : "5",
            "FNumber" : "1.8",
            "FocalLength" : "2.6 mm",
            "Orientation" : "mirror vertical",
            "Flash" : "1"
        }, id = "2")

        self.exifData3 = ExifData(tags = {
            "ExposureTime" : np.nan,
            "FNumber" : "2.8",
            "FocalLength" : "4.6 mm",
            "Orientation" : np.nan,
            "Flash" : "0"
        }, id = "3")

        trainingInstance1 = EXIFImageTrainingData(exifData = self.exifData1, 
            image = Image(path = self.imageFolderPath + "67892_4f5f44cf68_q.jpg", id = "67892"))
        trainingInstance2 = EXIFImageTrainingData(exifData = self.exifData2, 
            image = Image(path = self.imageFolderPath + "14697868_3c47ab352f_q.jpg", id = "14697868"))
        trainingInstance3 = EXIFImageTrainingData(exifData = self.exifData3, 
            image = Image(path = self.imageFolderPath + "26873013_03578980b1_q.jpg", id = "26873013"))
        trainingInstance4 = EXIFImageTrainingData(exifData = self.exifData1, 
            image = Image(path = self.imageFolderPath + "27404465_c170a5b7bf_q.jpg", id = "27404465"))
        trainingInstance5 = EXIFImageTrainingData(exifData = self.exifData2, 
            image = Image(path = self.imageFolderPath + "39142989_89145bce04_q.jpg", id = "39142989"))
        trainingInstance6 = EXIFImageTrainingData(exifData = self.exifData3, 
            image = Image(path = self.imageFolderPath + "51913452_310ad5aeab_q.jpg", id = "51913452"))

        concept1 = TrainingConcept[EXIFImageTrainingData](names = "test1")
        concept1.addTrainingInstances({trainingInstance1, trainingInstance2, trainingInstance3}) 
        concept2 = TrainingConcept[EXIFImageTrainingData](names = "test2")
        concept2.addTrainingInstances({trainingInstance4, trainingInstance5, trainingInstance6})
        dataSource = StaticTrainingDataSource(concepts = {concept1, concept2})

        # create transformers for transforming exif data
        exifTagTransformer = ExifTagTransformer.standard()
        exifTagTransformer.addWhitelistedValue(np.nan)
        exifTagFilter = ExifTagFilter(filterTags = ["ExposureTime", "FNumber", "FocalLength", "Orientation", "Flash"], dummyValue = np.nan)

        # create pipeline for pre-processing the data of the data source
        transformers = [DataSourceInstanceTransformer(transformers = [exifTagFilter, exifTagTransformer], variableSelector = "exif"),
                        ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target"),
                        TabularDataFramePreProcessor(numericalColumns = ["ExposureTime", "FNumber", "FocalLength"], categoricalColumns = ["Orientation"], binaryColumns = ["Flash"]),
                        DataFrameLabelEncoder(targetLabel = "Target", dropTargetColumn = True, encodedLabel = "Target_Encoded"),
                        DataFrameTrainingSplitter(targetLabel = "Target_Encoded", stratify = False)]

        transformer = CompositeTransformer(transformers = transformers)

        self.trainX, self.trainY, self.testX, self.testY, self.valX, self.valY = transformer.transform(data = dataSource)
    
        self.idTrain = self.trainX["id"]
        self.trainX.drop("id", axis = 1, inplace = True)


    def testEXIFDataGeneratorWithSingleLabels(self):
        generator = EXIFGenerator(xDataFrame = self.trainX, 
                                  yDataFrame = self.trainY, 
                                  idDataFrame = self.idTrain,
                                  ignoreColumns = "ImagePath", 
                                  batchSize = 2)

        x, y = generator.__getitem__(index = 1)
        self.assertEqual(2, len(x)) # exif data
        self.assertEqual(6, len(x[0])) # according to the test data specified above, the exif data must contain 6 columns
        self.assertEqual(6, len(x[1]))
        self.assertEqual(2, len(y)) # label data
        self.assertEqual(2, len(y[0])) # two classes one-hot encoded
        self.assertEqual(2, len(y[1])) # two classes one-hot encoded

    def testEXIFImageDataGeneratorWithSingleLabels(self):
        generator = EXIFImageGenerator(xDataFrame = self.trainX, 
                                       yDataFrame = self.trainY, 
                                       idDataFrame = self.idTrain,
                                       imagePathColumn = "ImagePath", 
                                       imageSize = (128, 128),
                                       batchSize = 2)
        
        x, y = generator.__getitem__(index = 1)
        self.assertEqual(2, len(x[0])) # image data
        self.assertEqual(2, len(x[1])) # exif data
        self.assertEqual(6, len(x[1][0]))
        self.assertEqual(6, len(x[1][1]))
        self.assertEqual(2, len(y[0])) # label data
        self.assertEqual((128, 128, 3), x[0][0].shape)
        self.assertEqual((128, 128, 3), x[0][1].shape)
    
    def testImageDataGeneratorWithSingleLabels(self):
        generator = ImageGenerator(xDataFrame = self.trainX, 
                                   yDataFrame = self.trainY, 
                                   idDataFrame = self.idTrain,
                                   imagePathColumn = "ImagePath", 
                                   imageSize = (128, 128),
                                   batchSize = 2)
        
        x, y = generator.__getitem__(index = 1)
        self.assertEqual(2, len(x)) # image data
        self.assertEqual((128, 128, 3), x[0].shape)
        self.assertEqual((128, 128, 3), x[1].shape)
        self.assertEqual(2, len(y)) # label data
        self.assertEqual(2, len(y[0])) # two classes one-hot encoded
        self.assertEqual(2, len(y[1]))

    def testGeneratorReturnsAllElements(self):
        generator = EXIFGenerator(xDataFrame = self.trainX, 
                                  yDataFrame = self.trainY, 
                                  idDataFrame = self.idTrain,
                                  ignoreColumns = "ImagePath", 
                                  batchSize = 4)

        x, y = generator.__getitem__(index = 0)
        self.assertEqual(4, len(x))
        self.assertEqual(4, len(y))


if __name__ == '__main__':
    unittest.main()
    