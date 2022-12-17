from pathlib import Path
from DataSource.TrainingDataSource import StaticTrainingDataSource, TrainingDataSource, CompositeTrainingDataSource
from DomainModel.ExifData import ExifData
from DomainModel.Image import Image
from Transformers.DataSource.DataSourceInstanceTransformer import DataSourceInstanceTransformer
from Transformers.DataSource.DataSourceToDataFrameTransformer import ExiImageDataSourceToDataFrameTransformer
from Transformers.Exif.ExifTagFilter import ExifTagFilter
from Transformers.Exif.ExifTagTransformer import ExifTagTransformer
from DataSource.DatasetFile import DatasetFile
from DomainModel.TrainingConcept import TrainingConcept
from DomainModel.TrainingData import EXIFImageTrainingData, TestTrainingData
from unittest.mock import Mock
import unittest
import numpy as np
from Transformers.Preprocessing.DataFrameLabelEncoder import DataFrameLabelEncoder
from Transformers.Preprocessing.DataFrameTrainingSplitter import DataFrameTrainingSplitter
from Transformers.Preprocessing.DataFramePreProcessor import TabularDataFramePreProcessor
from Transformers.Transformer import CompositeTransformer
from Tests.Training.TestTrainingConstants import TestTrainingConstants

# Note: To successfully run all data source tests, the indoor / outdoor data set must be present within the resource folder
datasetFilepath = TestTrainingConstants.indoorOutdoorFilePath

class DatasetFileTests(unittest.TestCase):

    datasetFile = DatasetFile(zipFilePath = datasetFilepath)

    def testDatasetFileExists(self):
        self.assertTrue(self.datasetFile.exists)

    def testDatasetFileExtractsCorrectly(self):
        self.datasetFile.removeDatasetDirecotry()
        self.assertFalse(self.datasetFile.isExtracted)
        self.datasetFile.unpack()
        self.assertTrue(self.datasetFile.isExtracted)

class DataSourceTests(unittest.TestCase):

    concept1 = TrainingConcept[TestTrainingData](names = ("ape"), superConceptNames = ("animal"))
    concept1.addTrainingInstances({TestTrainingData(id = "ape1"), TestTrainingData(id = "ape2")})

    concept2 = TrainingConcept[TestTrainingData](names = ("bird"), superConceptNames = ("animal"))
    concept2.addTrainingInstance(TestTrainingData(id = "bird1"))

    concept3 = TrainingConcept[TestTrainingData](names = ("ape"), superConceptNames = ("humanoid"))
    concept3.addTrainingInstance(TestTrainingData(id = "ape3"))

    concept4 = TrainingConcept[TestTrainingData](names = ("ape"), superConceptNames = ("animal"))
    concept4.addTrainingInstances({TestTrainingData(id = "ape4"), TestTrainingData(id = "ape5")})

    def ds1_retConcept(self, names):
        if names == ("ape"):
            return self.concept1
        return None

    def ds2_retConcept_Ok_Duplicates(self, names):
        if names == ("bird"):
            return self.concept2
        elif names == ("ape"):
            return self.concept1
        return None
    
    def ds2_retConcept_Ok_No_Duplicates(self, names):
        if names == ("bird"):
            return self.concept2
        elif names == ("ape"):
            return self.concept4
        return None
    
    def ds2_retConcept_Fail(self, names):
        if names == ("bird"):
            return self.concept2
        elif names == ("ape"):
            return self.concept3
        return None

    def testSameConceptsWithDifferentSuperConceptsFail(self):
        ds1 = Mock(spec = TrainingDataSource)
        ds1.getConceptNames.return_value = [self.concept1.names]
        ds1.getConcepts.return_value = [self.concept1]
        ds1.getConcept.side_effect = self.ds1_retConcept

        ds2 = Mock(spec = TrainingDataSource)
        ds2.getConceptNames.return_value = [self.concept2.names, self.concept3.names]
        ds2.getConcepts.return_value = [self.concept2, self.concept3]
        ds2.getConcept.side_effect = self.ds2_retConcept_Fail

        cds = CompositeTrainingDataSource()
        cds.addDataSource(ds1)
        cds.addDataSource(ds2)

        self.assertTrue(cds.getConceptNames() == {"ape", "bird"})
        self.assertRaises(ValueError, cds.getConcepts)
    
    def testSameConceptsWithSameSuperConceptsPassNoDuplicates(self):
        ds1 = Mock(spec = TrainingDataSource)
        ds1.getConceptNames.return_value = [self.concept1.names]
        ds1.getConcepts.return_value = [self.concept1]
        ds1.getConcept.side_effect = self.ds1_retConcept

        ds2 = Mock(spec = TrainingDataSource)
        ds2.getConceptNames.return_value = [self.concept2.names, self.concept1.names]
        ds2.getConcepts.return_value = [self.concept2, self.concept1]
        ds2.getConcept.side_effect = self.ds2_retConcept_Ok_No_Duplicates

        cds = CompositeTrainingDataSource()
        cds.addDataSource(ds1)
        cds.addDataSource(ds2)

        self.assertEqual({"animal"}, cds.getSuperConceptNames())
        self.assertEqual(4, len(cds.getConcept(names = ("ape")).instances))
        self.assertEqual(1, len(cds.getConcept(names = ("bird")).instances))
        self.assertEqual(2, len(cds.getConcepts()))
        self.assertTrue(cds.getConceptNames() == {"ape", "bird"})
    
    def testSameConceptsWithSameSuperConceptsPassWithDuplicates(self):
        ds1 = Mock(spec = TrainingDataSource)
        ds1.getConceptNames.return_value = [self.concept1.names]
        ds1.getConcepts.return_value = [self.concept1]
        ds1.getConcept.side_effect = self.ds1_retConcept

        ds2 = Mock(spec = TrainingDataSource)
        ds2.getConceptNames.return_value = [self.concept2.names, self.concept1.names]
        ds2.getConcepts.return_value = [self.concept2, self.concept1]
        ds2.getConcept.side_effect = self.ds2_retConcept_Ok_Duplicates

        cds = CompositeTrainingDataSource()
        cds.addDataSource(ds1)
        cds.addDataSource(ds2)

        self.assertEqual({"animal"}, cds.getSuperConceptNames())
        self.assertEqual(2, len(cds.getConcept(names = ("ape")).instances))
        self.assertEqual(1, len(cds.getConcept(names = ("bird")).instances))
        self.assertEqual(2, len(cds.getConcepts()))
        self.assertTrue(cds.getConceptNames() == {"ape", "bird"})
        
    def testStaticInMemoryTrainingDataSourceReturnsCorrectConceptsAndNames(self):
        dataSource = StaticTrainingDataSource(concepts = {self.concept1, self.concept2})
        self.assertEqual(2, len(dataSource.getConcepts()))
        self.assertEqual({"ape", "bird"}, dataSource.getConceptNames())
        self.assertEqual({"animal"}, dataSource.getSuperConceptNames())

    def testStaticInMemoryTrainingDataSourceThrowsErrorOnDuplicateConcepts(self):
        self.assertRaises(ValueError, StaticTrainingDataSource, {self.concept1, self.concept2, self.concept3, self.concept4})
    
    def testStaticInMemoryTrainingDataSourceFiltersConcepts(self):
        dataSource = StaticTrainingDataSource(concepts={self.concept1, self.concept2})
        self.assertEqual(2, len(dataSource.getConcepts()))
        dataSource.filter(filterFunction = lambda concept: concept.names == ("bird"))
        self.assertEqual(1, len(dataSource.getConcepts()))
        self.assertEqual({self.concept1}, dataSource.getConcepts())
    
    def testStaticTrainingDataSourceRaisesErrorOnReload(self):
        dataSource = StaticTrainingDataSource(concepts={self.concept1})
        self.assertRaises(RuntimeError, dataSource.reloadTrainingData)
    
class ExifImageDataSourceTests(unittest.TestCase):

    def setUp(self):
        self.testImage1 = Mock(spec = Image)
        self.testImage1.id = "12345"
        self.testImage1.path = Path("/test/path")

        exifData1 = ExifData(tags = {
            "ExposureTime" : "10",
            "FNumber" : "2.3",
            "FocalLength" : "4.23 mm",
            "Orientation" : "mirror vertical"
        }, id = "1")

        exifData2 = ExifData(tags = {
            "ExposureTime" : "5",
            "FNumber" : "1.8",
            "FocalLength" : "2.6 mm",
            "Orientation" : "mirror vertical",
            "Flash" : "1"
        }, id = "2")

        exifData3 = ExifData(tags = {
            "ExposureTime" : np.nan,
            "FNumber" : "2.8",
            "FocalLength" : "4.6 mm",
            "Orientation" : np.nan,
            "Flash" : "0"
        }, id = "3")

        trainingInstance1 = EXIFImageTrainingData(exifData = exifData1, image = self.testImage1)
        trainingInstance2 = EXIFImageTrainingData(exifData = exifData2, image = self.testImage1)
        trainingInstance3 = EXIFImageTrainingData(exifData = exifData3, image = self.testImage1)
        concept1 = TrainingConcept[EXIFImageTrainingData](names = "test1", superConceptNames = "super")
        concept1.addTrainingInstances({trainingInstance1}) 

        concept2 = TrainingConcept[EXIFImageTrainingData](names = "test2", superConceptNames = "super")
        concept2.addTrainingInstances({trainingInstance2, trainingInstance3})

        concept3 = TrainingConcept[EXIFImageTrainingData](names = ("test1", "test2"), superConceptNames = ("super1", "super2"))
        concept3.addTrainingInstances({trainingInstance1}) 

        concept4 = TrainingConcept[EXIFImageTrainingData](names = ("test2", "test3"), superConceptNames = ("super2", "super3"))
        concept4.addTrainingInstances({trainingInstance2, trainingInstance3}) 
        
        concept5 = TrainingConcept[EXIFImageTrainingData](names = ("test1",), superConceptNames = ("super2", "super3"))
        concept5.addTrainingInstances({trainingInstance1}) 

        self.dataSource = StaticTrainingDataSource(concepts = {concept1, concept2})
        self.multiLabelDataSource = StaticTrainingDataSource(concepts = {concept3, concept4, concept5})

    def testDataSourceToExifDataFrameTransformer(self):
        # test for individual concepts
        transformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = False)
        dataFrame = transformer.transform(data = self.dataSource)
        self.assertEqual(3, len(dataFrame))
        self.assertEqual(1, len(list(filter(lambda elem: elem == "test1", dataFrame["Target"].tolist()))))
        self.assertEqual(2, len(list(filter(lambda elem: elem == "test2", dataFrame["Target"].tolist()))))
        self.assertEqual(3, dataFrame.isna().sum().sum()) # filled values

        # test for super concepts
        transformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = True)
        dataFrame = transformer.transform(data = self.dataSource)
        self.assertEqual(3, len(dataFrame))
        self.assertEqual(3, len(list(filter(lambda elem: elem == "super", dataFrame["Target"].tolist()))))
        self.assertEqual(3, dataFrame.isna().sum().sum()) # filled values
    
    def testDataSourceToExifDataFrameTransformerWithMultipleLabels(self):
        # test for individual concepts
        transformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = False)
        dataFrame = transformer.transform(data = self.multiLabelDataSource)

        self.assertEqual(4, len(dataFrame))
        self.assertEqual(1, len(list(filter(lambda elem: elem == ("test1", "test2"), dataFrame["Target"].tolist()))))
        self.assertEqual(2, len(list(filter(lambda elem: elem == ("test2", "test3"), dataFrame["Target"].tolist()))))
        self.assertEqual(1, len(list(filter(lambda elem: elem == ("test1",), dataFrame["Target"].tolist()))))
        self.assertEqual(4, dataFrame.isna().sum().sum()) # filled values

        # test for super concepts
        transformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = True)
        dataFrame = transformer.transform(data = self.multiLabelDataSource)
        self.assertEqual(1, len(list(filter(lambda elem: elem == ("super1", "super2"), dataFrame["Target"].tolist()))))
        self.assertEqual(3, len(list(filter(lambda elem: elem == ("super2", "super3"), dataFrame["Target"].tolist()))))
        self.assertEqual(4, dataFrame.isna().sum().sum()) # filled values

    def testDataSourceInstanceTransformerForExifDataWithDummyValue(self):
        exifTagTransformer = ExifTagTransformer.standard()
        exifTagTransformer.addWhitelistedValue(np.nan)
        exifTagFilter = ExifTagFilter(filterTags = exifTagTransformer.transformedTags, dummyValue = np.nan)

        dataSourceTransformer = DataSourceInstanceTransformer[EXIFImageTrainingData](transformers = [exifTagFilter, exifTagTransformer], 
                                                              variableSelector = "exif")

        transformedDataSource = dataSourceTransformer.transform(self.dataSource)
        self.assertEqual(0, dataSourceTransformer.errorCount)
        self.assertEqual(3, len(list(transformedDataSource.allInstances)))

        for instance in transformedDataSource.allInstances:
            self.assertTrue(all(name != "Unknown" for name in instance.exif.tagNames))
            self.assertEqual(len(ExifTagTransformer.standard().transformedTags), len(instance.exif.tagNames))                   
       
    def testDataSourceInstanceTransformerForExifDataProducesErrorsWhenHandlingMissingTags(self):                          
        exifTagTransformer = ExifTagTransformer.standard()
        exifTagFilter = ExifTagFilter(filterTags = exifTagTransformer.transformedTags)

        dataSourceTransformer = DataSourceInstanceTransformer(transformers = [exifTagFilter, exifTagTransformer], 
                                                              variableSelector = "exif", verbose = 0)
        
        transformedDataSource = dataSourceTransformer.transform(self.dataSource)
        self.assertEqual(3, dataSourceTransformer.errorCount)
        self.assertEqual(0, len(list(transformedDataSource.allInstances)))
    
    def testExiDataFramePreProcessorImputesScalesAndOneHotEncodesValuesCorrectly(self):
        exifTagTransformer = ExifTagTransformer.standard()
        exifTagTransformer.addWhitelistedValue(np.nan)
        exifTagFilter = ExifTagFilter(filterTags = ["ExposureTime", "FNumber", "FocalLength", "Orientation", "Flash"], dummyValue = np.nan)

        dataSourceTransformer = DataSourceInstanceTransformer[EXIFImageTrainingData](transformers = [exifTagFilter, exifTagTransformer], 
                                                              variableSelector = "exif")

        transformedDataSource = dataSourceTransformer.transform(self.dataSource)
        dataFrameTransformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = False)

        dataFrame = dataFrameTransformer.transform(data = transformedDataSource)
        self.assertEqual(3, dataFrame.isna().sum().sum())

        preProcessor = TabularDataFramePreProcessor(numericalColumns = ["ExposureTime", "FNumber", "FocalLength"], 
                                                    categoricalColumns = ["Orientation"], 
                                                    binaryColumns = ["Flash"], scale = False)

        dataFrame = preProcessor.transform(data = dataFrame)    
        self.assertTrue(7.5 in dataFrame["ExposureTime"].values) # imputed mean value for exposure time
        self.assertTrue(dataFrame["Orientation_4.0"].unique() == [1]) # imputed most common value for orientation
        self.assertEqual(1, dataFrame["Flash_0.0"].value_counts()[0]) # one-hot encoded binary values for flash
        self.assertEqual(2, dataFrame["Flash_0.0"].value_counts()[1])
        self.assertEqual(2, dataFrame["Flash_1.0"].value_counts()[0])
        self.assertEqual(1, dataFrame["Flash_1.0"].value_counts()[1])
    
    def testExifDataFrameTrainingSplitterSplitsValuesCorrectly(self):
        exifData = ExifData(tags = { "ExposureTime" : "10" })

        # add one additional element to the set to have an equal size (4 elements)
        trainingInstance = EXIFImageTrainingData(exifData = exifData, image = self.testImage1)
        concept = next(iter(self.dataSource.concepts))
        concept.addTrainingInstance(trainingInstance)

        dataFrameTransformer = ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target")
        dataFrame = dataFrameTransformer.transform(data = self.dataSource)

        trainX, trainY, valX, valY, testX, testY = DataFrameTrainingSplitter(targetLabel = "Target", 
                                                                             trainRatio = 0.5, 
                                                                             validationRatio = 0.25, 
                                                                             testRatio = 0.25,
                                                                             stratify = False).transform(data = dataFrame)
        self.assertEqual(2, len(trainX))
        self.assertEqual(2, len(trainY))   
        self.assertEqual(1, len(valX))  
        self.assertEqual(1, len(valY))         
        self.assertEqual(1, len(testX))
        self.assertEqual(1, len(testY))             

        trainX, trainY, valX, valY, testX, testY = DataFrameTrainingSplitter(targetLabel = "Target", 
                                                                             trainRatio = 0.25, 
                                                                             validationRatio = 0.25, 
                                                                             testRatio = 0.5,
                                                                             stratify = False).transform(data = dataFrame)                                                       
        self.assertEqual(1, len(trainX))
        self.assertEqual(1, len(trainY))            
        self.assertEqual(1, len(valX))  
        self.assertEqual(1, len(valY))
        self.assertEqual(2, len(testX))
        self.assertEqual(2, len(testY))   
    
    def testDataFrameLabelEncoderWithSingleLabels(self):
        encoder = DataFrameLabelEncoder(targetLabel = "Target", encodedLabel = "Target_Encoded")
        transformer = CompositeTransformer(transformers = [ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = False),
                                                           encoder])

        # test for single column
        dataFrame = transformer.transform(self.dataSource)
        self.assertEqual(3, len(dataFrame))
        self.assertEqual([1, 0], dataFrame.loc[dataFrame["Target"] == "test1"].iloc[0]["Target_Encoded"])
        self.assertEqual([0, 1], dataFrame.loc[dataFrame["Target"] == "test2"].iloc[0]["Target_Encoded"])
        self.assertEqual([0, 1], dataFrame.loc[dataFrame["Target"] == "test2"].iloc[1]["Target_Encoded"])

        # test inverse transform
        reversed = encoder.labelEncoder.inverse_transform(np.array([[1, 0], [0, 1], [0, 1]]))
        self.assertEqual([("test1",), ("test2",), ("test2",)], reversed)
    
    def testDataFrameLabelEncoderWithMultipleLabels(self):
        encoder = DataFrameLabelEncoder(targetLabel = "Target", encodedLabel = "Target_Encoded")
        transformer = CompositeTransformer(transformers = [ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = False),
                                                           encoder])

        # test for single column
        dataFrame = transformer.transform(self.multiLabelDataSource)
        self.assertEqual(4, len(dataFrame))
        self.assertEqual([1, 0, 0], dataFrame.loc[dataFrame["Target"] == ("test1",)].iloc[0]["Target_Encoded"])
        self.assertEqual([0, 1, 1], dataFrame.loc[dataFrame["Target"] == ("test2", "test3")].iloc[0]["Target_Encoded"])
        self.assertEqual([0, 1, 1], dataFrame.loc[dataFrame["Target"] == ("test2", "test3")].iloc[1]["Target_Encoded"])
        self.assertEqual([1, 1, 0], dataFrame.loc[dataFrame["Target"] == ("test1", "test2")].iloc[0]["Target_Encoded"])

        # test inverse transform
        reversed = encoder.labelEncoder.inverse_transform(np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]))
        self.assertEqual([("test1",), ("test2", "test3"), ("test2",)], reversed)


if __name__ == '__main__':
    unittest.main()
    