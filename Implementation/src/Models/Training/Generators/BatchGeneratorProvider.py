from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import csv
from typing import Dict, List, Tuple
from Models.Training.Generators.BatchGenerators import BatchGenerator
from Tools.Data.DataFrameHelpers import writeCSVDataFrame, readCSVDataFrame
from Tools.Data.ExifTagCounter import ExifTagCounter
from DataSource.DatasetFile import ExifImageDatasetFile
from DataSource.TrainingDataSource import CompositeTrainingDataSource, StaticTrainingDataSource
from DataSource.DatasetFile import ExifImageDatasetFile
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats
from Transformers.DataSource.DataSourceInstanceTransformer import DataSourceInstanceTransformer
from Transformers.DataSource.DataSourceToDataFrameTransformer import ExiImageDataSourceToDataFrameTransformer
from Transformers.Exif.ExifTagFilter import ExifTagFilter
from Transformers.Exif.ExifTagTransformer import ExifTagTransformer
from Transformers.Preprocessing.DataFrameLabelEncoder import DataFrameLabelEncoder
from Transformers.Preprocessing.DataFramePreProcessor import TabularDataFramePreProcessor
from Transformers.Preprocessing.DataFrameTrainingSplitter import DataFrameTrainingSplitter
from Transformers.Transformer import CompositeTransformer
from Models.Training.Generators.BatchGenerators import EXIFGenerator, EXIFImageGenerator, ImageGenerator

class BatchGeneratorProvider(ABC):
    """ Provides a training / test / validation batch data generator for training ML models. The generator
    shall automatically determine the input / output shape of the used training data, as well as the target class names. """

    trainSize: float
    testSize: float
    validationSize: float
    inputShape: Tuple
    outputShape: Tuple
    classes: List[str]
    seed: int
    batchSize: int
    multiLabel: bool
    cachedGenerators: Tuple[BatchGenerator]

    @abstractmethod
    def createGenerators(self, verbose: int = 1) -> Tuple[BatchGenerator, BatchGenerator, BatchGenerator]:
        pass

class ExifImageProvider(BatchGeneratorProvider):
    """ A exif/image batch data generator provider creates batch generators from a exif / image data set file. The
    provider will cache (store / load) the created pre-processed data frames on the hard disk, if a cache path is provided.
    The data sets of the created generators will include a column named 'ImagePath', if exifOnly is set to false. """

    datasetsFilePaths: Dict[str, ExifImageTrainingDataFormats]
    cachePath: Path
    imageSize: Tuple
    useSuperConcepts: bool
    filterConcepts: List[str]
    exifOnly: bool
    imageOnly: bool
    exifTags: List[str]
    exifTagTransformer: ExifTagTransformer
    trainingDataFormat: ExifImageTrainingDataFormats
    
    def __init__(self, dataSetsPaths: Dict[str, ExifImageTrainingDataFormats],
                       cachePath: str, 
                       trainSize: float,
                       testSize: float,
                       validationSize: float,
                       batchSize: int = 32,
                       imageSize: Tuple = (150, 150),
                       useSuperConcepts: bool = False,
                       filterConcepts: List[str] = [], # if set, the given concepts will be ignored
                       exifOnly: bool = False,
                       imageOnly: bool = False,
                       exifTags: List[str] = [], #if set, only these exif tags will be used for training
                       seed: int = 131):
        self.datasetsFilePaths = dataSetsPaths
        self.cachePath = Path(cachePath)
        self.trainSize = trainSize
        self.testSize = testSize
        self.validationSize = validationSize
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.useSuperConcepts = useSuperConcepts
        self.filterConcepts = filterConcepts
        self.exifOnly = exifOnly
        self.imageOnly = imageOnly
        self.seed = seed
        self.inputShape = None
        self.outputShape = None
        self.classes = []
        self.multiLabel = False
        self.cachedGenerators = None
        self.useStandardTags = False

        # create exif tag transformer for the generator provider
        self.exifTagTransformer = ExifTagTransformer.standard()
        self.exifTagTransformer.insertNanOnTransformError = True
        self.exifTagTransformer.addWhitelistedValue(np.nan) 

        # we use the pre-defined list of exif tags or all available tags there is a transformer function for
        if len(exifTags) > 0:
            self.exifTags = sorted(exifTags)
        else:
            self.exifTags = self.exifTagTransformer.transformedTags
            self.useStandardTags = True

        if exifOnly and imageOnly:
            raise ValueError("either exif only or image only can be set to true, but not both at the same time.")
    
    @property
    def cachedCSVFileName(self) -> str:
        """ Returns the csv file name of the cached data frame of this generator provider configuration. """
        return "{name}_cache_{super}_{filter}_{tags}.csv".format(name = "_".join(map(lambda path: Path(path).stem, self.datasetsFilePaths.keys())),
                                                   super = "super" if self.useSuperConcepts else "normal",
                                                   filter = "_".join(self.filterConcepts) if len(self.filterConcepts) > 0 else "allconcepts",
                                                   tags = "_".join(self.exifTags) if not self.useStandardTags else "alltags")
    
    @property
    def cachedExifCountCSVFileName(self) -> str:
        """ Returns the csv file name of the file containing the exif tag distribution of this generator provider configuration. """
        return self.cachedCSVFileName.replace(".csv", "_tag_distribution.csv")
        
    def createGenerators(self, verbose: int = 1) -> Tuple[BatchGenerator, BatchGenerator, BatchGenerator]:
        if verbose > 0:
            print("try to load cached dataframe ...")

        # we try to load the pre-processed dataframes if they are already created
        if not self.cachePath is None:
            cachedFilePath = self.cachePath.joinpath(self.cachedCSVFileName)
            dataFrame = readCSVDataFrame(cachedFilePath, converters = { "Target_Encoded" : lambda string: list(map(float, string.replace("[", "").replace("]", "").split(','))),
                                                                        "Target" : lambda string: tuple(string.replace("(", "").replace(")", "").replace("'", "").replace(" ", "").split(',')) if "," in string else string })

        if dataFrame is None:
            if verbose > 0:
                print("dataframe not found on hard disk, creating from data set(s) ...")

            # !important note!: if the pre-processing parameters (e.g. concept names), or the data of the exif / image data 
            # set changes, the stored dataframe must be re-created and thus first deleted from the hard disk manually.

            # we use the exif / image data set file to create the data frame (to don't have to parse the data every time)
            print("loading data set(s) ...")

            # create a composite data source
            dataSource = CompositeTrainingDataSource()

            for dataSetFilePath, dataFormat in self.datasetsFilePaths.items(): 
                filePath = Path(dataSetFilePath)
                dataSetFile = ExifImageDatasetFile(filePath, 
                                                   exifReader = dataFormat.value.exifReader, 
                                                   imagePathReader = dataFormat.value.imagePathReader)
                # unpack data set if needed
                if not dataSetFile.isExtracted:
                    if verbose > 0:
                        print("extracting data set file " + filePath.stem + " ...")
                    dataSetFile.unpack()
                
                dataSource.addDataSource(dataSource = dataSetFile)
            
            # reload composite data source (this will reload all attached data sources)
            dataSource.reloadTrainingData()

            if verbose > 0:
                print("data set loaded ...")
                print("preprocessing training data ...")

            # filter out unwanted classes
            dataSource = StaticTrainingDataSource(concepts = dataSource.getConcepts())
            dataSource.filter(filterFunction = lambda concept: concept.names in self.filterConcepts)

            # count the distribution of exif tags in the training data and store it to cache directory
            if not self.cachePath is None:
                counter = ExifTagCounter(dataSource = dataSource)
                with open(self.cachePath.joinpath(self.cachedExifCountCSVFileName), "w") as file:
                    csvWriter = csv.writer(file)
                    csvWriter.writerows(counter.count().most_common())

            # create typed tag lists for the exif tags
            numerical, categorical, binary = ExifTagTransformer.typedTagLists(exifTags = self.exifTags, transformer = self.exifTagTransformer)

            # create dict which contains the number of possible values for each categorical feature
            featureValueCounts = {}
            for feature in categorical + binary:
                featureValueCounts[feature] = self.exifTagTransformer.transformerFunctions[feature].valueCount

            # create exif tag filter
            exifTagFilter = ExifTagFilter(filterTags = self.exifTags, dummyValue = np.nan)

            # create pipeline for pre-processing the data of the data source
            transformers = [DataSourceInstanceTransformer(transformers = [exifTagFilter, self.exifTagTransformer], variableSelector = "exif"),
                            ExiImageDataSourceToDataFrameTransformer(targetLabel = "Target", useSuperConceptNames = self.useSuperConcepts, imagePathColumnName = "ImagePath", idColumnName = "id"),
                            TabularDataFramePreProcessor(numericalColumns = numerical, categoricalColumns = categorical, binaryColumns = binary, categoricalFeatureCounts = featureValueCounts),
                            DataFrameLabelEncoder(targetLabel = "Target", dropTargetColumn = False, encodedLabel = "Target_Encoded")]

            # crate dataframe
            dataFrame = CompositeTransformer(transformers = transformers).transform(data = dataSource)

            # write data frame to csv file to reload it later
            if not self.cachePath is None:
                if verbose > 0:
                    print("saving pre-processed dataframe to csv file " + str(cachedFilePath) + " ...")
                writeCSVDataFrame(dataFrame = dataFrame, filePath = cachedFilePath)
        else:
            if verbose > 0:
                print("successfully loaded dataframe from file ...")
        
        # print dataframe size
        if verbose > 0:
            print("the dataframe contains " + str(len(dataFrame)) + " elements ...")

        # load class names
        labels = dataFrame["Target"].unique()
        if any(isinstance(label, tuple) for label in labels):
            # read multi-labels
            self.multiLabel = True
            self.classes = sorted(tuple(set(label for sub in labels for label in sub if label != "" and label != " ")))
        else:
            self.classes = sorted(labels)

        dataFrame.drop("Target", axis = 1, inplace = True)

        # split in train / test / val set
        splitter = DataFrameTrainingSplitter(targetLabel = "Target_Encoded", 
                                                trainRatio = self.trainSize,
                                                testRatio = self.testSize,
                                                validationRatio = self.validationSize, 
                                                seed = self.seed,
                                                dropColumns = ["ImagePath"] if self.exifOnly else [])
        
        trainX, trainY, testX, testY, valX, valY = splitter.transform(data = dataFrame)

        # replace the id columns from the dataframes and store them separately
        idTrain = trainX["id"]
        trainX.drop("id", axis = 1, inplace = True)

        if not testX is None:
            idTest = testX["id"]
            testX.drop("id", axis = 1, inplace = True)

        if not valX is None:
            idVal = valX["id"]
            valX.drop("id", axis = 1, inplace = True)
        
        self.outputShape = len(dataFrame.iloc[0]["Target_Encoded"])

        if self.exifOnly:
            self.inputShape = trainX.shape[1]
            self.cachedGenerators = (EXIFGenerator(trainX, trainY, idTrain, batchSize = self.batchSize),
                                     EXIFGenerator(testX, testY, idTest, batchSize = self.batchSize) if not testX is None and not testY is None else None,
                                     EXIFGenerator(valX, valY, idVal, batchSize = self.batchSize) if not valX is None and not valY is None else None)
            return self.cachedGenerators
        elif self.imageOnly:
            self.inputShape = (self.imageSize[0], self.imageSize[1], 3)
            self.cachedGenerators = (ImageGenerator(trainX, trainY, idTrain, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize),
                                     ImageGenerator(testX, testY, idTest, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize) if not testX is None and not testY is None else None,
                                     ImageGenerator(valX, valY, idVal, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize) if not valX is None and not valY is None else None)
            return self.cachedGenerators
        else:
            self.inputShape = ((self.imageSize[0], self.imageSize[1], 3), trainX.shape[1] - 1)
            self.cachedGenerators = (EXIFImageGenerator(trainX, trainY, idTrain, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize),
                                     EXIFImageGenerator(testX, testY, idTest, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize) if not testX is None and not testY is None else None,
                                     EXIFImageGenerator(valX, valY, idVal, imagePathColumn = "ImagePath", imageSize = self.imageSize, batchSize = self.batchSize) if not valX is None and not valY is None else None)
            return self.cachedGenerators
