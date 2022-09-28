import zipfile
import shutil
from pathlib import Path
from typing import Set
import os
from DomainModel.TrainingData import EXIFImageTrainingData
from DomainModel.ExifData import ExifData
from DomainModel.Image import Image
from DomainModel.TrainingConcept import TrainingConcept
from DataSource.TrainingDataSource import TrainingDataSource
from Tools.Data.ExifReader import ExifReader
from Tools.Data.ImagePathReader import ImagePathReader

class DatasetFile(object):
    """ The class DatasetFile represents a single .zip file which contains training data for
    various training concepts. The directory structure within the zip file shall represent the structure of 
    the contained concepts, e.g.   [dataSet.zip] -> [indoor]    -> [kitchen]    -> [data]
                                                                -> [bedroom]    -> ...
                                                                -> [...]
                                                    [outdoor]   -> [beach]      -> [data]
                                                                -> [river]      -> ...
                                                                -> [...]
    """
    zipFilePath: Path

    def __init__(self, zipFilePath: str):
        path = Path(zipFilePath).resolve()
        if not path.exists():
            raise ValueError("There is no file at the given path.")
        elif not zipfile.is_zipfile(zipFilePath):
            raise ValueError("The given file is not a zip file.")
        else:
            self.zipFilePath = path

    @property
    def exists(self) -> bool:
        return self.zipFilePath.exists()

    @property
    def fileName(self) -> str:
        return self.zipFilePath.name

    @property
    def datasetDirectory(self) -> Path:
        return self.zipFilePath.parent.resolve().joinpath(self.zipFilePath.stem).resolve()

    @property
    def isExtracted(self) -> bool:
        return self.datasetDirectory.exists()

    def unpack(self):
        if not self.isExtracted:
            shutil.unpack_archive(self.zipFilePath, self.zipFilePath.parent.resolve())
        else:
            print("Dataset already unpacked")
    
    def removeDatasetDirecotry(self):
        if self.isExtracted:
            shutil.rmtree(self.datasetDirectory)

class ExifImageDatasetFile(DatasetFile, TrainingDataSource[EXIFImageTrainingData]):
    """ The class ExifImageDatasetFile represents a single .zip file which contains training (exif / image) data for
    various training concepts. The directory structure within the zip file shall represent the structure of 
    the contained concepts, e.g.   [dataSet.zip] -> [indoor]    -> [kitchen]    -> [image] [exif]
                                                                -> [bedroom]    -> ...
                                                                -> [...]
                                                    [outdoor]   -> [beach]      -> [image] [exif]
                                                                -> [river]      -> ...
                                                                -> [...]
    Multiple concept names must be split by ','

    The given exif-reader is used to parse the files which contain the exif tags.
    The given image-reader is used to parse the image files.
    """
                                                            
    __concepts: Set[TrainingConcept[EXIFImageTrainingData]]
    exifReader: ExifReader
    imagePathReader: ImagePathReader

    def __init__(self, zipFilePath: str, exifReader: ExifReader, imagePathReader: ImagePathReader):
        self.__concepts = set()
        super().__init__(zipFilePath = zipFilePath)
        self.exifReader = exifReader
        self.imagePathReader = imagePathReader
    
    def getConceptNames(self) -> Set[str]:
        return { concept.names for concept in self.__concepts }
    
    def getConcepts(self) -> Set[TrainingConcept[EXIFImageTrainingData]]:
        return self.__concepts
    
    def reloadTrainingData(self):
        concepts = set()

        if self.exists and self.isExtracted:
            # first get top-level folders representing top-level concepts
            topLevelFolders = []
            [topLevelFolders.append(folder) for folder in os.scandir(self.datasetDirectory) if folder.is_dir()]
            
            for topLevelFolder in topLevelFolders:
                # get sub-level folders for each top-level folder and create training concepts
                subLevelFolders = []
                [subLevelFolders.append(folder) for folder in os.scandir(topLevelFolder.path) if folder.is_dir()]

                for subLevelFolder in subLevelFolders:
                    # multiple concepts are seperated by ','
                    conceptNames = tuple(subLevelFolder.name.split(",")) if "," in subLevelFolder.name else subLevelFolder.name
                    superConceptNames = tuple(topLevelFolder.name.split(",")) if "," in topLevelFolder.name else topLevelFolder.name
                    concept = TrainingConcept[EXIFImageTrainingData](names = conceptNames, superConceptNames = superConceptNames)
                    
                    # read the exif data for the sub-level concept
                    rawExifData = self.exifReader.readExifFiles(directoryPath = os.path.join(subLevelFolder.path, "exif"))

                    # read the image paths for the sub-level concept
                    imageFilePaths = self.imagePathReader.readImageFiles(directoryPath = os.path.join(subLevelFolder.path, "images")) 
                    
                    # we now match the read exif data to the coresponding image
                    for key in rawExifData.keys():
                        if imageFilePaths[key]:
                            exifData = ExifData(tags = rawExifData[key], id = key)
                            imageData = Image(path = imageFilePaths[key], id = key)
                            trainingInstance = EXIFImageTrainingData(exifData = exifData, image = imageData)
                            concept.addTrainingInstance(trainingInstance)
                
                    # add concept to concepts
                    concepts.add(concept)

            self.__concepts = concepts
            