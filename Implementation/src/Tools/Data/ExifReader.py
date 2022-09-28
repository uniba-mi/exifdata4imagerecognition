from abc import ABC, abstractmethod
import os
from DomainModel.ExifData import toInternalExifTagName
from Tools import PathHelpers
from Tools.Flickr.FlickrTools import readExifTags
import itertools

class ExifReader(ABC):
    """ An exif reader parses multiple files which contain exif tags, stored in a specific format. """

    @abstractmethod
    def readExifFiles(self, directoryPath: str) -> dict:
        """ Reads the exif data of all exif files in the given directory and returns the exif tags parsed in a dictionary 
        of dictionaries where the key of the outer dictionary is the id (filename) of the file and the inner
        dictionary contains the keys of the exif tags. """
        pass

class FlickrExifReader(ExifReader):
    """ Reads the exif data of json files that comply to the Flickr specification: 
    https://www.flickr.com/services/api/flickr.photos.getExif.htm """

    def readExifFiles(self, directoryPath: str) -> dict:
        if not os.path.isdir(directoryPath):
            raise FileExistsError("the given directory does not exist.")

        # get all json files in target directory
        jsonFiles = PathHelpers.matchingFilePaths(folderPath = directoryPath, extension = "json")

        exifData = {}

        # loop through all files and read exif tags
        for jsonFile in jsonFiles:
            imageId = os.path.splitext(os.path.basename(jsonFile))[0]
            exifTags = readExifTags(filePath = jsonFile)

            # convert the names of all tags to internal tag names
            cleanTags = {}

            if not exifTags == None:
                for exifTag in exifTags.keys():
                    tags = toInternalExifTagName(tagName = exifTag)
                    if not tags is None:
                        if isinstance(tags, tuple):
                            for tag in tags:
                                cleanTags[tag] = exifTags[exifTag]
                        else:
                            cleanTags[tags] = exifTags[exifTag]
                    else:
                        cleanTags[exifTag] = exifTags[exifTag]
                    
                if len(cleanTags.keys()) > 0:
                    exifData[imageId] = cleanTags

        return exifData

class MIRFLICKRExifReader(ExifReader):
    """ Reads the raw exif data of text files that comply to the MIRFLICKR specification: 
    https://press.liacs.nl/mirflickr/ """

    def readExifFiles(self, directoryPath: str) -> dict:
        if not os.path.isdir(directoryPath):
            raise FileExistsError("the given directory does not exist.")

        # get all text files in target directory
        textFiles = PathHelpers.matchingFilePaths(folderPath = directoryPath, extension = "txt")

        exifData = {}

        # loop through all files and read exif tags
        for file in textFiles:
            imageId = os.path.splitext(os.path.basename(file))[0].replace("exif", "")
            
            # the format of the text files is always tag / value for two lines
            exifTags = {}
            if os.path.getsize(file) > 0:
                with open(file, errors = "ignore") as f:
                    for tag, value in itertools.zip_longest(*[f]*2):
                        if not tag is None and not value is None:
                            tag = tag.replace("\n", "").replace("-", "").replace(" ", "")
                            internalTags = toInternalExifTagName(tagName = tag)
                            if not internalTags is None:
                                if isinstance(internalTags, tuple):
                                    for exifTagName in internalTags:
                                        exifTags[exifTagName] = value.strip()
                                else:
                                    if not internalTags in exifTags:
                                        exifTags[internalTags] = value.strip()
                            else:
                                exifTags[tag] = value.strip()

                if len(exifTags.keys()) > 0: 
                    exifData[imageId] = exifTags

        return exifData
