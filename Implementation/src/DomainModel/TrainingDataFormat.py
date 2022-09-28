from enum import Enum
from Tools.Data.ExifReader import ExifReader, FlickrExifReader, MIRFLICKRExifReader
from Tools.Data.ImagePathReader import FlickrImagePathReader, FlickrImagePathReader, ImagePathReader, MIRFLICKRImagePathReader


class ExifImageTrainingDataFormat(object):

    name: str
    exifReader: ExifReader
    imagePathReader: ImagePathReader

    def __init__(self, exifReader: ExifReader, imagePathReader: ImagePathReader, name: str):
        self.name = name
        self.exifReader = exifReader
        self.imagePathReader = imagePathReader


CONST_FLICKR_DATA_FORMAT_NAME = "flickr"
CONST_MIRFLICKR_DATA_FORMAT_NAME = "mirflickr"

class ExifImageTrainingDataFormats(Enum):
    Flickr = ExifImageTrainingDataFormat(exifReader = FlickrExifReader(), 
                                         imagePathReader = FlickrImagePathReader(), 
                                         name = CONST_FLICKR_DATA_FORMAT_NAME)

    MirFlickr = ExifImageTrainingDataFormat(exifReader = MIRFLICKRExifReader(), 
                                            imagePathReader = MIRFLICKRImagePathReader(), 
                                            name = CONST_MIRFLICKR_DATA_FORMAT_NAME)

    @staticmethod
    def toTrainingDataFormat(name: str) -> "ExifImageTrainingDataFormats":
        """ Returns the training data format for the given string or None if no such training data format exists. """
        if name == CONST_FLICKR_DATA_FORMAT_NAME:
            return ExifImageTrainingDataFormats.Flickr
        elif name == CONST_MIRFLICKR_DATA_FORMAT_NAME:
            return ExifImageTrainingDataFormats.MirFlickr
        else:
            return None
    