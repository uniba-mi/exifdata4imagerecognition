from enum import Enum
from typing import List, Set, Tuple

class ExifTagType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2
    BINARY = 3
    UNKNOWN = 4


class ExifTag(Enum):
    Flash = 1,
    ISO = 2,
    FNumber = 3,
    Saturation = 4,
    Sharpness = 5,
    Contrast = 6,
    Orientation = 7,
    FocalLength = 8,
    ExposureTime = 9,
    BrightnessValue = 10,
    ExposureCompensation = 11,
    FocalLengthIn35mmFormat = 12,
    LightSource = 13,
    MaxApertureValue = 14,
    """ DateTimeOriginalCos = 15,
    DateTimeOriginalSin = 16 """

def toInternalExifTagName(tagName: str) -> Tuple[str]:
        """ Returns the internal tag names for the given exif tag name. The given name shall be trimmed and all whitespaces shall 
        be removed. If there is no internal tag name for the given string, None will be returned. """
        if tagName in ["ISOSpeed", "ISO"]:
            return ExifTag.ISO.name
        elif tagName in ["Aperture", "FNumber"]:
            return ExifTag.FNumber.name
        elif tagName in ["Flash"]:
            return ExifTag.Flash.name
        elif tagName in ["Saturation"]:
            return ExifTag.Saturation.name
        elif tagName in ["Sharpness"]:
            return ExifTag.Sharpness.name
        elif tagName in ["Contrast"]:
            return ExifTag.Contrast.name
        elif tagName in ["Orientation"]:
            return ExifTag.Orientation.name
        elif tagName in ["FocalLength"]:
            return ExifTag.FocalLength.name
        elif tagName in ["ExposureTime", "Exposure"]:
            return ExifTag.ExposureTime.name
        elif tagName in ["BrightnessValue", "Brightness"]:
            return ExifTag.BrightnessValue.name
        elif tagName in ["ExposureCompensation", "ExposureBias"]:
            return ExifTag.ExposureCompensation.name
        elif tagName in ["FocalLengthIn35mmFormat", "FocalLengthIn35mmFilm"]:
            return ExifTag.FocalLengthIn35mmFormat.name
        elif tagName in ["LightSource"]:
            return ExifTag.LightSource.name
        elif tagName in ["MaxApertureValue", "MaximumLensAperture"]:
            return ExifTag.MaxApertureValue.name
        """ elif tagName in ["DateTimeOriginal", "DateandTime(Original)"]:
            return (ExifTag.DateTimeOriginalCos.name, ExifTag.DateTimeOriginalSin.name)
        elif tagName in ["DateTimeOriginalCos"]:
            return ExifTag.DateTimeOriginalCos.name
        elif tagName in ["DateTimeOriginalSin"]:
            return ExifTag.DateTimeOriginalSin.name """
        return None

class ExifData(object):
    """ An "ExifData" object represents the exif data for an image. The tags are stored in a dictionary. 
    The keys of the dictionary represent the names of the corresponding exif-tags.
    """

    @staticmethod
    def tagType(exifTag: str) -> "ExifTagType":
        """ Returns the type of the given exif tag which can be either binary, categorical or numerical. """
        if exifTag in [ExifTag.Flash.name]:
            return ExifTagType.BINARY
        elif exifTag in [ExifTag.Saturation.name, ExifTag.Sharpness.name, ExifTag.Contrast.name, ExifTag.Orientation.name, 
                         ExifTag.LightSource.name]:
            return ExifTagType.CATEGORICAL
        elif exifTag in [ExifTag.FocalLength.name, ExifTag.ISO.name, ExifTag.FNumber.name, ExifTag.ExposureTime.name, 
                         ExifTag.BrightnessValue.name, ExifTag.ExposureCompensation.name, ExifTag.FocalLengthIn35mmFormat.name,
                         ExifTag.MaxApertureValue.name]: #ExifTag.DateTimeOriginalCos.name, ExifTag.DateTimeOriginalSin.name
            return ExifTagType.NUMERICAL
        else:
            return ExifTagType.UNKNOWN
    
    @staticmethod
    def typedTagLists(exifTags: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """ Returns three sets with the given tags put into depending on their type, either numerical, categorical or binary.  """
        numerical = []
        categorical = []
        binary = []
        for exifTag in exifTags:
            type = ExifData.tagType(exifTag = exifTag)
            if type == ExifTagType.BINARY:
                binary.append(exifTag)
            elif type == ExifTagType.CATEGORICAL:
                categorical.append(exifTag)
            elif type == ExifTagType.NUMERICAL:
                numerical.append(exifTag)
            else:
                raise ValueError("unknown exif tag {tag} in tag list".format(tag = exifTag))
        
        return numerical, categorical, binary

    tags: dict
    id: str

    def __init__(self, tags: dict, id: str = None):
        if not tags == None and bool(tags):
            self.tags = tags
            self.id = id
        else:
            raise ValueError("the tags of an exif data object may not be None or empty.")

    @property
    def tagNames(self) -> Set[str]:
        """ returns a sorted list of all exif tag names of this exif data object. """
        return set(sorted(self.tags.keys()))

    def containsTag(self, tag: str) -> bool:
        """ returns true if the exif data object contains an exif-tag with the given tag name, false otherwise. """
        return tag in self.tags
    
    def containsTags(self, tags: set) -> bool:
        """ returns true if the exif data object contains all exif-tags in the given list, false otherwise. """
        return all(tag in self.tags for tag in tags)
