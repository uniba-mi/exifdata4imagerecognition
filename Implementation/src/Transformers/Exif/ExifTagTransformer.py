from typing import Callable, Dict, List, Set, Tuple
from DomainModel.ExifData import ExifData, ExifTag
from Transformers.Transformer import Transformer, TransformError
from numpy.core.numeric import Infinity
import copy
import numpy as np
import math

TOTAL_SECONDS_PER_DAY = 86400

class ExifTagTransformerFunction(object):
    """ Represents a transformer function for an exif tag transformer object. The function provides a callable which
    will be called for transforming an exif tag. Additionally, the number of possible uniqe values the callable can produce
    will be stored.  """
    callable: Callable
    valueCount: int

    def __init__(self, callable: Callable, valueCount: int = math.inf):
        """ creates a new transformer function object. The value count indicates the number of unique values that 
        can be produced by the transformer function. """
        self.callable = callable
        self.valueCount = valueCount


class ExifTagTransformer(Transformer[ExifData, ExifData]):
    """ The exif tag transformer transforms strings representations of exif tags into their corresponding numeric value by using
    attached transformer functions. """

    @staticmethod
    def standard() -> "ExifTagTransformer":
        """ Returns an exif tag transformer with registered transformer functions for the tags  
        [FocalLength, ISO, FNumber, ExposureTime, Flash, BrightnessValue, ExposureCompensation, FocalLengthIn35mmFormat,
        Saturation, Sharpness, Contrast, Orientation, LightSource, MaxApertureValue]. """
        transformer = ExifTagTransformer()
        transformer.registerTransformerFunction(ExifTag.FocalLength.name, ExifTagTransformerFunction(transformFocalLength))
        transformer.registerTransformerFunction(ExifTag.ISO.name, ExifTagTransformerFunction(transformNumber))
        transformer.registerTransformerFunction(ExifTag.FNumber.name, ExifTagTransformerFunction(transformFNumber))
        transformer.registerTransformerFunction(ExifTag.ExposureTime.name, ExifTagTransformerFunction(transformExposureTime))
        transformer.registerTransformerFunction(ExifTag.Flash.name, ExifTagTransformerFunction(transformFlash, 2))
        transformer.registerTransformerFunction(ExifTag.BrightnessValue.name, ExifTagTransformerFunction(transformFractureOrDecimal))
        transformer.registerTransformerFunction(ExifTag.ExposureCompensation.name, ExifTagTransformerFunction(transformExposureCompensation))
        transformer.registerTransformerFunction(ExifTag.FocalLengthIn35mmFormat.name, ExifTagTransformerFunction(transformFocalLength))
        transformer.registerTransformerFunction(ExifTag.Saturation.name, ExifTagTransformerFunction(transformSaturation, 3))
        transformer.registerTransformerFunction(ExifTag.Sharpness.name, ExifTagTransformerFunction(transformSharpness, 3))
        transformer.registerTransformerFunction(ExifTag.Contrast.name, ExifTagTransformerFunction(transformContrast, 3))
        transformer.registerTransformerFunction(ExifTag.Orientation.name, ExifTagTransformerFunction(transformOrientation, 9))
        transformer.registerTransformerFunction(ExifTag.LightSource.name, ExifTagTransformerFunction(transformLightSource, 17))
        transformer.registerTransformerFunction(ExifTag.MaxApertureValue.name, ExifTagTransformerFunction(transformFractureOrDecimal))
        #transformer.registerTransformerFunction(ExifTag.DateTimeOriginalCos.name, ExifTagTransformerFunctiontransformDateTimeCos))
        #transformer.registerTransformerFunction(ExifTag.DateTimeOriginalSin.name, ExifTagTransformerFunctiontransformDateTimeSin))
        return transformer

    transformerFunctions: Dict[str, ExifTagTransformerFunction]
    whiteList: list
    insertNanOnTransformError: bool
    verbose: int

    def __init__(self, insertNanOnTransformError: bool = False, verbose: int = 1):
        self.transformerFunctions = {}
        self.whiteList = []
        self.insertNanOnTransformError = insertNanOnTransformError
        self.verbose = verbose
    
    @property
    def typedTagLists(self) -> Tuple[List[str], List[str], List[str]]:
        """ Returns three sets with the tansformers transformed tags put into depending on their type, either numerical, categorical or binary.  """
        numerical = []
        categorical = []
        binary = []
        for exifTag in self.transformedTags:
            transformerFunction = self.transformerFunctions[exifTag]
            if transformerFunction.valueCount == math.inf:
                numerical.append(exifTag)
            elif transformerFunction.valueCount > 2:
                categorical.append(exifTag)
            elif transformerFunction.valueCount == 2:
                binary.append(exifTag)
        
        return numerical, categorical, binary

    @property
    def transformedTags(self) -> Set[str]:
        """ Returns a list of all exif tags the transformer currently has a transformer function for. """
        return set(self.transformerFunctions.keys())

    def registerTransformerFunction(self, exifTag: str, function: ExifTagTransformerFunction):
        if not str is None and not function is None:
            self.transformerFunctions[exifTag] = function
    
    def addWhitelistedValue(self, value):
        """ Adds a whitelisted value to the transformer. Exif tags with whitelisted values won't be transformed with their 
        registered transformer function. """
        if not value in self.whiteList:
            self.whiteList.append(value)
    
    def transform(self, data: ExifData) -> ExifData:
        """ Transforms each exif tag of the given exif data object using the registered transformer functions. If the data
         contains a tag which there is no transformer function for, the transformer will throw an error. """
        if not data is None:
            newTags = {}
            for tag in data.tags:
                if tag in self.transformerFunctions:
                    exifTagValue = data.tags[tag]
                    if exifTagValue not in self.whiteList:
                        try:
                            newTags[tag] = self.transformerFunctions[tag].callable(exifTagValue)
                        except TransformError as exception:
                            if self.insertNanOnTransformError:
                                newTags[tag] = np.nan
                                if self.verbose > 0:
                                    print(exception)
                            else:
                                raise exception
                    else:
                        newTags[tag] = copy.copy(exifTagValue)
                else:
                    raise TransformError("the given exif data contains at least one exif tag for which there is no transformer function registered -> " + tag)

            # check if we have an instance with only nan values
            if all(value == np.nan for value in newTags.values()):
                raise TransformError("the transformed instance contains only nan values: " + data.id if not data.id == None else "")

            return ExifData(tags = newTags, id = data.id)
        else:   
            raise ValueError("exif data None object in exif tag transformer.")


# transformer functions for specific exif tags, created according to the exif specification located at: 
# https://exiftool.org/TagNames/EXIF.html

def transformNumber(value: str) -> float:
    """ Transforms a numeric string representation to a numeric float value, e.g. '1.3' -> 1.3 """
    try:
        val = float(value)
        if(val == Infinity):
            raise TransformError("invalid numeric value - infinity.")
        return val
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformDateTimeCos(value: str) -> float:
    """ Transforms a numeric date / time representation to a numeric cos-scaled(24H) float value, e.g. '2005:05:15 17:08:15' -> 1.3 """
    try:
        val = value.split(" ")[1].split(":")
        seconds = int(val[0]) * 60 * 60 + int(val[1]) * 60.0 + int(val[2])
        cosinusTime = np.cos(2 * np.pi * seconds / TOTAL_SECONDS_PER_DAY)
        return cosinusTime
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformDateTimeSin(value: str) -> float:
    """ Transforms a numeric date / time representation to a numeric sin-scaled(24H) float value, e.g. '2005:05:15 17:08:15' -> 1.3 """
    try:
        val = value.split(" ")[1].split(":")
        seconds = int(val[0]) * 60 * 60 + int(val[1]) * 60.0 + int(val[2])
        sinusTime = np.sin(2 * np.pi * seconds / TOTAL_SECONDS_PER_DAY)
        return sinusTime
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformFracture(value: str) -> float:
    """ Transforms the given fracture representation to its corresponding decimal value, e.g. '1/10' -> 0.1 """
    try:
        split = value.split("/")
        if len(split) == 2:
            return float(split[0]) / float(split[1])
        else:
            raise TransformError("the given fracture has an invalid format and thus cannot be converted to a decimal value -> " + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformFractureOrDecimal(value: str) -> float:
    """ Transforms the given input, which must be given as decimal number or as fracture, to a decimal value. """
    try:
        if "/" in value:
            # we have a fracture
            return transformFracture(value = value)
        else:
            # we have a number
            return transformNumber(value = value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformExposureCompensation(value: str) -> float:
    """ Transforms the given input, which must be given as decimal number or as fracture, can contain ev at the end """
    try:
        return transformFractureOrDecimal(value = value.lower().replace("ev", ""))
    except Exception as exception:
        raise TransformError(exception)

def transformFNumber(value: str) -> float:
    """ Transforms the exif tag "FNumber" and "Aperture". The input must be given as decimal number or fraction.
    It can be preceeded by 'f/' e.g. 'f/4.2' """
    try:
        return transformFractureOrDecimal(value = value.replace("f/", ""))
    except Exception as exception:
        raise TransformError(exception)

def transformExposureTime(value: str) -> float:
    """ Transforms the exif tag "ExposureTime". The input must be given as decimal number or fraction or in the
    following format: '0.05 sec (1/20)' """
    try:
        if "sec" in value:
            value = value.split(" ")[0]
        return transformFractureOrDecimal(value = value)
    except Exception as exception:
        raise TransformError(exception)
    
def transformFocalLength(value: str) -> float:
    """ Transforms the exif tag 'FocalLength'. The input must be given as decimal number with unit 'mm'. 
    The function will return the decimal value of the focal length without unit, e.g. '3.1mm' -> 3.1. """
    try:
        return float(value.replace("mm", ""))
    except Exception as exception:
        raise TransformError(exception)

def transformFlash(value: str) -> int:
    """ Transforms various representations of the exif tag 'Flash' to its corresponding numeric integer value, 
    e.g. 'Off, Did not fire' -> 0, 'Fired' -> 1. The input value must comply to the exif specification. """
    try:
        value = value.lower()
        if "no flash" in value or "did not fire" in value or "off" in value or "0" == value:
            return 0
        else:
            return 1
    except Exception as exception:
        raise TransformError(exception)

def transformSaturation(value: str) -> int:
    """ Transforms various representations of the exif tag 'Saturation' to its corresponding ordinal value, 
    e.g. 'low' -> 0, 'normal' -> 1, 'high' -> 2. The input value must comply to the exif specification. """
    try:
        value = value.lower()
        if "low" in value or value == "1":
            return 0
        elif "normal" in value or value == "0":
            return 1
        elif "high" in value or "enhanced" in value or value == "2":
            return 2
        elif "auto" in value:
            return np.nan
        else:
            raise TransformError("the given saturation value has an invalid format -> " + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformContrast(value: str) -> int:
    """ Transforms the exif tag 'Contrast' to its corresponding ordinal value, e.g.
    'low' -> 0, 'normal' -> 1, 'high' -> 2. The input value must comply to the exif specification.
    """
    try:
        value = value.lower()
        if "low" in value or "soft" in value or value == "1":
            return 0
        elif "normal" in value or value == "0":
            return 1
        elif "high" in value or "hard" in value or value == "2":
            return 2
        else:
            raise TransformError("the given contrast value has an invalid format ->" + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformSharpness(value: str) -> int:
    """ Transforms the exif tag 'Sharpness' to its corresponding ordinal value, e.g.
    'soft' -> 0, 'normal' -> 1, 'hard' -> 2. The input value must comply to the exif specification.
    """
    try:
        value = value.lower()
        if "soft" in value or value == "1":
            return 0
        elif "normal" in value or value == "0":
            return 1
        elif "hard" in value or value == "2":
            return 2
        else:
            raise TransformError("the given sharpness value has an invalid format ->" + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformOrientation(value: str) -> int:
    """ Transforms the exif tag 'Orientation' to its corresponding categorical integer value, e.g. 'rotate 180' -> 3.
    The input value most comply to the exif specification.
    1 = Horizontal (normal)
    2 = Mirror horizontal
    3 = Rotate 180
    4 = Mirror vertical
    5 = Mirror horizontal and rotate 270 CW
    6 = Rotate 90 CW
    7 = Mirror horizontal and rotate 90 CW
    8 = Rotate 270 CW
    """
    try:
        value = value.lower()
        if "unknown" in value or value == "0":
            return 0
        elif "horizontal (normal)" in value or value == "1":
            return 1
        elif "mirror horizontal and rotate 270" in value or value == "5":
            return 5
        elif "rotate 180" in value or "rotated 180 degrees" in value or value == "3":
            return 3
        elif "mirror vertical" in value or value == "4":
            return 4
        elif "mirror horizontal and rotate 90" in value or value == "7":
            return 7
        elif "rotate 90" in value or "rotated 90 degrees clockwise" in value or value == "6":
            return 6
        elif "mirror horizontal" in value or value == "2":
            return 2
        elif "rotate 270" in value or "rotated 90 degrees counter clockwise" in value or value == "8":
            return 8
        else:
            raise TransformError("the given orientation value has an invalid format -> " + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)

def transformLightSource(value: str) -> int:
    """ Transforms the exif tag 'LightSource' to its corresponding categorical integer value, e.g. 'Unknown' -> 0.
    The input value most comply to the exif specification.
    0 = Unknown
    1 = Fluorescent (different sources)
    2 = Daylight
    3 = Tungsten (Incandescent) 	
    4 = Flash
    5 = Fine Weather
    6 = Cloudy
    7 = Shade
    8 = Standard Light A
    9 = Standard Light B
    10 = Standard Light C
    11 = D55
    12 = D65
    13 = D75
    14 = D50
    15 = SO Studio Tungsten
    16 = Other
    """
    try:
        value = value.lower()
        if "unknown" in value or value == "0":
            return 0
        elif "fluorescent" in value or value == "2" or value == "12" or value == "13" or value == "14" or value == "15" or value == "16":
            return 1
        elif "daylight" in value or value == "1":
            return 2
        elif "tungsten" in value or value == "3":
            return 3
        elif "flash" in value or value == "4":
            return 4
        elif "weather" in value or value == "9":
            return 5
        elif "cloudy" in value or value == "10":
            return 6
        elif "shade" in value or value == "11":
            return 7
        elif "standard light a" in value or value == "17":
            return 8
        elif "standard light b" in value or value == "18":
            return 9
        elif "standard light c" in value or value == "19":
            return 10
        elif "d55" in value or value == "20":
            return 11
        elif "d65" in value or value == "21":
            return 12
        elif "d75" in value or value == "22":
            return 13
        elif "d50" in value or value == "23":
            return 14
        elif "iso" in value or value == "24":
            return 15
        elif "other" in value or value == "255":
            return 16
        else:
            raise TransformError("the given light source value has an invalid format -> " + value)
    except TransformError:
        raise
    except Exception as exception:
        raise TransformError(exception)
