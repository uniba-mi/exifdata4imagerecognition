from DomainModel.ExifData import ExifData
from DomainModel.Image import Image

class EXIFImageTrainingData(object):
    """ An "EXIFImageTrainingData" object is a tuple that consists exif data and an image for one training example that belongs
    to a certain target concept. """

    exif: ExifData
    image: Image

    def __init__(self, exifData: ExifData, image: Image):
        if not exifData == None and not image == None:
            self.exif = exifData
            self.image = image
        else:
            raise ValueError("exif data and image data of a training instance may not be None.")
    
    def __eq__(self, other):
        if isinstance(other, EXIFImageTrainingData):
            if not self.exif.id is None and not other.exif.id is None:
                return self.exif.id is other.exif.id and self.image.id == other.image.id
            return self.exif.tags == other.exif.tags and self.image.id == other.image.id
        return False
    
    def __hash__(self):
        if not self.exif.id is None:
            return hash(self.exif.id + self.image.id)
        return hash((frozenset(self.exif.tags.keys()), self.image.id))

class TestTrainingData(object):
    """ An empty training data object for testing purposes. """

    id: str

    def __init__(self, id: str):
        self.id = id
    
    def __eq__(self, other):
        if isinstance(other, TestTrainingData):
            return self.id is other.id
        return False
    
    def __hash__(self):
        return hash(self.id)
        