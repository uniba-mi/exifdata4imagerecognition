from abc import ABC, abstractmethod
from Tools import PathHelpers
from pathlib import Path

class ImagePathReader(ABC):
    """ An image path reader reads all image files in a directory and returns the corresponding file paths. """

    extension: str
    
    def __init__(self, extension: str):
        self.extension = extension

    @abstractmethod
    def readImageFiles(self, directoryPath: str) -> dict:
        """ Reads the image files in the given directory and returns a dictionary where the keys of the 
        dictionary contain the ids (filenames) of the image files and the values of the dictionary contain the file paths. """
        pass

class FlickrImagePathReader(ImagePathReader):
    """ Reads the ids and image paths of images where the flickr image id is on start of the file name followed by '_'. """

    def __init__(self, extension: str = "jpg"):
        super().__init__(extension = extension)

    def readImageFiles(self, directoryPath: str) -> dict:
        return { Path(path).name.split("_")[0] : path for path in PathHelpers.matchingFilePaths(folderPath = directoryPath, 
                                                                                                extension = self.extension) }

class MIRFLICKRImagePathReader(ImagePathReader):
    """ Reads the ids and image paths of images where the name of images starts with 'im' followed by the image id.  """

    def __init__(self, extension: str = "jpg"):
        super().__init__(extension = extension)

    def readImageFiles(self, directoryPath: str) -> dict:
        return { Path(path).name.replace("im", "").replace(self.extension, "").replace(".", "") : path for path in PathHelpers.matchingFilePaths(folderPath = directoryPath, 
                                                                                                                                                 extension = self.extension) }                                                                                       
                                                                                                                                                 