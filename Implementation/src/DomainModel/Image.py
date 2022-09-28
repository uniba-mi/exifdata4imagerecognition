from pathlib import Path

class Image(object):
    """ An "Image" object represents an image that is identified by a file path, stored on the hard disk. """

    path: Path

    def __init__(self, path: str, id: str = None):
        filePath = Path(path)
        if not filePath.exists():
            raise ValueError("There is no file at the given image path.")
        else:
            self.path = filePath
            self.__id = id

    @property
    def id(self) -> str:
        """ returns the identifier of the image or the file name of the image if no identifier was assigned. """
        if self.__id == None:
            return self.path.name
        else: 
            return self.__id
