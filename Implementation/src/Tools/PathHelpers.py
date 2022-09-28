import os
import glob
from typing import List

def matchingFilePaths(folderPath: str, name: str = "", extension: str = "") -> List[str]:
    """ returns the file paths of all files within the given directory, with the given file extension,
    starting with the given name. """
    return getFilePaths(wildCard = os.path.join(folderPath, name + "*.*" + extension))

def getFilePaths(wildCard: str) -> List[str]:
    return glob.glob(wildCard)