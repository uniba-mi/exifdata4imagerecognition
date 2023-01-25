from enum import Enum
from pathlib import Path
import json
import os
import shutil
from Tools import PathHelpers

class FlickrCodingKeys(Enum):
    IMAGE_ID = "id"
    SERVER = "server"
    SECRET = "secret"
       
def readExifTags(filePath: str) -> dict:
    """ Returns a dictionary of all exif tags stored in the json file at the given file path. 
    The json file must comply to the Flickr specification: https://www.flickr.com/services/api/flickr.photos.getExif.htm """

    # check if the given file exists
    path = Path(filePath).resolve()
    if not path.exists():
        return None

    with open(filePath, encoding="utf8") as fileContent:
        fileContent = json.load(fileContent)

        # first we check, if we have a valid Flickr JSON-Exif data file
        # this is the case if the json file contains a "photo" entry, an "exif" entry and an "id" entry
        if ("photo" in fileContent and "exif" in fileContent["photo"] and "id" in fileContent["photo"]):
            exifTags = {}
            # now we loop through all exif-tags and store them in our dictionary, if
            # the tag contains a value
            for tag in fileContent["photo"]["exif"]:
                if "tag" in tag and "raw" in tag and "_content" in tag["raw"]:
                    exifTags[tag["tag"]] = tag["raw"]["_content"]
            return exifTags
        else:
            # we have an invalid exif tag file
            return None

def checkExifTagsExsist(filePath: str, tags: list) -> bool:
    """ Returns true if the file at the given path contains all given exif tags, false otherwise. 
    The json file must comply to the Flickr specification: https://www.flickr.com/services/api/flickr.photos.getExif.htm"""
    
    exifTags = readExifTags(filePath = filePath)
    if not exifTags is None and all(tag in exifTags for tag in tags):
        return True
    return False

def toFlickrBase58Url(photoId: int) -> str:
    """ Converts the given Flickr photo id to a base58 Flickr short website url. """
    alphabet = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
    bc = len(alphabet)
    enc = ""
    while photoId >= bc:
        div, mod=divmod(photoId, bc)
        enc = alphabet[mod] + enc
        photoId = int(div)
    enc = alphabet[photoId] + enc
    return "flic.kr/p/{base58}".format(base58 = enc) 

def readImageMetadata(directoryPath: str, serverDummy: str) -> list:
    """ Reads the image file names of the given directory into a metadata dictionary, such that it can be fed into a Flickr crawler object. """
    datasetPath = Path(directoryPath)
    if datasetPath.exists() and datasetPath.is_dir():
        imagePaths = [file for file in datasetPath.rglob("*.jpg") if file.is_file()]
        print("found " + str(len(imagePaths)) + " images")
        imageMetadata = {}
        for path in imagePaths:
            p = Path(path)
            parts = p.name.split("_")
            conceptName = p.parent.parent.stem
            if len(parts) > 1:
                if not conceptName in imageMetadata.keys():
                    imageMetadata[conceptName] = []
                imageMetadata[conceptName].append({ FlickrCodingKeys.IMAGE_ID.value: parts[0], 
                                                    FlickrCodingKeys.SECRET.value: parts[1],
                                                    FlickrCodingKeys.SERVER.value: serverDummy if len(parts) < 4 else parts[2] })
    else:
        raise ValueError("not dataset at given location")
    return imageMetadata

def reArangeMIRFlickr25Dataset(annotationDirectoryPath: str, 
                               imageDirectoryPath: str, 
                               exifDirectoryPath: str, 
                               destinationDirectoryPath: str,
                               relevantOnly: bool = False,
                               minImagesPerConcept: int = 0) -> dict:
    """ re-aranges the images contained in the MIRFlickr25Dataset (https://press.liacs.nl/mirflickr/mirdownload.html). The images will be
    sorted according to their label names, such that there is a directory for each unique combination of class labels which containing the 
    corresponding images and exif files. """
    if (not os.path.isdir(annotationDirectoryPath) or 
        not os.path.isdir(exifDirectoryPath) or
        not os.path.isdir(imageDirectoryPath)):
        raise FileExistsError("at least one of the given input directories does not exist. ")
    
    textFiles = PathHelpers.matchingFilePaths(folderPath = annotationDirectoryPath, extension = "txt")

    # select file paths
    paths = []
    for file in textFiles:
        filePath = Path(file)
        if relevantOnly:
            # we only use the r1 concepts, the releveant ones
            if "_r1" in filePath.name.lower():
                paths.append(filePath)
        else:
            if not "_r1" in filePath.name.lower():
                paths.append(filePath)

    # read in the concepts and the image numbers for each concept
    imagesAndConcepts = {}
    
    for filePath in paths:
        print("reading in " + filePath.name.lower())
        with open(filePath) as conceptImages:
            for line in conceptImages.readlines():
                imageNumber = line.strip()

                # create entry for image number, if there is not already one
                if not imageNumber in imagesAndConcepts.keys():
                    imagesAndConcepts[imageNumber] = list()
                    
                # add concept name to image number
                imagesAndConcepts[imageNumber].append(filePath.stem.replace("_r1", ""))
            
    # create / clear destination folder
    superFolder = Path(destinationDirectoryPath).joinpath("concepts")
    if superFolder.exists():
        shutil.rmtree(superFolder)
    superFolder.mkdir(parents = True)

    # move images to new folder structure
    imageDirectoryPath = Path(imageDirectoryPath)
    exifDirectoryPath = Path(exifDirectoryPath)
    
    # create multi-label concept folders for the images and copy exif / image data
    print("copying images and exif data to destination folders")
    for imageNumber in imagesAndConcepts.keys():
        concepts = imagesAndConcepts[imageNumber]
        conceptFolderName = ",".join(concepts)

        # create directories for image / exif data
        conceptImageDirectory = superFolder.joinpath(conceptFolderName).joinpath("images")
        conceptExifDirectory = superFolder.joinpath(conceptFolderName).joinpath("exif")

        if not conceptImageDirectory.exists():
            conceptImageDirectory.mkdir(parents = True)
        if not conceptExifDirectory.exists():  
            conceptExifDirectory.mkdir(parents = True)

        # copy exif / image data
        imageName = "im" + imageNumber + ".jpg"
        exifName = "exif" + imageNumber + ".txt"
        imagePath = Path(imageDirectoryPath.joinpath(imageName))
        exifPath = Path(exifDirectoryPath.joinpath(exifName))
        if not conceptImageDirectory.joinpath(imageName).exists():
            shutil.copy(imagePath, conceptImageDirectory)
        if not conceptExifDirectory.joinpath(exifName).exists():
            shutil.copy(exifPath, conceptExifDirectory)
    
    # filter out concepts with less than n images
    conceptDirectories = [dir for dir in superFolder.iterdir() if dir.is_dir()]
    for directory in conceptDirectories:
        subDirectory = directory.joinpath("images")
        imagePaths = [file for file in subDirectory.iterdir() if file.is_file()]
        print("concept " + directory.name.lower() + " contains " + str(len(imagePaths)) + " images")
        if len(imagePaths) < minImagesPerConcept:
            print("... removing concept because of to less images")
            shutil.rmtree(directory)
    
def shrinkDataset(imagesPerClass: int, directoryPath: str, destinationDirectoryPath: str):
    """ Reduces the dataset at the given file path to contain only the given amount of images per training concept. """
    datasetPath = Path(directoryPath)
    destinationPath = Path(destinationDirectoryPath)
    if datasetPath.exists and datasetPath.is_dir():
        subDirs = [directory for directory in datasetPath.glob("*") if directory.is_dir()]
        for superConceptDir in subDirs:
            subConceptDirs = [directory for directory in superConceptDir.glob("*") if directory.is_dir()]
            for dataDir in subConceptDirs:
                print("copying images and exif data for concept: " + dataDir.stem)
                destinationDirectory = destinationPath.joinpath(superConceptDir.stem).joinpath(dataDir.stem)
                imageDirPath = dataDir.joinpath("images")
                exifDirPath = dataDir.joinpath("exif")
                    
                # read in images
                imagePaths = [image for image in imageDirPath.glob("*") if image.is_file()]

                for index in range(0, min(imagesPerClass, len(imagePaths))):
                    imagePath = imagePaths[index]

                    # extract image-id and find corresponding exif file
                    imageId = imagePath.stem.split("_")[0]

                    exifPaths = [exif for exif in exifDirPath.glob(imageId + "*.json") if exif.is_file()]
                    if len(exifPaths) == 1:
                        exifPath = exifPaths[0]
                        exifDestinationPath = destinationDirectory.joinpath("exif").joinpath(exifPath.name)
                        imageDestinationPath = destinationDirectory.joinpath("images").joinpath(imagePath.name)

                        # create directories if needed and copy files
                        os.makedirs(os.path.dirname(exifDestinationPath), exist_ok = True)
                        os.makedirs(os.path.dirname(imageDestinationPath), exist_ok = True)
                        shutil.copy(exifPath, exifDestinationPath)
                        shutil.copy(imagePath, imageDestinationPath)

                    else:
                        raise ValueError("no exif file found for image with id: " + imageId)
    else:
        raise ValueError("no dataset directory found at: " + directoryPath)

if __name__ == '__main__':
    # example to create Flickr short URLs
    exampleImageIds = [26154431303, 113541139, 48948687776, 6780392960, 5791593186]

    for id in exampleImageIds:
        print("\\url{https://" + toFlickrBase58Url(photoId = id) + "}")