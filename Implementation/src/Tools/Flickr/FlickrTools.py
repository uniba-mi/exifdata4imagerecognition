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

def toFlickrBase58Url(photoId: str) -> str:
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

def readMIProjectImageMetadata(directoryPath: str, serverDummy: str) -> list:
    """ Reads the image file names of the given directory into a metadata dictionary, such that it can be fed into a Flickr crawler object. 
    Since in the MI project the server was not encoded into the filename, it must be filled with a dummy value. """
    imageMetadata = []
    imagePaths = PathHelpers.matchingFilePaths(folderPath = directoryPath, extension = "jpg")
    for path in imagePaths:
        parts = Path(path).name.split("_")
        if len(parts) > 1:
            imageMetadata.append({ FlickrCodingKeys.IMAGE_ID.value: parts[0], 
                                   FlickrCodingKeys.SECRET.value: parts[1],
                                   FlickrCodingKeys.SERVER.value:  serverDummy })
    return imageMetadata

def reArangeMIRFlickr25Dataset(annotationDirectoryPath: str, 
                               imageDirectoryPath: str, 
                               exifDirectoryPath: str, 
                               destinationDirectoryPath: str) -> dict:
    """ re-aranges the images contained in the MIRFlickr25Dataset (https://press.liacs.nl/mirflickr/mirdownload.html). The images will be
    sorted according to their label names, such that there is a directory for each unique combination of class labels which containing the 
    corresponding images and exif files. """
    if (not os.path.isdir(annotationDirectoryPath) or 
        not os.path.isdir(exifDirectoryPath) or
        not os.path.isdir(imageDirectoryPath)):
        raise FileExistsError("at least one of the given input directories does not exist. ")
    
    textFiles = PathHelpers.matchingFilePaths(folderPath = annotationDirectoryPath, extension = "txt")

    # read in the concepts and the image numbers for each concept
    imagesAndConcepts = {}

    for file in textFiles:
        filePath = Path(file)

        # we only use the r1 concepts, the releveant ones
        if "_r1" in filePath.name.lower():
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

    
if __name__ == '__main__':
    """ reArangeMIRFlickr25Dataset(
        annotationDirectoryPath = "/Users/ralflederer/Downloads/mirflickr25k_annotations_v080", 
        imageDirectoryPath = "/Users/ralflederer/Downloads/mirflickr",
        exifDirectoryPath = "/Users/ralflederer/Downloads/mirflickr/meta/exif",
        destinationDirectoryPath = "/Users/ralflederer/Desktop/mirflickr_rearanged") """

    """ indorClasses = ["bathroom", "bedroom", "corridor", "kitchen", "office"]
    outdoorClasses = ["beach", "forest", "mountain", "river", "urban"] 

    # write metadata files for old mi project to be able to re-crawl the data
    metadataPath = Path("resources/metadata/")
    metadataPath.mkdir(parents = True, exist_ok = True)
    for indoorClass in indorClasses:
        path = "resources/dataset/indoor/{className}/images".format(className = indoorClass)
        metadata = readMIProjectImageMetadata(path, "1234")
        with open(metadataPath.joinpath(indoorClass + ".json"), "w") as file:
            json.dump(metadata, file)

    for outdoorClass in outdoorClasses:
        path = "resources/dataset/outdoor/{className}/images".format(className = outdoorClass)
        metadata = readMIProjectImageMetadata(path, "1234")
        with open(metadataPath.joinpath(outdoorClass + ".json"), "w") as file:
            json.dump(metadata, file) """
       