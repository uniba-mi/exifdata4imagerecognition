from Tools.Flickr.FlickrCrawler import FlickrCrawler, FlickrUrlBuilder
from Tools.GUI.Spinner import Spinner
from Tools.GUI.ArgsHelper import argForParam, paramSet
import sys
from pathlib import Path
import time
import json

# global variables
crawler = FlickrCrawler()

# crawling parameters
PARAM_GROUP = "-group"
PARAM_STARTPAGE = "-sp"
PARAM_PAGE_LIMIT = "-pl"
PARAM_PHOTOS_PER_PAGE = "-ppp"
PARAM_FILTER_TAG = "-ft"
PARAM_METADATA_SAVE_PATH = "-sm"
PARAM_METADATA_ONLY = "-om"
PARAM_API_KEY = "-key"

PARAM_PHOTOS = "-photos"
PARAM_PHOTOS_DIR = "-odir"
PARAM_CRAWL_EXIF = "-exif"
PARAM_REQUIRED_EXIF = "-re"
PARAM_PHOTO_SIZE = "-ps"

PARAM_SEARCH = "-search"

PARAM_HELP = "-help"

def printUsage():
    print("Flickr Crawler Usage: ")
    print(" ")
    print("{groupId} 'groupId' of the Flickr group to crawl".format(groupId = PARAM_GROUP))
    print("\trequired:")
    print("\t \t{key} Flickr api key".format(key = PARAM_API_KEY))
    print("\t \t{dir} directory for storing crawled photos and exif data".format(dir = PARAM_PHOTOS_DIR))
    print("\toptional:")
    print("\t \t{sp}: start page of the group".format(sp = PARAM_STARTPAGE))
    print("\t \t{pl}: page limit".format(pl = PARAM_PAGE_LIMIT))
    print("\t \t{ppp}: photos per group page (default 500)".format(ppp = PARAM_PHOTOS_PER_PAGE))
    print("\t \t{ft}: additional filter tag for photos".format(ft = PARAM_FILTER_TAG))
    print("\t \t{om}: if set, only metadata will be crawled, no photos or exif data".format(om = PARAM_METADATA_ONLY))
    print("\t \t{sm}: path for storing crawled metadata file (needed when {om} is set)".format(sm = PARAM_METADATA_SAVE_PATH, om = PARAM_METADATA_ONLY))
    print("\t \t -- all parameters of the 'photos' function, see below, only if {om} is not set".format(om = PARAM_METADATA_ONLY))

    print(" ")
    print("{searchTerm} 'search term' to use for searching photos".format(searchTerm = PARAM_SEARCH))
    print("\trequired:")
    print("\t \t{key} Flickr api key".format(key = PARAM_API_KEY))
    print("\t \t{dir} directory for storing crawled photos and exif data".format(dir = PARAM_PHOTOS_DIR))
    print("\toptional:")
    print("\t \t{sp}: start page of the photo search".format(sp = PARAM_STARTPAGE))
    print("\t \t{pl}: page limit".format(pl = PARAM_PAGE_LIMIT))
    print("\t \t{ppp}: photos per photo page (default 500)".format(ppp = PARAM_PHOTOS_PER_PAGE))
    print("\t \t{om}: if set, only metadata will be crawled, no photos or exif data".format(om = PARAM_METADATA_ONLY))
    print("\t \t{sm}: path for storing crawled metadata file (needed when {om} is set)".format(sm = PARAM_METADATA_SAVE_PATH, om = PARAM_METADATA_ONLY))
    print("\t \t -- all parameters of the 'photos' function, see below, only if {om} is not set".format(om = PARAM_METADATA_ONLY))

    print(" ")
    print("{photos} 'path' to metadata file containing photoIds, secrets and servers".format(photos = PARAM_PHOTOS))
    print("\trequired:")
    print("\t \t{dir} directory for storing crawled photos and exif data".format(dir = PARAM_PHOTOS_DIR))
    print("\t \t{key} Flickr api key (only if exif data is to be crawled)".format(key = PARAM_API_KEY))
    print("\toptional:")
    print("\t \t{exif} if set, exif data will be crawled".format(exif = PARAM_CRAWL_EXIF))
    print("\t \t{requiredExif} list of (required) exif tags photos must provide, seperated by ','".format(requiredExif = PARAM_REQUIRED_EXIF))
    print("\t \t{ps} size of the crawled photos (default: ' ')".format(ps = PARAM_PHOTO_SIZE))
    print("\t \t  possible values: {sizes}".format(sizes = FlickrUrlBuilder.validSizeKeys))
    print("\t \t  see: https://www.flickr.com/services/api/misc.urls.html")

# crawling functions

def crawlGroup(groupId: str, startPage: int, pageLimit: int, photosPerPage: int, filterTag: str, metadataPath: str) -> list:
    msg = "starting group crawling process for group {groupId}, startPage: {startPage}, pageLimit: {pageLimit}, photosPerPage: {photosPerPage}".format(
        groupId = groupId, startPage = startPage, pageLimit = pageLimit, photosPerPage = photosPerPage)
    if not filterTag is None:
        msg += ", filterTag: {filterTag}".format(filterTag = filterTag)
    msg += " ..."

    print(msg)

    spinner = Spinner()
    spinner.start()

    start = time.time()
    result = crawler.crawlGroup(groupId = groupId, startPage= startPage, pageLimit = pageLimit, filterTag = filterTag, photosPerPage = photosPerPage)
    end = time.time()

    spinner.stop()

    print("group crawling process finished, downloaded metadata for {count} photos in {time:.1f} seconds".format(
        count = len(result.imageMetadata), time = end - start))

    if not metadataPath is None:
        path = result.saveMetadata(directory = Path(metadataPath))
        print("saved photo metadata to: " + str(path))

    return result.imageMetadata

def searchPhotos(searchTerm: str, startPage: int, pageLimit: int, photosPerPage: int, metadataPath: str) -> list:
    msg = "starting photo search process for term '{searchTerm}', startPage: {startPage}, pageLimit: {pageLimit}, photosPerPage: {photosPerPage}".format(
        searchTerm = searchTerm, startPage = startPage, pageLimit = pageLimit, photosPerPage = photosPerPage)
    msg += " ..."

    print(msg)

    spinner = Spinner()
    spinner.start()

    start = time.time()
    result = crawler.searchPhotos(searchTerm = searchTerm, startPage= startPage, pageLimit = pageLimit, photosPerPage = photosPerPage)
    end = time.time()

    spinner.stop()

    print("searching process finished, downloaded metadata for {count} photos in {time:.1f} seconds".format(
        count = len(result.imageMetadata), time = end - start))

    if not metadataPath is None:
        path = result.saveMetadata(directory = Path(metadataPath))
        print("saved photo metadata to: " + str(path))

    return result.imageMetadata

def crawlPhotos(metadata: list, photosDir: str, crawlExif: bool, requiredExifTags: list = [], photoSizeKey: str = " "):
    if not crawlExif:
        print("starting photo crawling process for {metadataLength} photos ...".format(metadataLength = len(metadata)))
    else:
        print("starting photo crawling process for {metadataLength} photos with exif data ...".format(metadataLength = len(metadata)))
        if not requiredExifTags == []:
            print("only crawling photos that provide (at least) the following exif tags: {re}".format(re = requiredExifTags))

    spinner = Spinner()
    spinner.start()

    start = time.time()
    result = crawler.crawlPhotos(metaData = metadata, 
                        directory = Path(photosDir), 
                        sizeKey = photoSizeKey, 
                        crawlExif = crawlExif, 
                        requiredExifTags = requiredExifTags)
    end = time.time()
    
    spinner.stop()

    print("crawling process finished, downloaded a total of {count} photos in {time:.1f} seconds".format(
        count = result.crawledPhotoCount, time = end - start))
    print("results are located at: " + photosDir)

# main entry point
if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0 or (len(args) == 1 and paramSet(args, PARAM_HELP)):
        printUsage()
    else:
        try:
            if paramSet(args, PARAM_GROUP):
                # args for the group crawling process
                groupId = argForParam(args, PARAM_GROUP)
                startPage = int(argForParam(args, PARAM_STARTPAGE, 1))
                pageLimit = int(argForParam(args, PARAM_PAGE_LIMIT, 0))
                photosPerPage = int(argForParam(args, PARAM_PHOTOS_PER_PAGE, 500))
                filterTag = argForParam(args, PARAM_FILTER_TAG)
                metadataPath = argForParam(args, PARAM_METADATA_SAVE_PATH)
                onlyMetadata = paramSet(args, PARAM_METADATA_ONLY)

                # args for the photo crawling process
                photosDir = argForParam(args, PARAM_PHOTOS_DIR)
                crawlExif = paramSet(args, PARAM_CRAWL_EXIF)
                requiredExifTags = argForParam(args, PARAM_REQUIRED_EXIF)
                requiredExifTags = requiredExifTags.split(",") if not requiredExifTags == None else []
                photoSize = argForParam(args, PARAM_PHOTO_SIZE, " ")
                apiKey = argForParam(args, PARAM_API_KEY)

                if apiKey is None:
                    raise ValueError("error: no api key provided")
                elif onlyMetadata and metadataPath is None:
                    raise ValueError("error: set to only crawl metadata, but no metadata path specified")
                elif not onlyMetadata and photosDir is None:
                    raise ValueError("error: no photos storage dir provided")
                
                # api-key check
                if not FlickrCrawler.testApiKey(apiKey = apiKey):
                    raise ValueError("the provided Flickr api key is invalid.")
                else:
                    crawler.apiKey = apiKey
                
                # first crawl group and get photo metadata, then crawl photos if wanted
                metadata = crawlGroup(groupId, startPage, pageLimit, photosPerPage, filterTag, metadataPath)
                
                if not onlyMetadata:
                    # crawl photos
                    crawlPhotos(metadata, photosDir, crawlExif, requiredExifTags, photoSize)
            
            elif paramSet(args, PARAM_SEARCH):
                # args for the group crawling process
                searchTerm = argForParam(args, PARAM_SEARCH)
                startPage = int(argForParam(args, PARAM_STARTPAGE, 1))
                pageLimit = int(argForParam(args, PARAM_PAGE_LIMIT, 0))
                photosPerPage = int(argForParam(args, PARAM_PHOTOS_PER_PAGE, 500))
                metadataPath = argForParam(args, PARAM_METADATA_SAVE_PATH)
                onlyMetadata = paramSet(args, PARAM_METADATA_ONLY)

                # args for the photo crawling process
                photosDir = argForParam(args, PARAM_PHOTOS_DIR)
                crawlExif = paramSet(args, PARAM_CRAWL_EXIF)
                requiredExifTags = argForParam(args, PARAM_REQUIRED_EXIF)
                requiredExifTags = requiredExifTags.split(",") if not requiredExifTags == None else []
                photoSize = argForParam(args, PARAM_PHOTO_SIZE, " ")
                apiKey = argForParam(args, PARAM_API_KEY)

                if apiKey is None:
                    raise ValueError("error: no api key provided")
                elif onlyMetadata and metadataPath is None:
                    raise ValueError("error: set to only crawl metadata, but no metadata path specified")
                elif not onlyMetadata and photosDir is None:
                    raise ValueError("error: no photos storage dir provided")
                
                # api-key check
                if not FlickrCrawler.testApiKey(apiKey = apiKey):
                    raise ValueError("the provided Flickr api key is invalid.")
                else:
                    crawler.apiKey = apiKey
                
                # first search photos and get photo metadata, then crawl photos if wanted
                metadata = searchPhotos(searchTerm, startPage, pageLimit, photosPerPage, metadataPath)
                
                if not onlyMetadata:
                    # crawl photos
                    crawlPhotos(metadata, photosDir, crawlExif, requiredExifTags, photoSize)

            elif paramSet(args, PARAM_PHOTOS):
                metadataPath = argForParam(args, PARAM_PHOTOS)
                photosDir = argForParam(args, PARAM_PHOTOS_DIR)
                crawlExif = paramSet(args, PARAM_CRAWL_EXIF)
                requiredExifTags = argForParam(args, PARAM_REQUIRED_EXIF)
                requiredExifTags = requiredExifTags.split(",") if not requiredExifTags == None else []
                photoSize = argForParam(args, PARAM_PHOTO_SIZE, " ")
                apiKey = argForParam(args, PARAM_API_KEY)

                if metadataPath is None:
                    raise ValueError("error: no metadata file provided")
                elif photosDir is None:
                    raise ValueError("error: no photos storage dir provided")
                elif crawlExif and apiKey is None:
                    raise ValueError("error: set to crawl exif data, but no api key provided")
                
                # api-key check
                if not apiKey is None and not FlickrCrawler.testApiKey(apiKey = apiKey):
                    raise ValueError("the provided Flickr api key is invalid.")
                else:
                    crawler.apiKey = apiKey
                
                # load metadata
                try:
                    with open(metadataPath) as json_file:
                        metadata = json.load(json_file)
                except:
                    raise ValueError("error: could not load given metadata file")

                # crawl photos
                crawlPhotos(metadata, photosDir, crawlExif, requiredExifTags, photoSize)

            else:
                printUsage()
     
        except Exception as exception:
            print(" ") 
            print(exception)
            print(" ")
