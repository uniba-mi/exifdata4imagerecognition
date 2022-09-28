from typing import List
from Tools.Downloads.Downloader import DownloadResult, DownloadResultState, Downloader, DownloadError
from Tools.Flickr.FlickrTools import readExifTags, FlickrCodingKeys
import json
from pathlib import Path
from datetime import date

class FlickrCrawler(object):
    """ A flicker crawler can be used to extract images and exif data from the image portal flickr (https://flickr.com/). 
    It can be used to extract images / exif data of a specific group or download images / exif data using a 
    metadata file which specifies the (image-id, server, secret) of images to be downloaded. """

    downloader: Downloader
    apiKey: str
    encoding: str = "utf-8"
    chunkSize: int = 50 # chunk size for image/exif pairs
    
    def __init__(self, apiKey: str = None):
        self.downloader = Downloader()
        self.apiKey = apiKey

    @staticmethod
    def testApiKey(apiKey: str) -> bool:
        """ Verifies whether the given api key is a valid Flickr api key. """
        if not apiKey is None:
            try:
                # use a simple exif url to check if the api key is valid
                testUrl = FlickrUrlBuilder.buildFlickrTestApiKeyUrl(apiKey = apiKey)
                jsonResponse = json.loads(str(Downloader().download(url = testUrl), FlickrCrawler.encoding))

                # check for the code of an invalid api key
                if "code" in jsonResponse and jsonResponse["code"] != 100: 
                    return True
            except Exception as exception:
                raise CrawlerError(message = "error while executing apiKey test request", cause = exception)
        
        return False
    
    def __retrievePageInformation(self, pageUrl: str) -> tuple:
        """ Retrieves the page information of the given Flickr photo page url. This includes the amount of pages
        and the total number of photos. """
        totalPages = 0
        totalPhotos = 0
        try:                                              
            content = str(self.downloader.download(url = pageUrl), self.encoding)
            jsonContent = json.loads(content)
            if "photos" in jsonContent:
                totalPhotos = jsonContent["photos"]["total"]
                totalPages = jsonContent["photos"]["pages"]
        except DownloadError as exception:
            raise CrawlerError("error while downloading photo page overview url", cause = exception)
        except Exception as exception:
            raise CrawlerError("error while parsing photo page overview data", cause = exception)
        return totalPages, totalPhotos
    
    def __downloadPagePhotoMetadata(self, pageUrls: List[str]) -> List[dict]:
        """ Downloads the photo information of the given Flickr photo page urls and 
        returns the crawled metadata, the number of photos with wrong media type and the download time.  """
        # download pages
        downloadResults, downloadTime = self.downloader.downloadAll(urls = pageUrls)

        crawledData = []
        wrongMediaFiles = 0

        # parse downloaded pages
        for result in downloadResults:
            if isinstance(result, bytes):
                jsonContent = json.loads(str(result, self.encoding))
                for photo in jsonContent["photos"]["photo"]:
                    if "media" in photo and photo["media"] != "photo":
                        wrongMediaFiles += 1
                    else:
                        photoMetadata = {FlickrCodingKeys.IMAGE_ID.value: photo["id"],
                                         FlickrCodingKeys.SECRET.value: photo["secret"],
                                         FlickrCodingKeys.SERVER.value: photo["server"]}
                        crawledData.append(photoMetadata)
            elif isinstance(result, DownloadError):
                raise CrawlerError("the crawler failed to download at least one page", cause = result)
        
        return crawledData, wrongMediaFiles, downloadTime

    def crawlPhotos(self, metaData: list, directory: Path, sizeKey: str = " ", crawlExif: bool = False, 
                    photosSubDirectory: str = "images", exifSubDirectory: str = "exif", format: str = "jpg", 
                    requiredExifTags: list = []):
        """ Crawls the image files for the given metadata and stores them in the given directory.
        The metadata must provide image ids, server and secret information.
        The exif data can only be crawled if an api key was provided to the crawler. 
        If exif data is crawled, the corresponding image will only be crawled if Flickr provides exif data for that image. 
        For crawling exif data, a list with required exif tags can be applied.
        The size parameter must comply to the specification found at: https://www.flickr.com/services/api/misc.urls.html """

        if sizeKey not in FlickrUrlBuilder.validSizeKeys:
            raise ValueError("the given size key " + sizeKey + " is invalid, possible values = " + FlickrUrlBuilder.validSizeKeys)

        if self.apiKey is None and crawlExif:
            raise RuntimeError("no api key was provided to the crawler, exif crawling is not possible.")
        
        # first, create the destination directory and the sub directories
        photoDirectory = directory.joinpath(photosSubDirectory)
        photoDirectory.mkdir(parents = True, exist_ok = True)

        exifDirectory = directory.joinpath(exifSubDirectory)
        if crawlExif:
            exifDirectory.mkdir(parents = True, exist_ok = True)

        urlsAndPaths = []
        
        # create urls and file paths
        for photoMetaData in metaData:
            photoId = photoMetaData[FlickrCodingKeys.IMAGE_ID.value]
            photoServer = photoMetaData[FlickrCodingKeys.SERVER.value]
            photoSecret = photoMetaData[FlickrCodingKeys.SECRET.value]

            currentUrlAndPath = {}

            photoUrl = FlickrUrlBuilder.buildFlickrImageUrl(photoId = photoId, 
                                                            server = photoServer, 
                                                            secret = photoSecret,
                                                            size = sizeKey, 
                                                            format = format)
            
            photoPath = photoDirectory.joinpath("{photoId}_{photoSecret}_{photoServer}_{size}.{format}".format(
                photoId = photoId, photoSecret = photoSecret, photoServer = photoServer, size = sizeKey, format = format))
            
            currentUrlAndPath[photoPath] = photoUrl
            
            if crawlExif:
                exifUrl = FlickrUrlBuilder.buildFlickrExifUrl(photoId = photoId, secret = photoSecret, apiKey = self.apiKey)
                exifPath = exifDirectory.joinpath("{photoId}.json".format(photoId = photoId))
                currentUrlAndPath[exifPath] = exifUrl
            
            urlsAndPaths.append(currentUrlAndPath)

        # download image / exif data in chunks
        for i in range(0, len(urlsAndPaths), self.chunkSize):
            chunk = urlsAndPaths[i:i + self.chunkSize]
            result = self.downloader.downloadAllTo(urlsAndPaths = chunk, 
                                                         callback = self.__handleImageDownloadResult, 
                                                         exifDir = exifSubDirectory if crawlExif else None,
                                                         requiredExifTags = requiredExifTags)

        return PhotoCrawlResult(crawledPhotosPaths = [next(iter(paths)) for paths in urlsAndPaths], downloadTime = result[1])

    def __handleImageDownloadResult(self, downloadResult: DownloadResult, exifDir: str, requiredExifTags: list):
        """ Handles a download result. If the download contains errors, all files of the download will be removed. 
        A download contains exactly one image (and possibly the exif data for that image). """
        badResult = False
        if downloadResult.state == DownloadResultState.SUCCESS:
            # we downloaded a pair of (image, exif) data -> verify the exif data, if we crawled it
            exifPath = next(iter([path for path in downloadResult.urlsAndPaths if not exifDir is None and exifDir in str(path)]), None)
            if not exifPath is None:
                tags = readExifTags(filePath = exifPath)
                if not tags is None:
                    # check if all desired exif tags are there
                    badResult = not all(requiredTag in tags for requiredTag in requiredExifTags)
                else:
                    badResult = True
        else:
            badResult = True
        
        # if we have a bad result (download errors, bad exif tags, we delete the image and the exif tags)
        if badResult:
            for path in downloadResult.urlsAndPaths:
                if path.exists():
                    path.unlink()
            
    def crawlGroup(self, groupId: str, startPage: int = 1, pageLimit: int = 0, filterTag: str = None, photosPerPage: int = 500) -> "GroupCrawlResult":
        """ Crawls a flickr group with the given parameters and returns a group crawl result. 
        The data can only be crawled if an api key was provided to the crawler """
        if self.apiKey is None:
            raise RuntimeError("no api key was provided to the crawler, group crawling is not possible.")
        
        groupOverviewUrl = FlickrUrlBuilder.buildFlickrGroupUrl(groupId = groupId, 
                                                                apiKey = self.apiKey, 
                                                                filterTag = filterTag,
                                                                page = 1,
                                                                perPage = photosPerPage)
       
        # download the first group page to gather information about the amount of pages and photos
        totalPages, totalPhotos = self.__retrievePageInformation(pageUrl = groupOverviewUrl)
       
        if totalPages > 0 and totalPhotos > 0:
            # check if our start page is valid
            if startPage > totalPages:
                raise CrawlerError("the given start page exceeds the amount of pages in the group: " + str(totalPages))

            # if we have a limit, calculate the pages to crawl
            endPage = totalPages
            if pageLimit > 0:
                endPage = startPage + pageLimit - 1
                if endPage > totalPages:
                   raise CrawlerError("the page limit exceeds the amount of pages in the group: " + str(totalPages))
            
            # create download urls
            urls = [FlickrUrlBuilder.buildFlickrGroupUrl(groupId = groupId, 
                                                         apiKey = self.apiKey,
                                                         filterTag = filterTag,
                                                         page = pageNumber,
                                                         perPage = photosPerPage) 
                    for pageNumber in range(startPage, endPage + 1)]
            
            # download pages
            metaData, wrongMediaFiles, downloadTime = self.__downloadPagePhotoMetadata(pageUrls = urls)
        
            # create and return crawl result
            return GroupCrawlResult(imageMetadata = metaData, 
                                    startPage = startPage, 
                                    endPage= endPage, 
                                    wrongMedia = wrongMediaFiles,
                                    pageUrls = urls,
                                    downloadTime = downloadTime,
                                    groupId = groupId,
                                    photosPerPage = photosPerPage,
                                    tag = filterTag)
            
        else:
            raise CrawlerError("the group " + groupId + " does not exist")
    
    def searchPhotos(self, searchTerm: str, startPage: int = 1, pageLimit: int = 0, photosPerPage: int = 500) -> "PhotosSearchResult":
        """ Searches for photos on flickr matching the given search term. """
        if self.apiKey is None:
            raise RuntimeError("no api key was provided to the crawler, photo search is not possible.")

        photosSearchUrl = FlickrUrlBuilder.buildFlickrSearchUrl(searchTerm = searchTerm,                                                        
                                                                page = 1,
                                                                perPage = photosPerPage,
                                                                apiKey = self.apiKey)

        # download the first group page to gather information about the amount of pages and photos
        totalPages, totalPhotos = self.__retrievePageInformation(pageUrl = photosSearchUrl)

        if totalPages > 0 and totalPhotos > 0:
            # check if our start page is valid
            if startPage > totalPages:
                raise CrawlerError("the given start page exceeds the amount of pages of the search result: " + str(totalPages))

            # if we have a limit, calculate the pages to crawl
            endPage = totalPages
            if pageLimit > 0:
                endPage = startPage + pageLimit - 1
                if endPage > totalPages:
                   raise CrawlerError("the page limit exceeds the amount of pages in the search result: " + str(totalPages))

            # create download urls
            urls = [FlickrUrlBuilder.buildFlickrSearchUrl(searchTerm = searchTerm, 
                                                          page = pageNumber,
                                                          perPage = photosPerPage,
                                                          apiKey = self.apiKey) 
                    for pageNumber in range(startPage, endPage + 1)]
            
            # download pages
            metaData, wrongMediaFiles, downloadTime = self.__downloadPagePhotoMetadata(pageUrls = urls)
        
            # create and return crawl result
            return PhotosSearchResult(imageMetadata = metaData, 
                                      startPage = startPage, 
                                      endPage= endPage, 
                                      wrongMedia = wrongMediaFiles,
                                      pageUrls = urls,
                                      downloadTime = downloadTime,
                                      searchTerm = searchTerm,
                                      photosPerPage = photosPerPage)
        else:
            raise CrawlerError("no photos found for the search term: " + searchTerm)


class GroupCrawlResult(object):
    """ A group crawl result provides the result of a Flickr group crawling process. It provides the metadata of the crawled images,
    but not the images themselves. The metadata (image ids, server, secret) can be used to download the corresponding images. """
    imageMetadata: list
    startPage: int
    endPage: int
    photosPerPage: int
    wrongMedia: int
    pageUrls: list
    downloadTime: float
    groupId: str
    tag: str

    def __init__(self, imageMetadata: list, startPage: int, endPage: int, 
                    wrongMedia: int, pageUrls: list, downloadTime: float, groupId: str, photosPerPage: int, tag: str = None):
        self.imageMetadata = imageMetadata
        self.startPage = startPage
        self.endPage = endPage
        self.wrongMedia = wrongMedia
        self.pageUrls = pageUrls
        self.downloadTime = downloadTime
        self.groupId = groupId
        self.photosPerPage = photosPerPage
        self.tag = tag
    
    def saveMetadata(self, directory: Path) -> Path:
        """ stores the image metadata of the crawl result as json file in the given directory. Returns the filePath of the
        created file """
        fileName = "{groupId}_{startPage}_{endPage}_{photosPerPage}_{date}".format(
            groupId = self.groupId, startPage = self.startPage, endPage = self.endPage, photosPerPage = self.photosPerPage, date = date.today())

        if not self.tag is None:
            fileName += "_" + self.tag
        
        fileName += ".json"

        filePath = directory.joinpath(fileName)
        
        directory.mkdir(parents = True, exist_ok = True)
        with open(filePath, "w") as file:
            json.dump(self.imageMetadata, file)
        return filePath


class PhotoCrawlResult(object):
    """ A photo crawl result provides the results of a Flickr photo crawling process. """
    crawledPhotoCount: int
    crawledPhotoIds: list
    downloadTime: float

    def __init__(self, crawledPhotosPaths: list, downloadTime: float):
        self.crawledPhotoIds = []
        self.crawledPhotoCount = 0
        self.downloadTime = downloadTime

        for path in crawledPhotosPaths:
            if path.exists():
                self.crawledPhotoCount += 1
                self.crawledPhotoIds.append(path.name.split("_")[0])


class PhotosSearchResult(object):
    """ A photos search result provides the result of a Flickr photo search crawling process. It provides the metadata of the crawled images,
    but not the images themselves. The metadata (image ids, server, secret) can be used to download the corresponding images. """
    imageMetadata: list
    startPage: int
    endPage: int
    photosPerPage: int
    wrongMedia: int
    pageUrls: list
    downloadTime: float
    searchTerm: str

    def __init__(self, imageMetadata: list, startPage: int, endPage: int, 
                    wrongMedia: int, pageUrls: list, downloadTime: float, searchTerm: str, photosPerPage: int):
        self.imageMetadata = imageMetadata
        self.startPage = startPage
        self.endPage = endPage
        self.wrongMedia = wrongMedia
        self.pageUrls = pageUrls
        self.downloadTime = downloadTime
        self.searchTerm = searchTerm
        self.photosPerPage = photosPerPage
    
    def saveMetadata(self, directory: Path) -> Path:
        """ stores the image metadata of the crawl result as json file in the given directory. Returns the filePath of the
        created file """
        fileName = "{searchTerm}_{startPage}_{endPage}_{photosPerPage}_{date}.json".format(
            searchTerm = self.searchTerm, startPage = self.startPage, endPage = self.endPage, photosPerPage = self.photosPerPage, date = date.today())

        filePath = directory.joinpath(fileName)
        
        directory.mkdir(parents = True, exist_ok = True)
        with open(filePath, "w") as file:
            json.dump(self.imageMetadata, file)
        return filePath
                

class CrawlerError(Exception):

    cause: Exception
    message: str

    def __init__(self, message: str = None, cause: Exception = None):
        self.message = message
        self.cause = cause


class FlickrUrlBuilder(object):

    validSizeKeys = ["s", "q", "t", "m", "n", "w", "z", "c", "b", "h", "k", "3k", "4k", "f", "5k", "6k", "o", " "]

    @staticmethod
    def buildFlickrImageUrl(photoId: str, server: str, secret: str, size: str, format: str = "jpg") -> str:
        """ Builds a Flickr image url with the given parameters. The size parameter must comply to the specification
        found at: https://www.flickr.com/services/api/misc.urls.html  """
        if size not in FlickrUrlBuilder.validSizeKeys:
            raise ValueError("the given size key " + size + " is invalid, possible values = " + FlickrUrlBuilder.validSizeKeys)
        url = "https://live.staticflickr.com/{server}/{photoId}_{secret}".format(
            server = server, photoId = photoId, secret = secret, size = size, format = format)
        if not size == " ":
            url += "_{size}".format(size = size)
        url += ".{format}".format(format = format)
        return url
    
    @staticmethod
    def buildFlickrExifUrl(photoId: str, secret: str, apiKey: str) -> str:
        """ Builds a Flickr exif data url with the given parameters. To retrieve exif data from Flickr, a valid api key is required. """
        return "https://www.flickr.com/services/rest/?method=flickr.photos.getExif" \
                "&api_key={apiKey}&photo_id={photoId}&secret={secret}&format=json&nojsoncallback=1".format(
                apiKey = apiKey,  photoId = photoId, secret = secret)
    
    @staticmethod
    def buildFlickrGroupUrl(groupId: str, apiKey: str, filterTag: str = None, page: int = 1, perPage: int = 500) -> str:
        """ Builds a Flickr group url with the given parameters. Querying the url will return a list of pages which contain metadata 
        that can be used to download the group images. The optional parameter filterTag can be used to filter images. """
        url = "https://www.flickr.com/services/rest/?method=flickr.groups.pools.getPhotos" \
                   "&api_key={apiKey}&group_id={groupId}&page={page}&per_page={perPage}&format=json&nojsoncallback=1&extras=media".format(
                    groupId = groupId, apiKey = apiKey, page = str(page), perPage = str(perPage))
        if not filterTag is None:
            url += "&tags={filterTag}".format(filterTag = filterTag)
        return url 
    
    @staticmethod
    def buildFlickrSearchUrl(searchTerm: str, page: int = 1, perPage: int = 500, apiKey: str = None) -> str:
        """ Builds a Flickr search url with the given parameters. Querying the url will return a list of pages which contain metadata 
        that can be used to download the group images. The optional parameter filterTag can be used to filter images. """
        url = "https://www.flickr.com/services/rest/?method=flickr.photos.search" \
                   "&text={searchTerm}&page={page}&per_page={perPage}&format=json&nojsoncallback=1&media=photos&sort=relevance".format(
                    searchTerm = searchTerm, page = str(page), perPage = str(perPage))
        if not apiKey is None:
            url += "&api_key={apiKey}".format(apiKey = apiKey)
        return url 
    
    @staticmethod
    def buildFlickrTestApiKeyUrl(apiKey: str) -> bool:
        """ Builds a Flickr url that can be used to check whether the given api key is valid. """
        # use an exif url without any metadata for photos to check the api key
        return FlickrUrlBuilder.buildFlickrExifUrl("", "", apiKey = apiKey)