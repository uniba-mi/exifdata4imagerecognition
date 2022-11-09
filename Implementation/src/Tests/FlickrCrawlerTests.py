import unittest
from pathlib import Path
from Tools.Flickr.FlickrCrawler import CrawlerError, FlickrCrawler, FlickrUrlBuilder
import shutil

class FlickrCrawlerTests(unittest.TestCase):

    crawler = FlickrCrawler(apiKey = "66462c38164102ed0e48a3dea4b5a12d")
    apiKey = crawler.apiKey
    flickrGroupId = "997802@N23"
    metadataDirectory = Path("resources/test_metadata/test")
    photoAndExifDirectory = Path("resources/test")

    def testBuildFlickrImageUrl(self):
        url = FlickrUrlBuilder.buildFlickrImageUrl(photoId = "4722446270", 
                                                server = "1169", 
                                                secret = "257f0fc36e", 
                                                size = "q", 
                                                format = "jpg")
        
        self.assertEqual("https://live.staticflickr.com/1169/4722446270_257f0fc36e_q.jpg", url)

    def testBuildFlickrExifUrl(self):
        url = FlickrUrlBuilder.buildFlickrExifUrl(photoId = "4722446270",
                                               secret = "257f0fc36e",
                                               apiKey = self.apiKey)
        
        self.assertEqual("https://www.flickr.com/services/rest/?method=flickr.photos.getExif&api_key=66462c38164102ed0e48a3dea4b5a12d&photo_id=4722446270&secret=257f0fc36e&format=json&nojsoncallback=1", url)

    def testBuildFlickrGroupUrl(self):
        url = FlickrUrlBuilder.buildFlickrGroupUrl(groupId = self.flickrGroupId, apiKey = self.apiKey)
        self.assertEqual("https://www.flickr.com/services/rest/?method=flickr.groups.pools.getPhotos&api_key=66462c38164102ed0e48a3dea4b5a12d&group_id=997802@N23&page=1&per_page=500&format=json&nojsoncallback=1&extras=media", url)
    
    def testBuildFlickrGroupUrlWithFilterTag(self):
        url = FlickrUrlBuilder.buildFlickrGroupUrl(groupId = self.flickrGroupId, apiKey = self.apiKey, filterTag = "cat")
        self.assertEqual("https://www.flickr.com/services/rest/?method=flickr.groups.pools.getPhotos&api_key=66462c38164102ed0e48a3dea4b5a12d&group_id=997802@N23&page=1&per_page=500&format=json&nojsoncallback=1&extras=media&tags=cat", url)

    def testBuildFlickrGroupUrlWithPage(self):
        url = FlickrUrlBuilder.buildFlickrGroupUrl(groupId = self.flickrGroupId, apiKey = self.apiKey, page = 2)
        self.assertEqual("https://www.flickr.com/services/rest/?method=flickr.groups.pools.getPhotos&api_key=66462c38164102ed0e48a3dea4b5a12d&group_id=997802@N23&page=2&per_page=500&format=json&nojsoncallback=1&extras=media", url)
    
    def testBuildFlickrPhotosSearchUrl(self):
        url = FlickrUrlBuilder.buildFlickrSearchUrl(searchTerm = "", apiKey = self.apiKey)
        self.assertEqual("https://www.flickr.com/services/rest/?method=flickr.photos.search&text=&page=1&per_page=500&format=json&nojsoncallback=1&media=photos&sort=relevance&api_key=66462c38164102ed0e48a3dea4b5a12d", url)

    def testValidApiKey(self):
        self.assertTrue(FlickrCrawler.testApiKey(apiKey = self.crawler.apiKey))
    
    def testInValidApiKey(self):
        self.assertFalse(FlickrCrawler.testApiKey(apiKey = "66462c38164102ed0e48a3dea4b5a12e"))

    def testCrawlFlickrGroupWithInvalidGroupIdThrowsError(self):
        self.assertRaises(CrawlerError, self.crawler.crawlGroup, "NotExisting@N23")

    def testCrawlFlickrGroupWithExceededStartPageThrowsError(self):
        self.assertRaises(CrawlerError, self.crawler.crawlGroup, self.flickrGroupId, 99999999)

    def testCrawlFlickrGroupWithExceededPageLimitThrowsError(self):
        self.assertRaises(CrawlerError, self.crawler.crawlGroup, self.flickrGroupId, 1, 99999999)
    
    def testCrawlFlickrGroupWithStartPageAndPageLimit(self):
        result = self.crawler.crawlGroup(groupId = self.flickrGroupId, startPage = 6, pageLimit = 5, photosPerPage = 10)
        self.assertEqual(6, result.startPage)
        self.assertEqual(10, result.endPage)
        self.assertEqual(50, len(result.imageMetadata))
        self.assertEqual(5, len(result.pageUrls))
        self.assertGreater(result.downloadTime, 0)
        filePath = result.saveMetadata(self.metadataDirectory)
        self.assertTrue(filePath.exists())
        filePath.unlink()
        self.assertFalse(filePath.exists())
        shutil.rmtree(self.metadataDirectory)
        self.assertFalse(self.metadataDirectory.exists())
    
    def testCrawlFlickrGroupWithStartPageAndPageLimitAndFilterTag(self):
        result = self.crawler.crawlGroup(groupId = self.flickrGroupId, startPage = 1, pageLimit = 1, photosPerPage = 5, filterTag = "dog")
        self.assertEqual(1, result.startPage)
        self.assertEqual(1, result.endPage)
        self.assertEqual(5, len(result.imageMetadata))
        self.assertEqual(1, len(result.pageUrls))
        self.assertGreater(result.downloadTime, 0)
        filePath = result.saveMetadata(self.metadataDirectory)
        self.assertTrue(filePath.exists())
        filePath.unlink()
        self.assertFalse(filePath.exists())
        shutil.rmtree(self.metadataDirectory)
        self.assertFalse(self.metadataDirectory.exists())
    
    def testSearchPhotosWithStartPageAndPageLimit(self):
        result = self.crawler.searchPhotos(searchTerm = "outdoor cat", startPage = 2, pageLimit = 5, photosPerPage = 10)
        self.assertEqual(2, result.startPage)
        self.assertEqual(6, result.endPage)
        self.assertEqual(50, len(result.imageMetadata))
        self.assertEqual(5, len(result.pageUrls))
        self.assertGreater(result.downloadTime, 0)
        filePath = result.saveMetadata(self.metadataDirectory)
        self.assertTrue(filePath.exists())
        filePath.unlink()
        self.assertFalse(filePath.exists())
        shutil.rmtree(self.metadataDirectory)
        self.assertFalse(self.metadataDirectory.exists())
    
    def testDownloadImagesAndExifFilesForMetadata(self):
        # metadata contains 8 fine images and 2 images without valid exif data
        metadata = [{"id": "52208725281", "secret": "9a0c70d844", "server": "65535"}, 
                    {"id": "52206796078", "secret": "f1ef0d1a6f", "server": "65535"}, 
                    {"id": "52204869774", "secret": "43fc565fa5", "server": "65535"}, 
                    {"id": "52207926785", "secret": "1aa57da50f", "server": "65535"}, 
                    {"id": "52207382498", "secret": "129ff2a624", "server": "65535"}, 
                    {"id": "52202848597", "secret": "1bba676535", "server": "65535"}, 
                    {"id": "52206888963", "secret": "e60df89ff0", "server": "65535"}, 
                    {"id": "52205699512", "secret": "62bc347bda", "server": "65535"}, 
                    {"id": "51592704136", "secret": "5eaac3ca16", "server": "65535"}, 
                    {"id": "52206332666", "secret": "8313287b82", "server": "65535"}]

        validIds = ["52206796078", "52204869774", "52207382498", "52202848597", "52206888963", "52205699512", "51592704136", "52206332666"]

        # crawl photos and exif with metadata
        result = self.crawler.crawlPhotos(metaData = metadata, 
                                          directory = self.photoAndExifDirectory, 
                                          crawlExif = True)

        self.assertTrue(self.photoAndExifDirectory.exists())
        self.assertEqual(8, result.crawledPhotoCount)
        self.assertEqual(validIds, result.crawledPhotoIds)
        shutil.rmtree(self.photoAndExifDirectory)
        self.assertFalse(self.photoAndExifDirectory.exists())
    

    def testDownloadImagesAndExifFilesForMetadataWithRequiredExifTagList(self):
        # metadata contains 7 images with all exif tags, 1 image without all tags
        metadata = [{"id": "52206796078", "secret": "f1ef0d1a6f", "server": "65535"}, 
                    {"id": "52204869774", "secret": "43fc565fa5", "server": "65535"}, 
                    {"id": "52207382498", "secret": "129ff2a624", "server": "65535"}, 
                    {"id": "52202848597", "secret": "1bba676535", "server": "65535"}, 
                    {"id": "52206888963", "secret": "e60df89ff0", "server": "65535"}, 
                    {"id": "52205699512", "secret": "62bc347bda", "server": "65535"}, 
                    {"id": "51592704136", "secret": "5eaac3ca16", "server": "65535"}, 
                    {"id": "52206332666", "secret": "8313287b82", "server": "65535"}]

        validIds = ["52206796078", "52204869774", "52207382498", "52202848597", "52206888963", "52205699512", "51592704136"]

        # crawl photos and exif with metadata
        result = self.crawler.crawlPhotos(metaData = metadata, 
                                          directory = self.photoAndExifDirectory, 
                                          crawlExif = True,
                                          requiredExifTags = ["ISO", "FocalLength", "Flash"])

        self.assertTrue(self.photoAndExifDirectory.exists())
        self.assertEqual(7, result.crawledPhotoCount)
        self.assertEqual(validIds, result.crawledPhotoIds)
        shutil.rmtree(self.photoAndExifDirectory)
        self.assertFalse(self.photoAndExifDirectory.exists())
    
    def testDownloadImagesForMetadata(self):
        metadata = [{"id": "52208725281", "secret": "9a0c70d844", "server": "65535"}, 
                    {"id": "52206796078", "secret": "f1ef0d1a6f", "server": "65535"}, 
                    {"id": "52204869774", "secret": "43fc565fa5", "server": "65535"}, 
                    {"id": "52207926785", "secret": "1aa57da50f", "server": "65535"}, 
                    {"id": "52207382498", "secret": "129ff2a624", "server": "65535"}, 
                    {"id": "52202848597", "secret": "1bba676535", "server": "65535"}, 
                    {"id": "52206888963", "secret": "e60df89ff0", "server": "65535"}, 
                    {"id": "52205699512", "secret": "62bc347bda", "server": "65535"}, 
                    {"id": "51592704136", "secret": "5eaac3ca16", "server": "65535"}, 
                    {"id": "52206332666", "secret": "8313287b82", "server": "65535"}]
            
        validIds = [elem["id"] for elem in metadata]

        # crawl photos with metadata
        result = self.crawler.crawlPhotos(metaData = metadata, 
                                          directory = self.photoAndExifDirectory)

        self.assertTrue(self.photoAndExifDirectory.exists())
        self.assertEqual(10, result.crawledPhotoCount)
        self.assertEqual(validIds, result.crawledPhotoIds)
        shutil.rmtree(self.photoAndExifDirectory)
        self.assertFalse(self.photoAndExifDirectory.exists())
    
    def testLicenceCheckForMetadata(self):
        license, visibility = self.crawler.checkLicense(photoId = "52208725281")
        self.assertEqual(license, 0)
        self.assertEqual(visibility, 1)

if __name__ == '__main__':
    unittest.main()
