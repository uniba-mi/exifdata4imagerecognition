from Tools.Downloads.Downloader import DownloadError, DownloadResult, DownloadResultState, Downloader
import unittest
from pathlib import Path
import asyncio
from hashlib import md5
from unittest.mock import Mock

class DownloaderTests(unittest.TestCase):

    downloader = Downloader()
    testFileDirectory = Path("src/Tests/")
    testFilePath = testFileDirectory.joinpath("test.file")
    testFileUrl = "https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png"
    testFileContent = "71a50dbba44c78128b221b7df7bb51f1" # md5 hash of the 1x1 px image
    badFileUrl = "http://0.0.0.0/notThere"

    def testDownloaderThrowsWhenDownloadingInvalidUrl(self):
        self.assertRaises(DownloadError, self.downloader.download, self.badFileUrl)

    def testDownloaderDownloadsTestFileSuccessfullySync(self):
        testFile = self.downloader.download(url = self.testFileUrl)
        self.assertTrue(len(testFile) > 0)
        self.assertIsNotNone(self.downloader.lastDownloadDate)
        self.assertEqual(self.testFileContent, md5(testFile).hexdigest())
    
    def testDownloaderDownloadsTestFileToFilePathSync(self):
        self.assertFalse(self.testFilePath.exists())
        self.downloader.downloadTo(url = self.testFileUrl, path = self.testFilePath)
        self.assertTrue(self.testFilePath.exists())
        with open(self.testFilePath, "rb") as file:
            self.assertEqual(self.testFileContent, md5(file.read()).hexdigest())
        self.testFilePath.unlink()
        self.assertFalse(self.testFilePath.exists())

    def testDownloaderDownloadsTestFileSuccessfullyAsync(self):
        loop = asyncio.get_event_loop()
        testFile = loop.run_until_complete(self.downloader.downloadAsync(url = self.testFileUrl))
        self.assertTrue(len(testFile) > 0)
        self.assertEqual(self.testFileContent, md5(testFile).hexdigest())
    
    def testDownloaderDownloadsTestFileToFilePathAsync(self):
        loop = asyncio.new_event_loop()
        self.assertFalse(self.testFilePath.exists())
        loop.run_until_complete(self.downloader.downloadToAsync(url = self.testFileUrl, path = self.testFilePath))
        self.assertTrue(self.testFilePath.exists())
        with open(self.testFilePath, "rb") as file:
            self.assertEqual(self.testFileContent, md5(file.read()).hexdigest())
        self.testFilePath.unlink()
        self.assertFalse(self.testFilePath.exists())
    
    def testDownloadMultipleFilesToFilePathsAsync(self):
        loop = asyncio.get_event_loop()
        paths = {}
        
        for x in range(1, 11):
            path = self.testFileDirectory.joinpath("test" + str(x) + ".file") 
            paths[path] = self.testFileUrl
        
        loop.run_until_complete(self.downloader.downloadAllToAsync(paths))

        for path in paths:
            self.assertTrue(path.exists())
            path.unlink()
            self.assertFalse(path.exists())
    
    def testDownloadMultipleFilesToFilePathsAsyncWithBadPathContainsError(self):
        loop = asyncio.get_event_loop()
        badpath = self.testFileDirectory.joinpath("test2" + ".file") 
        paths = { badpath : self.badFileUrl, self.testFilePath : self.testFileUrl }
               
        downloadResult = loop.run_until_complete(self.downloader.downloadAllToAsync(paths))
       
        self.assertTrue(self.testFilePath.exists())
        with open(self.testFilePath, "rb") as file:
            self.assertEqual(self.testFileContent, md5(file.read()).hexdigest())
        self.testFilePath.unlink()
        self.assertFalse(self.testFilePath.exists())
        self.assertFalse(badpath.exists())
        self.assertEqual(DownloadResultState.FAILURE, downloadResult.state)
    
    def testDownloadMultipleUrlsAsyncNonBlocking(self):
        loop = asyncio.get_event_loop()
        urls = [self.testFileUrl for _ in range(1, 11)]
        results = loop.run_until_complete(self.downloader.downloadAllAsync(urls = urls))
        self.assertEqual(0, len([error for error in results if isinstance(error, DownloadError)]))
        self.assertEqual(10, len([data for data in results if isinstance(data, bytes)]))

        for result in results:
            self.assertEqual(self.testFileContent, md5(result).hexdigest())
    
    def testDownloadMultipleUrlsAsyncBlocking(self):
        urls = [self.testFileUrl for _ in range(1, 11)]
        results, time = self.downloader.downloadAll(urls = urls)
        self.assertEqual(0, len([error for error in results if isinstance(error, DownloadError)]))
        self.assertEqual(10, len([data for data in results if isinstance(data, bytes)]))
        self.assertGreater(time, 0)

        for result in results:
            self.assertEqual(self.testFileContent, md5(result).hexdigest())
    
    def testDownloadMultipleFilesToFilePathsSyncWithCallbackParams(self):
        paths = {self.testFilePath : self.testFileUrl}

        callback = Mock()
        self.downloader.downloadAllTo(urlsAndPaths = paths, callback = callback, callbackParam1 = 1, callbackParam2 = 2)

        args = callback.call_args.args
        kwargs = callback.call_args.kwargs
        downloadResult: DownloadResult = args[0]
    
        self.assertEqual(1, len(args))
        self.assertEqual(2, len(kwargs))
        self.assertEqual(1, kwargs['callbackParam1'])
        self.assertEqual(2, kwargs['callbackParam2'])
        self.assertEqual(1, callback.call_count)
        self.assertEqual(DownloadResultState.SUCCESS, downloadResult.state)
        self.assertTrue(self.testFilePath.exists())
        self.testFilePath.unlink()
        self.assertFalse(self.testFilePath.exists())
    
    def testDownloadMultipleFilesToFilePathsAsyncDictionary(self):
        loop = asyncio.get_event_loop()
        paths = {}
        
        for x in range(1, 11):
            path = self.testFileDirectory.joinpath("test" + str(x) + ".file") 
            paths[path] = self.testFileUrl
        
        callback = Mock()
        
        loop.run_until_complete(self.downloader.downloadAllToAsync(urlsAndPaths = [paths, paths, paths], 
                                                                   callback = callback))
        
        args = callback.call_args.args
        downloadResult: DownloadResult = args[0]

        self.assertEqual(paths, downloadResult.urlsAndPaths)
        self.assertEqual(1, len(args))
        self.assertEqual(3, callback.call_count)
        self.assertEqual(10, len(downloadResult.urlsAndPaths))
        self.assertEqual(0, len(downloadResult.errors))
        self.assertEqual(DownloadResultState.SUCCESS, downloadResult.state)

        for path in paths:
            self.assertTrue(path.exists())
            path.unlink()
            self.assertFalse(path.exists())
    
    def testDownloadMultipleFilesToFilePathsAsyncDictionaryWithBadPath(self):
        loop = asyncio.get_event_loop()
        badpath = self.testFileDirectory.joinpath("test2" + ".file") 
        paths = { badpath : self.badFileUrl, self.testFilePath : self.testFileUrl }
        callback = Mock()
        
        loop.run_until_complete(self.downloader.downloadAllToAsync(urlsAndPaths = [paths, paths, paths], 
                                                                   callback = callback))

        args = callback.call_args.args
        downloadResult: DownloadResult = args[0]

        self.assertEqual(paths, downloadResult.urlsAndPaths)
        self.assertEqual(1, len(args))
        self.assertEqual(3, callback.call_count)
        self.assertEqual(DownloadResultState.FAILURE, downloadResult.state)
        self.assertEqual(1, len(downloadResult.errors))
        self.assertTrue(self.testFilePath.exists())
        self.assertFalse(badpath.exists())
        self.testFilePath.unlink()
        self.assertFalse(self.testFilePath.exists())


if __name__ == '__main__': 
    unittest.main()