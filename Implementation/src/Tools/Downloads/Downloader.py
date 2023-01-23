from asyncio import ensure_future
from enum import Enum
import requests
import datetime
import time
import aiohttp
import aiofiles
import asyncio
import time

class Downloader(object):
    """ A downloader is able to synchronously or asynchronously download files. """
    politenessTimeout: float
    lastDownloadDate: datetime.date
    maxConcurrentRequests: int

    def __init__(self, politenessTimeout: float = 0.05, maxConcurrentRequests: int = 10):
        self.politenessTimeout = politenessTimeout
        self.lastDownloadDate = None
        self.maxConcurrentRequests = maxConcurrentRequests

    def download(self, url: str) -> bytes:
        """ Downloads the content at the given url and returns the data as bytes. (synchronously) """
        if not self.lastDownloadDate == None:
            if (datetime.datetime.now() - self.lastDownloadDate).total_seconds() < self.politenessTimeout:
                time.sleep(self.politenessTimeout)

        try:
            self.lastDownloadDate = datetime.datetime.now()
            response = requests.get(url, stream=True)
        except Exception as exception:
            raise DownloadError(code = 1, cause = exception)
        
        if response.status_code == 200:
            return response.content
        else: 
            raise DownloadError(code = response.status_code, message = "bad response code.")

    async def downloadAsync(self, url: str, session: aiohttp.ClientSession = None) -> bytes:
        """ Downloads the content at the given url and returns the data as bytes. (asynchronously) """
        
        sessionCreated = False
        if session == None:
            sessionCreated = True
            session = aiohttp.ClientSession(trust_env = True)

        try:
            async with session.get(url = url) as response:
                if response.status == 200:
                    return await response.read()
                else: 
                    raise DownloadError(code = response.status_code, message = "bad response code.")
        finally:
            if sessionCreated:
                await session.close()
    
    def downloadAll(self, urls: list) -> list:
        """ Downloads the contents at the given urls and returns it in a list. (blocking asynchronously). Also the
        needed time is to download all urls is returned. """
        start = time.time()
        result = asyncio.get_event_loop().run_until_complete(self.downloadAllAsync(urls = urls))
        end = time.time()
        return result, end - start
    
    async def downloadAllAsync(self, urls: list) -> list:
        """ Downloads the contents at the given urls and returns it in a list. (non-blocking asynchronously) """
        session = aiohttp.ClientSession(connector = aiohttp.TCPConnector(limit = self.maxConcurrentRequests), trust_env = True)
        tasks = list()
        for url in urls:
            dlTask = ensure_future(self.downloadAsync(url = url, session = session))
            tasks.append(dlTask)
        
        try:
            return await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await session.close()

    def downloadTo(self, url: str, path: str):
        """ Downloads the content at the given url and stores it to the given file path. (synchronously) """
        try:
            downloadedFile = self.download(url = url)
            with open(path, "wb") as file:
                file.write(downloadedFile)
        except Exception as exception:
            raise DownloadError(code = 2, cause = exception)

    async def downloadToAsync(self, url: str, path: str, session: aiohttp.ClientSession = None):
        """ Downloads the content at the given url and stores it to the given file path. (asynchronously) """
        try:
            content = await self.downloadAsync(url = url, session = session)
            async with aiofiles.open(path, "wb") as file:
                await file.write(content)
        except Exception as exception:
            raise DownloadError(code = 2, cause = exception)
    
    def downloadAllTo(self, urlsAndPaths, callback = None, **callbackArgs) -> list:
        """ Executes downloadAllTo() for each entry in the given list, within one download session, in a blocking manner.  
        Also, the needed time to download all urls is returned.  """
        start = time.time()       
        result = asyncio.get_event_loop().run_until_complete(self.downloadAllToAsync(urlsAndPaths = urlsAndPaths, 
                                                                                     callback = callback, **callbackArgs))
        end = time.time()
        return result, end - start

    async def downloadAllToAsync(self, urlsAndPaths, session: aiohttp.ClientSession = None, callback = None, **callbackArgs) -> list:
        """ Downloads the given urls to the specified paths. The given urls and paths must either be of type dict or of type list (which contains multiple dicts).
        Each dict will be downloaded individually and a corresponding download result will be created and returned. If a callback is applied, it gets called with the individual results. """

        tasks = list()
        if isinstance(urlsAndPaths, dict):
            # we have a dicionary -> it should contain paths and urls
            for path in urlsAndPaths:
                tasks.append(ensure_future(self.downloadToAsync(url = urlsAndPaths[path], path = path, session = session)))
            
            # check if we need to open a new session
            openedSession = False
            if session is None:
                session = aiohttp.ClientSession(connector = aiohttp.TCPConnector(limit = self.maxConcurrentRequests), trust_env = True)
                openedSession = True
            try:
                downloadResults = await asyncio.gather(*tasks, return_exceptions=True)
                result = DownloadResult(urlsAndPaths = urlsAndPaths, state = DownloadResultState.FAILURE 
                        if any(isinstance(result, DownloadError) for result in downloadResults) else DownloadResultState.SUCCESS,
                        errors = [error for error in downloadResults if isinstance(error, DownloadError)])
                if not callback is None:
                    callback(result, **callbackArgs)
                return result
            except Exception as exception:
                raise DownloadError(code = 3, message = "callback raised an error", cause = exception)
            finally:
                if openedSession:
                    await session.close()
        elif isinstance(urlsAndPaths, list):
            # we have a list -> it should contain multiple dictionaries with paths and urls
            session = aiohttp.ClientSession(connector = aiohttp.TCPConnector(limit = self.maxConcurrentRequests), trust_env = True)
            for dictionary in urlsAndPaths:
                tasks.append(ensure_future(self.downloadAllToAsync(urlsAndPaths = dictionary, session = session, callback = callback, **callbackArgs)))
            try:
                return await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await session.close()


class DownloadResult(object):
    " a class indicating the result of an async download task. "
    state: "DownloadResultState"
    urlsAndPaths: dict
    errors: list

    def __init__(self, urlsAndPaths: dict, state: "DownloadResultState", errors: list = []):
        self.urlsAndPaths = urlsAndPaths
        self.state  = state
        self.errors = errors


class DownloadResultState(Enum):
    FAILURE = 1
    SUCCESS = 2


class DownloadError(Exception):
    cause: Exception
    errorCode: int
    message: str

    def __init__(self, code: int, message: str = None, cause: Exception = None):
        self.errorCode = code
        self.message = message
        self.cause = cause