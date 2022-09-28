# Master Thesis Project - Ralf Lederer

While most semantic image classification approaches rely solely on raw image data to determine the contents of a photograph, EXIF metadata can also add value to image analysis. In addition to image data, many digital cameras and smartphones capture EXIF data, which contains information about lighting conditions and the camera parameters used when taking a photograph. The aim of this work is to examine whether EXIF data can improve image classification performed by *Convolutional Neural Networks*. We first analyze which EXIF tags are suitable for the classification of selected target concepts, and then incorporate EXIF data into the training process of three state of the art CNN architectures to create fusion-models. To evaluate the added value of camera metadata for image classification, we then compare the classification performance of the created fusion models with image-only and EXIF-only models. Besides the classification performance, we also evaluate economical and ecological aspects of the created models. The results show that ...

## Technical Documentation

### Flickr Crawler

The Flickr crawler is used to export image and EXIF data from the image portal Flickr and provides two basic functionalities. Photos can either be exported directly from Flickr groups or by conducting a free-text search using an arbitrary search term. Groups are identified by a unique ID and contain photos related to a particular topic. In some groups, only photos that exclusively show the group's topic may be uploaded. In other groups, however, it is also allowed to upload photos that contain objects that are not directly related to the group’s main topic, e.g. in the background of a photo. Therefore, and since the group rules are enforced with varying degrees of strictness, it can be assumed that training data collected from groups contains a certain amount of noise. When a free text search is performed, the Flickr API returns images whose title, description or user tags contain the corresponding search term. User tags are keywords used to succinctly describe the content of images. They can be added by the image authors. In addition, user tags are automatically determined and added by Flickr robots. The retrieved photos are then sorted according to their relevance determined by Flickr. Since Flickr does not define strict rules for image titles, descriptions and user tags, and the way Flickr determines the relevance of photos for search terms is unknown, it can be assumed that training data collected via free text search also contains a certain amount of images that do not match the desired target concept. Thus, training data collected with the Flickr crawler is always subject to noise if no further filtering is performed.

The crawler application can be executed via the command line, using the main entry point in **FlickrCrawlerMain.py**. The crawler is invoked with a command line parameter, which determines the function to be executed. Image exports can be carried out from a Flickr group or via free text search. Additionally, EXIF data can be exported, which requires a valid API key. The crawler offers the possibility to crawl an existing metadata file with image IDs, secrets and server information. 

```sh
python FlickrCrawlerMain.py
```

The following functions and parameters can be used:

```sh
python FlickrCrawlerMain.py -group 'groupId' of the Flickr group to crawl
```

```sh
python FlickrCrawlerMain.py -search 'search term' to use for searching photos
```

required parameters:
| parameter  | description  |
|---|---|
| -key | Flickr api key |
| -odir | directory for storing crawled photos and exif data |

optional parameters:
| parameter  | description  |
|---|---|
| -sp | start page of the group |
| -pl | page limit |
| -ppp | photos per group page (default 500) |
| -ft | additional filter tag for photos |
| -om | if set, only metadata will be crawled, no photos or exif data |
| -sm | path for storing crawled metadata file (needed when -om is set) |
| all parameters of the 'photos' function, see below, only if -om is not set |

```sh
python FlickrCrawlerMain.py -photos 'path' to metadata file containing photoIds, secrets and servers
```

required parameters:
| parameter  | description  |
|---|---|
| -odir | directory for storing crawled photos and exif data |
| -key | Flickr api key (only if exif data is to be crawled) |

optional parameters:
| parameter  | description  |
|---|---|
| -exif | if set, exif data will be crawled |
| -re | list of (required) exif tags photos must provide, separated by ',' |
| -ps | size of the crawled photos (default: ' ')
	possible values: ['s', 'q', 't', 'm', 'n', 'w', 'z', 'c', 'b', 'h', 'k', '3k', '4k', 'f', '5k', '6k', 'o', ' ']
	see: https://www.flickr.com/services/api/misc.urls.html |

 


