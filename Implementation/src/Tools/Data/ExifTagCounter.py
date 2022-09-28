from collections import Counter
from DataSource.TrainingDataSource import StaticTrainingDataSource
from DomainModel.TrainingData import EXIFImageTrainingData

class ExifTagCounter(object):
    """ Counts the distribution of exif tags in a data source containing exif image training data. """
    dataSource: StaticTrainingDataSource[EXIFImageTrainingData]

    def __init__(self, dataSource: StaticTrainingDataSource[EXIFImageTrainingData]):
        self.dataSource = dataSource
    
    def count(self) -> Counter:
        """ returns a counter object containing the distribution of exif tags in the data source. """
        tagCounts = {}

        # count the distribution of exif tags
        for instance in self.dataSource.allInstances:
            for exifTag in instance.exif.tags.keys():
                if exifTag not in tagCounts:
                    tagCounts[exifTag] = 1
                else:
                    tagCounts[exifTag] += 1
        
        return Counter(tagCounts)
