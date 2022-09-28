from typing import Any, List
from DomainModel.ExifData import ExifData
from Transformers.Transformer import TransformError, Transformer

class ExifTagFilter(Transformer[ExifData, ExifData]):
    """ Returns a copy of the exif data object containing only the given exif tags. 
        If a dummy value is applied, the value will be set for tags which are not contained 
        in filtered objects, but are present in the filter list, otherwise an exception will be raised. """

    filterTags: List[str]
    allowMissingValues: bool
    dummyValue: Any

    def __init__(self, filterTags: List[str], dummyValue: Any = None):
        self.filterTags = filterTags
        self.dummyValue = dummyValue        

    def transform(self, data: ExifData) -> ExifData:
        newTags = {}
        tagNames = data.tagNames
        for tag in self.filterTags:
            if tag in tagNames:
                newTags[tag] = data.tags[tag]                
            elif not self.dummyValue is None:
                newTags[tag] = self.dummyValue
            else:
                raise TransformError("tag '" + tag + "' is not included in the exif data.")
        
        # check if all values contain the dummy value, if so we will raise an exception
        uniqueValues = list(set(list(newTags.values())))
        if not self.dummyValue is None and len(self.filterTags) > 1 and len(uniqueValues) == 1 and uniqueValues[0] == self.dummyValue:
            raise TransformError("no filter tag is included in the exif data.")

        return ExifData(tags = newTags, id = data.id)
    