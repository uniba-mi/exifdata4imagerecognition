from DataSource.TrainingDataSource import TrainingDataSource
from DomainModel.TrainingData import EXIFImageTrainingData
from Transformers.Transformer import Transformer
import pandas as pd

class ExiImageDataSourceToDataFrameTransformer(Transformer[TrainingDataSource[EXIFImageTrainingData], pd.DataFrame]):
    """ Transforms a data source of type EXIFImageTrainingData to a dataframe describing the exif tags and the image
     of the training instances of the data source. The data frame does not contain the image data, but the image path. """

    targetLabel: str
    useSuperConceptNames: bool
    imagePathColumnName: str
    idColumnName: str
    verbose: int

    def __init__(self, targetLabel: str, useSuperConceptNames: bool = False, imagePathColumnName: str = "ImagePath", idColumnName: str = "id", verbose: int = 1):
        self.targetLabel = targetLabel
        self.useSuperConceptNames = useSuperConceptNames
        self.imagePathColumnName = imagePathColumnName
        self.idColumnName = idColumnName
        self.verbose = verbose

    def transform(self, data: TrainingDataSource[EXIFImageTrainingData]) -> pd.DataFrame:
        dataFrames = []
        labels = data.getSuperConceptNames() if self.useSuperConceptNames else data.getConceptNames()
        
        # check if we have multi-label classification targets
        multilabel = any(isinstance(label, tuple) for label in labels)

        for concept in data.getConcepts():
            dataFrame = pd.DataFrame([{**{self.idColumnName : instance.image.id}, **instance.exif.tags, **{self.imagePathColumnName : instance.image.path}} for instance in concept.instances])
            target = concept.superConceptNames if self.useSuperConceptNames else concept.names
            
            if multilabel:
                # convert single labels to tuple
                if not isinstance(target, tuple):
                    target = (target, )
                dataFrame[self.targetLabel] = [target] * len(dataFrame)
            else:
                dataFrame[self.targetLabel] = target
            
            dataFrames.append(dataFrame)
  
        dataFrame = pd.concat(dataFrames)
        dataFrame.reset_index(drop = True, inplace = True)

        # remove columns that contain nan-values only
        for column in dataFrame.columns:
            if dataFrame[column].isnull().values.all():
                if self.verbose:
                    print("info: removing column '" + column + "' from data frame because it contains only nan-values")
                dataFrame.drop(column, axis = 1, inplace = True)
        return dataFrame
