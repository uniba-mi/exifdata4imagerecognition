import pandas as pd
from typing import Any, List
from sklearn.preprocessing import MultiLabelBinarizer
from Transformers.Transformer import Transformer

class DataFrameLabelEncoder(Transformer[pd.DataFrame, pd.DataFrame]):
    """ A data frame label encoder one-hot encodes the target (multi class / single class) labels of a pandas data frame and 
    stores the one-hot encoded values as an array within one column of the resulting dataset. """

    targetLabel: str
    encodedLabel: str
    labelEncoder: Any
    dropTargetColumn: bool

    @property
    def classes(self) -> List[str]:
        """ Returns the names of the class labels that were currently transformed by this transformer. """
        if self.labelEncoder == None:
            return []
        return self.labelEncoder.classes_

    def __init__(self, targetLabel: str, dropTargetColumn: bool = False, encodedLabel: str = None):
        self.targetLabel = targetLabel
        if encodedLabel == None:
            self.encodedLabel = targetLabel + "_Encoded"
        else:
            self.encodedLabel = encodedLabel
        self.labelEncoder = None
        self.dropTargetColumn = dropTargetColumn

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # transform all single labels to tuples to be ready for the multilabel binarizer
        targetsToEncode = data[self.targetLabel].apply(lambda elem: (elem,) if not isinstance(elem, tuple) else elem)

        # create one-hot encoded labels
        self.labelEncoder = MultiLabelBinarizer()
        encodedTargets = self.labelEncoder.fit_transform(targetsToEncode)
        data[self.encodedLabel] = encodedTargets.tolist()

        if self.dropTargetColumn:
            data.drop(self.targetLabel, axis = 1, inplace = True)
        return data