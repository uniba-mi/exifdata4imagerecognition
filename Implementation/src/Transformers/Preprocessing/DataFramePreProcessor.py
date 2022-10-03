from typing import Dict, List
from Transformers.Transformer import Transformer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class TabularDataFramePreProcessor(Transformer[pd.DataFrame, pd.DataFrame]):
    """ The preprocessor applies encoding and scaling for all features of the given tabular data set. 
    Additionaly, imputing for missing values (np.nan) can be performed. For imputing binary and categorical features, 
    the most frequent value will be used, for numerical features, the mean value of the corresponding column.
    One-Hot encoding will be performed for binary and categorical features, while for numerical features standard scaling 
    will be applied. Optionally, a dictionary with categorical feature counts can be injected into the pre-processor. 
    The dictionary must contain the name of the categorical columns as keys and the number of possible values for 
    the featues as integer value. This can be helpfull when there are missing categories in categorical features that don't appear
    in the encoded data set."""
    numericalColumns: List[str]
    categoricalColumns: List[str]
    binaryColumns: List[str]
    impute: bool
    oneHotEncode: bool
    scale: bool
    categoricalFeatureCounts: Dict[str, int]

    def __init__(self, 
                numericalColumns: List[str], 
                categoricalColumns: List[str], 
                binaryColumns: List[str], 
                impute: bool = True, 
                oneHoteEncode: bool = True, 
                scale: bool = True, 
                categoricalFeatureCounts = None):
        self.numericalColumns = numericalColumns
        self.categoricalColumns = categoricalColumns
        self.binaryColumns = binaryColumns
        self.impute = impute
        self.oneHotEncode = oneHoteEncode
        self.scale = scale
        self.categoricalFeatureCounts = categoricalFeatureCounts

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.impute:
            data = self.__impute(dataFrame = data)
        if self.oneHotEncode:
            data = self.__oneHotEncode(dataFrame = data)
        if self.scale:
            data = self.__scale(dataFrame = data)
        return data

    def __impute(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        mostFrequentImputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
        meanImputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

        # impute categorical and binary features
        if len(self.binaryColumns + self.categoricalColumns) > 0:
            for column in self.binaryColumns + self.categoricalColumns:
                if column in dataFrame.columns:
                    dataFrame[column] = mostFrequentImputer.fit_transform(dataFrame[[column]]).ravel()

        # impute numerical features
        if len(self.numericalColumns) > 0:
            for column in self.numericalColumns:
                if column in dataFrame.columns:
                    dataFrame[column] = meanImputer.fit_transform(dataFrame[[column]]).ravel()

        return dataFrame

    def __oneHotEncode(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        # transform categorical features and binary features with one-hot encoding
        if len(self.binaryColumns + self.categoricalColumns) > 0:
            for column in self.binaryColumns + self.categoricalColumns:
                if column in dataFrame.columns:
                    # encode each categorical feature depending on the number of categories present in the data set
                    if not self.categoricalFeatureCounts == None and column in self.categoricalFeatureCounts.keys():
                        # we have a fixed number of possible values for the current categorical feature
                        # we encode it with this number of values instead of the number of values that appears in the data set
                        dataFrame[column] = dataFrame[column].astype(pd.CategoricalDtype(categories = list(range(0, self.categoricalFeatureCounts[column]))))
                    dataFrame = pd.get_dummies(dataFrame, columns = [column])
        return dataFrame
    
    def __scale(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        # apply standard scaler to numerical features, leave other columns as they are
        if len(self.numericalColumns) > 0:
            for column in self.numericalColumns:
                if column in dataFrame.columns:
                    columnTransformer = ColumnTransformer([("ct", StandardScaler(), [column])], remainder = "passthrough")
                    dataFrame[column] = columnTransformer.fit_transform(dataFrame[[column]]).ravel()
        return dataFrame
