from typing import Tuple
from Transformers.Transformer import Transformer
import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrameTrainingSplitter(Transformer[pd.DataFrame, Tuple]):
    """ A training splitter splits a dataframe in stratified train / validation / test sets. Additionally, a list of
    column keys can be specified, which will be removed before the data is split. """
    targetColumn: str
    trainRatio: float
    validationRatio: float
    testRatio: float
    dropColumns: list
    seed: int
    stratify: bool

    def __init__(self, targetLabel: str, trainRatio: float = 0.75, validationRatio: float = 0.15, testRatio: float = 0.10, seed: int = 131, dropColumns = [], stratify: bool = True):
        self.targetColumn = targetLabel
        self.trainRatio = trainRatio
        self.testRatio = testRatio
        self.validationRatio = validationRatio
        self.seed = seed
        self.dropColumns = dropColumns
        self.stratify = stratify
        sum = self.trainRatio + self.validationRatio + self.testRatio
        if sum < 0.9999999 or sum > 1.0000001: # floats don't sum up to exactly one
            raise ValueError("the size of the train, validation + test set must sum up to 1.0")

    def transform(self, data: pd.DataFrame) -> Tuple:
        if len(self.dropColumns) > 0:
            for column in self.dropColumns:
                if column in data.columns:
                    data = data.drop(column, axis = 1)

        yData = data[self.targetColumn]
        xData = data.drop(self.targetColumn, axis = 1)

        # if we want the whole dataset for training, (e.g. for k-cross validation) we return it here
        if self.trainRatio == 1.0:
            return xData, yData, None, None, None, None

        try:
            # frist split in train / validation set
            trainX, valX, trainY, valY = train_test_split(xData, 
                                                            yData, 
                                                            test_size = self.validationRatio,
                                                            stratify = yData if self.stratify else None,
                                                            random_state = self.seed)

            if self.testRatio > 0.0:
                # then, split train data in train / test set
                trainX, testX, trainY, testY = train_test_split(trainX, 
                                                              trainY, 
                                                              test_size = (self.testRatio / (self.trainRatio + self.testRatio)), 
                                                              stratify = trainY if self.stratify else None,
                                                              random_state = self.seed)
                return trainX, trainY, valX, valY, testX, testY
            else:
                return trainX, trainY, valX, valY, None, None
        except Exception as e:
            if len(e.args) > 0 and "the least populated class in y has only 1 member, which is too few" in e.args[0].lower():
                print("WARNING: some class labels appear too rare to perform a stratified split, hence omitting stratification.")
                self.stratify = False
                return self.transform(data = data)
                