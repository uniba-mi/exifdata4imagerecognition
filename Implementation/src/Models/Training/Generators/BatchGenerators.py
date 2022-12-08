from abc import ABC, abstractmethod
from typing import List, Set, Tuple
import pandas as pd
import numpy as np
from keras.utils import Sequence
from keras import Model
from keras_preprocessing import image as kimage
import copy
import time

def predictedTopKRounded(predictionY: np.ndarray, k: int = 2):
        """ returns an array with the top k values of the prediction rounded to 1.0. The remaining values will be set to 0.0. """
        topKIndizes = np.argpartition(predictionY, -k, axis = 1)[:, -k:]
        predictionYTopKRounded = np.zeros_like(predictionY)          
        np.put_along_axis(predictionYTopKRounded, topKIndizes, 1.0, axis = 1)
        return predictionYTopKRounded

def predictionYTopK(trueY: np.ndarray, predictionY: np.ndarray):
    """ returns an array which sets the maximum k values of each prediction to 1.0. K is determined based on the corresponding true label. """
    predictionYTopKOnes = np.zeros_like(predictionY)          
    for index, value in enumerate(trueY):
        k = np.count_nonzero(value == 1.0)
        topKIndizes = np.argpartition(predictionY[index], -k)[-k:]
        newValue = np.zeros_like(value)          
        np.put_along_axis(newValue, topKIndizes, 1.0, axis = 0)
        predictionYTopKOnes[index] = newValue
    return predictionYTopKOnes

class BatchGeneratorClassificationResult(object):
    """ Contains the predicted / true labels. Additionally to the predicted / true labels, the argMax predicted / true 
    label indices, as well as the rounded prediction vector and the prediction time will be returned. """

    predictionY: np.ndarray 
    predictionYMax: np.ndarray
    predictionYRounded: np.ndarray
    predictionYTop3: np.ndarray
    trueY: np.ndarray 
    trueYMax: np.ndarray
    predictionTime: float

    def __init__(self, 
                 predictionY: np.ndarray,
                 predictionYMax: np.ndarray, 
                 predictionYRounded: np.ndarray, 
                 trueY: np.ndarray, 
                 trueYMax: np.ndarray,
                 predictionTime: float):
        self.predictionY = predictionY
        self.predictionYMax = predictionYMax
        self.predictionYRounded = predictionYRounded
        self.trueY = trueY
        self.trueYMax = trueYMax
        self.predictionTime = predictionTime

class BatchGenerator(ABC, Sequence):
    """ An abstract generator for generating batch generators for keras models. 
    Expects one-hot encoded arrays in a single column data frame for y data. When the parameter ignoreColumns is set,
    the generator will remove the corresponding columns from the given training data frame. """

    xDataFrame: pd.DataFrame
    yDataFrame: pd.DataFrame
    idDataFrame: pd.DataFrame
    batchSize: int
    shuffle: bool
    size: int

    def __init__(self, xDataFrame: pd.DataFrame, yDataFrame: pd.DataFrame, idDataFrame: pd.DataFrame = None, batchSize: int = 32, ignoreColumns: Set[str] = None):
        self.xDataFrame = xDataFrame.copy()
        self.yDataFrame = yDataFrame.copy()
        self.idDataFrame = idDataFrame.copy()
        self.batchSize = batchSize
        self.shuffle = True

        # drop ignored columns
        if not ignoreColumns == None:
            self.xDataFrame.drop(ignoreColumns, axis = 1, inplace = True)
        self.size = len(self.xDataFrame)

    def on_epoch_end(self):
        if self.shuffle:
            index = np.random.permutation(self.xDataFrame.index)
            self.xDataFrame.reindex(index)
            self.yDataFrame.reindex(index)
            self.idDataFrame.reindex(index)
    
    @abstractmethod
    def __getitem__(self, index) -> Tuple:
        """ Returns a random data batch of the generator which contains batchSize elements. """
        pass
    
    def __len__(self) -> int:
        """ Returns batch count of the generator. """
        return self.size // self.batchSize 
    
    def trueLabels(self, end: int, start: int = 0):
        """ Returns the true labels of the training data instances for the given range. """
        return np.array([np.array(target) for target in self.yDataFrame.iloc[start : end]], dtype = np.float32)

    def ids(self, end: int, start: int = 0):
        """ Returns the ids of the training data instances for the given range. """
        return np.array([np.array(identifier) for identifier in self.idDataFrame.iloc[start : end]], dtype = object)
    
    def permutateX(self, columnNames: List[str]) -> "BatchGenerator":
        """ Returns a copy of the generator in which the values of the given columns are randomly shuffled (in the x-data set). """
        copiedGenerator = copy.deepcopy(self)
        for columnName in columnNames:
            copiedGenerator.xDataFrame[columnName] = np.random.permutation(self.xDataFrame[columnName].values)
        return copiedGenerator
    
    def predictOn(self, model: Model, multiLabel: bool, batchSize: int = 1) -> BatchGeneratorClassificationResult:
        """ Predicts the generators data on the given keras model and returns the classification result. """
        
        # disable shuffle for evaluation and set batchSize to 1
        self.shuffle = False
        self.batchSize = batchSize # a batch size of 1 allows to classify all elements

        # predict labels
        start = time.time()
        predictionY = model.predict(self, verbose = 0)
        duration = round((time.time() - start) / len(predictionY), 5)
        predictionYMax = np.argmax(predictionY, axis = 1)
        
        # get true labels from generator
        trueY = self.trueLabels(len(predictionY))
        trueYMax = np.argmax(trueY, axis = 1)

        if multiLabel:
            # round with threshold for each label individually
            threshold = 0.5
            predictionYRounded = np.where(predictionY >= threshold, 1, 0)
        else:
            # round prediction (this is setting the top k probabilities to 1.0 and all other to 0.0) k = 1 
            predictionYRounded = predictionYTopK(trueY, predictionY)

        return BatchGeneratorClassificationResult(predictionY, predictionYMax, predictionYRounded, trueY, trueYMax, duration)


class EXIFGenerator(BatchGenerator):
    """ A generator for generating batches of EXIF data for an EXIF data based (mlp) model. """

    def __init__(self, xDataFrame: pd.DataFrame, yDataFrame, idDataFrame: pd.DataFrame, batchSize: int = 32, ignoreColumns: Set[str] = None):
        super().__init__(xDataFrame, yDataFrame, idDataFrame, batchSize, ignoreColumns)
    
    def __getitem__(self, index) -> Tuple:
        x = self.xDataFrame[index * self.batchSize : (index + 1) * self.batchSize]
        y = self.yDataFrame[x.index]
        return np.array(x, dtype = np.float32), np.array([np.array(target) for target in y], dtype = np.float32)


class EXIFImageGenerator(BatchGenerator):
    """ A generator for generating batches of EXIF/Image data for a mixed model (cnn + mlp). """
    imageSize: Tuple
    imagePathColumn: str

    def __init__(self, xDataFrame: pd.DataFrame, yDataFrame: pd.DataFrame, idDataFrame: pd.DataFrame, imagePathColumn: str, imageSize: Tuple, batchSize: int = 32, ignoreColumns: Set[str] = None):
        super().__init__(xDataFrame, yDataFrame, idDataFrame, batchSize, ignoreColumns)
        self.imageSize = imageSize
        self.imagePathColumn = imagePathColumn
    
    def __getitem__(self, index) -> Tuple:
        x = self.xDataFrame[index * self.batchSize : (index + 1) * self.batchSize].copy()
        y = self.yDataFrame[x.index]
        
        # get image paths and then drop image path column from data set
        imagePaths = x[self.imagePathColumn]
        x.drop(self.imagePathColumn, axis = 1, inplace = True)
        
        images = []
        for path in imagePaths:
            image = kimage.load_img(path, target_size = self.imageSize)
            images.append(kimage.img_to_array(image))

        return (np.array(images), np.array(x, dtype = np.float32)), np.array([np.array(target) for target in y], dtype = np.float32)


class ImageGenerator(BatchGenerator):
    """ A generator for generating r-g-b-image data for a cnn model """
    imageSize: Tuple
    imagePathColumn: str

    def __init__(self, xDataFrame: pd.DataFrame, yDataFrame: pd.DataFrame, idDataFrame: pd.DataFrame, imagePathColumn: str, imageSize: Tuple, batchSize: int = 32, ignoreColumns: Set[str] = None):
        super().__init__(xDataFrame[imagePathColumn], yDataFrame, idDataFrame, batchSize, ignoreColumns)
        self.imageSize = imageSize
        self.imagePathColumn = imagePathColumn
    
    def __getitem__(self, index) -> Tuple:
        imagePaths = self.xDataFrame[index * self.batchSize : (index + 1) * self.batchSize]
        y = self.yDataFrame[imagePaths.index]
        
        images = []
        for path in imagePaths:
            image = kimage.load_img(path, target_size = self.imageSize)
            images.append(kimage.img_to_array(image))

        return np.array(images), np.array([np.array(target) for target in y], dtype = np.float32)