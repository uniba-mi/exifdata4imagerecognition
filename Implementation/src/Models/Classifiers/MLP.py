from typing import Tuple
from Models.Training.Generators.BatchGenerators import BatchGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2, l1
from keras.models import Model
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from Models.Classifiers.Classifier import Classifier, top_2_accuracy
import numpy as np

def createModel(inputSize: int, outputSize: int, optimizer, dropoutRate: float, alpha: float, multiLabel: bool = False) -> Model:
    """ Creates a model according to the given input parameters. """

    model = Sequential()
    model.add(Dense(256, input_dim = inputSize, activation = "relu", activity_regularizer = l1(0.0001)))
    model.add(Dense(128, activation = "relu", activity_regularizer = l2(alpha)))
    model.add(Dense(64, activation = "relu", activity_regularizer = l2(alpha)))
    model.add(Dense(32, activation = "relu", activity_regularizer = l2(alpha)))
    #model.add(Dropout(rate = dropoutRate))
    model.add(Dense(outputSize, activation = "sigmoid" if multiLabel else "softmax", activity_regularizer = l2(alpha)))

    # note: for multi-label classification we use a sigmoid activation + binary crossentropy loss
    model.compile(loss = "binary_crossentropy" if multiLabel else "categorical_crossentropy", 
                  optimizer = optimizer, 
                  metrics = [top_2_accuracy if multiLabel else "categorical_accuracy"])
    return model

def createTunedModel(inputSize: int, outputSize: int, multiLabel: bool = False) -> Model:
    """ Creates a model with tuned parameters for classyfing exif data. """
    return createModel(inputSize, 
                      outputSize, 
                      optimizer = Adam(epsilon = 1e-8, learning_rate = 0.0001), 
                      dropoutRate = 0.1, 
                      alpha = 0.0001,
                      multiLabel = multiLabel)

class MLPClassifier(Classifier):
    """ A model builder for MLP models, optimized for EXIF data classification. """

    def __init__(self, name: str, inputShape: Tuple = None, outputShape: Tuple = None, multiLabel: bool = False):
        super().__init__(inputShape = inputShape, 
                        outputShape = outputShape, 
                        name = name, 
                        modelCreationFunction = self.createOrReturnModel, #createTunedModel,
                        multiLabel = multiLabel)
    
    def createOrReturnModel(self, inputSize: int, outputSize: int, multiLabel: bool = False) -> Model:
        if self.model == None:
            return createTunedModel(inputSize = inputSize, outputSize = outputSize, multiLabel = multiLabel)
        else:
            return self.model

    def findHyperParameters(self, dataGenerator: BatchGenerator, verbose: int = 1) -> Tuple:
        """ Creates and trains multiple mlp models with the basic architecture using different hyper 
        parameters to find optimal parameters for the given data set. """

        model = KerasClassifier(model = createModel, verbose = verbose)
        trainX, trainY = np.array(dataGenerator.xDataFrame, dtype = np.float32), np.array([np.array(target) for target in dataGenerator.yDataFrame], dtype = np.float32)

        gridParams = {
            "model__inputSize": [self.inputShape],
            "model__outputSize": [self.outputShape],
            "model__optimizer": ["Adam", "Nadam"],
            "model__dropoutRate": [0.0, 0.25, 0.7],
            "model__alpha": [0.0001, 0.001],
            "batch_size": [32, 64, 128],
            "epochs": [250],
            #"optimizer__learning_rate": [0.001, 0.01, 0.1]
        }

        clf = GridSearchCV(model, param_grid = gridParams, n_jobs = 1, cv = 3, verbose = 1)
        result = clf.fit(trainX, trainY)

        # summarize results
        if verbose > 0:
            print("Hyperparameter Optimization Results:")
            print("Best Model: %f using parameters %s" % (result.best_score_, result.best_params_))
            means = result.cv_results_['mean_test_score']
            stds = result.cv_results_['std_test_score']
            params = result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        
        # return the best reached score with the used parameters
        return (result.best_score_, result.best_params_)
