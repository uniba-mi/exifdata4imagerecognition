from typing import Tuple
from keras.layers import Dense, concatenate
from keras.models import Model
from Models.Classifiers.Classifier import Classifier, FineTuneClassifier, top_2_accuracy

def createMixedModel(inputShape: Tuple, outputShape: Tuple, clf1: Classifier, clf2: Classifier, multiLabel: bool = False) -> Model:
    """  Generates a concatenated model using the given classifiers and adds a new classification head.
    The input shape must must match both input shapes of the individual classifiers. """

    # create cnn and mlp model
    firstModel = clf1.modelCreationFunction(inputShape[0], outputShape, multiLabel)
    secondModel = clf2.modelCreationFunction(inputShape[1], outputShape, multiLabel)

    # create a new classification head with combined outputs from both models + an additional dense layer
    outputLayer = concatenate([firstModel.layers[-2].output, secondModel.layers[-2].output])
    outputLayer = Dense(50, activation = "relu")(outputLayer)
    outputLayer = Dense(outputShape, activation = "sigmoid" if multiLabel else "softmax")(outputLayer)

    # create a new mixed model
    model = Model(inputs = [firstModel.input, secondModel.input], outputs = outputLayer)
    model.compile(optimizer = "adam", 
                  loss = "binary_crossentropy" if multiLabel else "categorical_crossentropy", 
                  metrics = [top_2_accuracy if multiLabel else "categorical_accuracy"])

    return model
    
class MixedModel(FineTuneClassifier):
    """ A mixded-model classifier generates a concatenated model using the two given classifiers. It removes the 
    classification head of both classifiers and adds a new one to support the desired classification task.
    During training, fine-tuning will be applied (see FineTuneClassifier). """

    clf1: Classifier
    clf2: Classifier

    def __init__(self, name: str, 
                       classifier1: Classifier,
                       classifier2: Classifier,
                       fineTuneLayersCount: int = 0, 
                       fineTuneEpochs: int = 0,
                       multiLabel: bool = False):

        super().__init__(name = name, 
                         modelCreationFunction = lambda inputShape, outputShape, multiLabel: createMixedModel(inputShape = inputShape,   
                                                                                                              outputShape = outputShape,
                                                                                                              clf1 = self.clf1, 
                                                                                                              clf2 = self.clf2,
                                                                                                              multiLabel = multiLabel),
                        fineTuneLayersCount = fineTuneLayersCount,
                        fineTuneEpochs = fineTuneEpochs,
                        inputShape = (classifier1.inputShape, classifier2.inputShape),
                        outputShape = classifier1.outputShape,
                        multiLabel = multiLabel)

        self.clf1 = classifier1
        self.clf2 = classifier2

        if self.clf1.outputShape != self.clf2.outputShape:
            raise ValueError("the output shapes of both classifiers of a mixed model must be equal.")
                