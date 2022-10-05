from typing import Callable, List, Tuple
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as pMobileNet
from keras.applications.resnet_v2 import ResNet152V2, ResNet101V2, ResNet50V2, preprocess_input as pResNet
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, BatchNormalization
from keras.layers.preprocessing.image_preprocessing import RandomFlip, RandomRotation, RandomTranslation
from keras.models import Model, Sequential
from Models.Classifiers.Classifier import FineTuneClassifier, top_2_accuracy

def validModelNames() -> List[str]:
    """ Returns a list of valid model names, the cnn model builder can build models for. """
    return ["EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7", "MobileNetV2", "ResNet152V2", "ResNet101V2", "ResNet50V2"]

def standardModelNames() -> List[str]:
    """ Returns a list of standard model names, the cnn model builder can build models for. """
    return ["EfficientNetB0", "EfficientNetB4", "MobileNetV2", "ResNet50V2"]

def modelNameToModelFunction(modelName: str) -> Callable[[], Model]:
    """ Returns the model build function for the given model name. possible values are given by validModelNames(). """
    if modelName == "EfficientNetB0":
        return EfficientNetB0
    elif modelName == "EfficientNetB1":
        return EfficientNetB1
    elif modelName == "EfficientNetB2":
        return EfficientNetB2
    elif modelName == "EfficientNetB3":
        return EfficientNetB3
    elif modelName == "EfficientNetB4":
        return EfficientNetB4
    elif modelName == "EfficientNetB5":
        return EfficientNetB5
    elif modelName == "EfficientNetB6":
        return EfficientNetB6
    elif modelName == "EfficientNetB7":
        return EfficientNetB7
    elif modelName == "MobileNetV2":
        return MobileNetV2
    elif modelName == "ResNet152V2":
        return ResNet152V2
    elif modelName == "ResNet101V2":
        return ResNet101V2
    elif modelName == "ResNet50V2":
        return ResNet50V2

def createTransferLearningModel(modelFunction: Callable[[], Model], 
                               imageShape: Tuple, 
                               outputSize: int, 
                               multiLabel: bool = False,
                               dropoutSize: float = 0.2,
                               batchNormalization: bool = True) -> Model:
    """ Creates a transfer learning model with a base model which will be created by using the given model 
    creation function. The base model will be extended with a new classification head. 
    The name of the base model will be preceeded by 'tf_base'. """
    dataAugmentationLayer = Sequential([
        RandomRotation(factor = 0.15),
        RandomTranslation(height_factor = 0.1, width_factor = 0.1),
        RandomFlip()
    ])

    # create input layer
    inputLayer = Input(shape = imageShape, dtype = "float32")

    # create base model for transfer learning
    baseModel = modelFunction(input_shape = imageShape, include_top = False, weights = "imagenet")
    baseModel.trainable = False

    # apply a special name so that we can identify the base model later
    baseModel._name = "tf_base_" + baseModel._name

    # choose correct pre-processing function (for efficientnet it is included in the model)
    preprocessInputFunction = None
    if "mobilenet" in modelFunction.__name__.lower():
        preprocessInputFunction = pMobileNet
    elif "resnet" in modelFunction.__name__.lower():
        preprocessInputFunction = pResNet

    # create output layer and new classification head
    outputLayer = dataAugmentationLayer(inputLayer)   
    if preprocessInputFunction != None:
        outputLayer = preprocessInputFunction(outputLayer) 
    outputLayer = baseModel(outputLayer, training = False)
    outputLayer = GlobalAveragePooling2D()(outputLayer)
    if batchNormalization == True:
        outputLayer = BatchNormalization()(outputLayer)
    outputLayer = Dropout(dropoutSize)(outputLayer)
    outputLayer = Dense(outputSize, activation = "sigmoid" if multiLabel else "softmax")(outputLayer)

    model = Model(inputLayer, outputLayer)
    model.compile(optimizer = "adam", 
                  loss = "binary_crossentropy" if multiLabel else "categorical_crossentropy", 
                  metrics = [top_2_accuracy if multiLabel else "categorical_accuracy"])
    return model

class CNNClassifier(FineTuneClassifier):
    """ A cnn classifier builds a keras model, using the given model creation function as baseline cnn architecture.
    The weights of the baseline model will be initialized with 'imagenet' weights to enable transfer-learning.
    Additionally, the classifier generates a classification head on top of the base model to support the desired 
    classification task. On the bottom of the model, an image augmentation layer will be added to increase the robustness
    of the overall model. During training, fine-tuning will be applied (see FineTuneClassifier). Valid model functions 
    can be obtained by calling validModelNames() and modelNameToModelFunction() respectively. """

    def __init__(self, name: str, 
                       modelFunction: Callable[[], Model],
                       fineTuneLayersCount: int = 0, 
                       fineTuneEpochs: int = 0,
                       inputShape: Tuple = None, 
                       outputShape: Tuple = None,
                       multiLabel: bool = False):

        super().__init__(name = name,
                        modelCreationFunction = lambda imageShape, outputSize, multiLabel: 
                            createTransferLearningModel(modelFunction, imageShape, outputSize, multiLabel, batchNormalization = "efficientnet" in modelFunction.__name__.lower()),
                            fineTuneLayersCount = fineTuneLayersCount,
                            fineTuneEpochs = fineTuneEpochs,
                            inputShape = inputShape,
                            outputShape = outputShape,
                            multiLabel = multiLabel)
