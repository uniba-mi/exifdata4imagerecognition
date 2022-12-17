from typing import Callable, Tuple
from Models.Training.Generators.BatchGenerators import BatchGenerator
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, History
from keras.layers import BatchNormalization

def fineTuningLearningRateScheduler(startEpoch, epoch, lr):
    # for transfer learning we use a big learning rate for the initial epochs and then lower the learning-rate
    if epoch < startEpoch + 5:
        return 5e-5
    elif epoch >= startEpoch + 5 and epoch < startEpoch + 10:
        return 1e-5
    else:
        return lr * (1.0 / 1.01)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 3)

class Classifier(object):
    """ A classifier represents a specific kind of keras model optimized for a specific classification task. 
    It creates & trains the keras model returned by calling the given model function. 
    The given model function must return a keras model while accepting the input and output shape of the model
    to be created as function parameters. Additionally the model function must accept an input flag whether the
    classification task is single-label or multi-label. """

    inputShape: Tuple
    outputShape: Tuple
    multiLabel: bool
    name: str
    modelCreationFunction: Callable[[Tuple, Tuple, bool], Model]
    model: Model
    lastEarlyStoppingPatience: int

    def __init__(self, name: str, 
                       modelCreationFunction: Callable[[Tuple, Tuple, bool], Model],
                       inputShape: Tuple = None, 
                       outputShape: Tuple = None,
                       multiLabel: bool = False):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.multiLabel = multiLabel
        self.name = name
        self.modelCreationFunction = modelCreationFunction
        self.model = None
        self.lastEarlyStoppingPatience = 0

    def train(self, trainDataGenerator: BatchGenerator, 
                    validationDataGenerator: BatchGenerator, 
                    epochs: int,
                    earlyStoppingPatience: int = 0,
                    optimize: str = "accuracy",
                    learningRateScheduler: Callable = None,
                    startEpoch: int = 0,
                    logDir: str = None,
                    verbose: int = 1) -> History:
        """ Trains the current model with the given train / validation data generators. If no model is created yet, 
        it will be created using the model creation function of the classifier. The training process will use early stopping
        for validation accuracy using the given patience (min_delta = 0.0001). """
        
        if self.model == None:
            self.model = self.modelCreationFunction(self.inputShape, self.outputShape, self.multiLabel)
            if not isinstance(self.model, Model):
                raise ValueError("the creation function of a model builder must create a keras Model instance.")
            elif verbose > 0:
                print(self.model.summary())

        callbacks = []

        # add early stopping for validation accuracy
        if earlyStoppingPatience > 0:
            earlyStoppingMetric = None
            if optimize == "loss":
                earlyStoppingMetric = "val_loss"
                if verbose > 0:
                    print("optimizing model for loss function")
            else:
                for metric in self.model.compiled_metrics._metrics:
                    if isinstance(metric, str) and "accuracy" in metric:
                        earlyStoppingMetric = "val_" + metric
                    elif isinstance(metric, Callable) and not metric.__name__ == None and "accuracy" in metric.__name__:
                        earlyStoppingMetric = "val_" + metric.__name__
                if verbose > 0:
                    print("optimizing model for accuracy metrics")
            
            if not earlyStoppingMetric == None:
                self.lastEarlyStoppingPatience = earlyStoppingPatience
                callbacks.append(EarlyStopping(monitor = earlyStoppingMetric, 
                                              patience = earlyStoppingPatience, 
                                               min_delta = 0.0001,
                                              restore_best_weights = True,
                                              verbose = verbose))

        if not learningRateScheduler == None:
            callbacks.append(LearningRateScheduler(learningRateScheduler, verbose = verbose))

        if not logDir == None:
            callbacks.append(TensorBoard(log_dir = logDir))

        history = self.model.fit(trainDataGenerator,
                                 validation_data = validationDataGenerator,
                                 epochs = epochs,
                                 initial_epoch = startEpoch,
                                 callbacks = callbacks,
                                 verbose = verbose)
        return history

class FineTuneClassifier(Classifier):
    """ A classifier which fine-tunes the model during training. The parameters
    fineTuneLayersCount and fineTuneEpochs specify the number of layers and epochs to use for fine-tuning. 
    The initial training will be performed with (epochs - fineTuneEpochs) epochs. The given model creation function
    must ensure to return a model with frozen layers, which will be unfrozen by the classifier in the fine-tune phase.
    If the model created by the model creation function encapsulates base models which are to be used for fine-tuning, 
    the base models names must start with the string 'tf_base'.
    Note: Fine-tuning optimizes the weights of a certain amount of model layers for a specific classification task.
    This is usually used when performing transfer-learning to optimize a pre-trained model for another classification task.
    The fine-tune classifier will use an optimized learning rate scheduler for the fine-tuning learning process."""

    fineTuneLayersCount: int
    fineTuneEpochs: int

    def __init__(self, name: str, 
                       modelCreationFunction: Callable[[Tuple, Tuple], Model],
                       fineTuneLayersCount: int = 0, 
                       fineTuneEpochs: int = 0,
                       inputShape: Tuple = None, 
                       outputShape: Tuple = None,
                       multiLabel: bool = False):

        super().__init__(name = name,
                         modelCreationFunction = modelCreationFunction,
                         inputShape = inputShape,
                         outputShape = outputShape,
                         multiLabel = multiLabel)

        self.fineTuneLayersCount = fineTuneLayersCount
        self.fineTuneEpochs = fineTuneEpochs

    def train(self, trainDataGenerator: BatchGenerator, 
                    validationDataGenerator: BatchGenerator, 
                    epochs: int,
                    earlyStoppingPatience: int = 10,
                    optimize: str = "accuracy",
                    learningRateScheduler: Callable = None,
                    startEpoch: int = 0,
                    logDir: str = None,
                    verbose: int = 1) -> History:
        
        if epochs - self.fineTuneEpochs <= 0:
            raise ValueError("the number of epochs minus the number of fine tune epochs must be greater than zero.")

        # first we train the initial epochs (with most model params locked), before we do the fine-tuning of the model
        history = super().train(trainDataGenerator = trainDataGenerator, 
                                validationDataGenerator = validationDataGenerator,
                                epochs = epochs - self.fineTuneEpochs,
                                optimize = optimize,
                                startEpoch = startEpoch,
                                logDir = logDir,
                                verbose = verbose)
    
        # start the fine-tuning of the model
        if self.fineTuneEpochs > 0:
            if verbose > 0:
                print("start fine-tuning the model ...")
            
            # get base models to fine tune or use self.model if no base models are present
            baseModels = [layer for layer in self.model.layers if "tf_base" in layer._name]
            baseModels = baseModels if len(baseModels) > 0 else self.model

            # prepeare all base models for fine tuning
            for baseModel in baseModels:
                if len(baseModel.layers) < self.fineTuneLayersCount:
                    raise ValueError("the number of fine tune layers: " 
                    + str(self.fineTuneLayersCount) + " is larger than the number of layers in the model: " 
                        + str(len(baseModel.layers)))
                
                # set the base model trainable to true
                baseModel.trainable = True
                fineTuneStartLayer = len(baseModel.layers) - self.fineTuneLayersCount

                # freeze all layers before our start layer
                for layer in baseModel.layers[:fineTuneStartLayer]:
                    layer.trainable = False

                # freeze all batch normalization layers behind our start layer
                for layer in baseModel.layers[fineTuneStartLayer:]:
                        if isinstance(layer, BatchNormalization):
                            layer.trainable = False

            # determine custom metrics applied to the model and re-compile them
            functionMetrics = []
            for metric in self.model.compiled_metrics._user_metrics:
                if isinstance(metric, Callable):
                    functionMetrics.append(metric)

            # re-compile the model and use the same optimizer / loss / metrics than before
            # note: loss must be removed since it is automatically inserted by keras
            self.model.compile(optimizer = self.model.optimizer.__class__(), 
                               loss = self.model.loss, 
                               metrics = [metric for metric in self.model.metrics_names if metric != "loss" and 
                                    metric not in [metric.__name__ for metric in functionMetrics]] + functionMetrics)

            if verbose > 0:
                print(self.model.summary())

            startEpoch = history.epoch[-1]
            if learningRateScheduler is None:
                learningRateScheduler = lambda epoch, lr: fineTuningLearningRateScheduler(startEpoch, epoch, lr)

            # for fine-tune training we use a learning-rate scheduler
            historyFineTuning =  super().train(trainDataGenerator = trainDataGenerator, 
                                               validationDataGenerator = validationDataGenerator,
                                               epochs = self.fineTuneEpochs,
                                               earlyStoppingPatience = earlyStoppingPatience,
                                               optimize = optimize,
                                               learningRateScheduler = learningRateScheduler,
                                               startEpoch = history.epoch[-1],
                                               logDir = logDir,
                                               verbose = verbose)
            
            # merge the histories of both train runs
            for key in history.history.keys():
                history.history[key] += historyFineTuning.history[key]
            history.params["epochs"] += historyFineTuning.params["epochs"]
            
        return history
