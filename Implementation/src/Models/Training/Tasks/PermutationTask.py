from typing import Any, Callable, List
from Models.Training.Generators.BatchGeneratorProvider import BatchGeneratorProvider, ExifImageProvider
from Models.Training.Generators.BatchGenerators import BatchGenerator
from Transformers.Exif.ExifTagTransformer import ExifTagTransformer
from Models.Classifiers.Classifier import Classifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path
import csv

def permutationScoreF1(yTrue, yPred) -> float:
    return f1_score(yTrue, yPred, average = "micro")

def permutationScoreAccuracy(yTrue, yPred) -> float:
    return accuracy_score(yTrue, yPred)

class PermutationEvaluationTask(object):
    """ A permutation evaluation task evaluates the feature importance of the input features of the given (trained) classifier, using
    the given data generator. Each input feature will n times be randomly permutated and the resulting model metric will be 
    compared to the un-permutated data metric. The most important feature will most reduce the model metric compared to the un-permutated case.
    Note: the task can only evaluate tabular-data features. The list combinedFeatures may contain categorical / binary one-hot encoded features that
    should be considered together, instead of permutating each feature individually. """

    dataGenerator: BatchGenerator
    permutations: int
    combinedFeatures: List[str]
    storagePath: Path
    metric: Callable[[Any, Any], float]

    def __init__(self, permutations: int = 10, storagePath: str = None, combinedFeatures: List[str] = [], metric: Callable[[Any, Any], float] = permutationScoreF1):
        self.permutations = permutations
        self.storagePath = Path(storagePath) if not storagePath is None else None
        self.combinedFeatures = combinedFeatures
        self.metric = metric
        
    def run(self, classifier: Classifier, dataGeneratorProvider: BatchGeneratorProvider, normalize: bool = False, verbose: int = 1) -> dict:
        """ Runs the permutation task and permutates each feature n times, evaluating its importance to the classification task.
        The method will return a dictionary with the features as keys and the score metric differences as values. """
        if verbose > 0:
                print("starting permutation task for " + classifier.name + " ...")

        # we use the validation data for permutation
        dataGenerator = dataGeneratorProvider.cachedGenerators[2]
        allColumns = dataGenerator.xDataFrame.columns
        allColumnFeatures = {}

        # get multi-column features
        for feature in self.combinedFeatures:
            featureColumns = [column for column in allColumns if feature in column]
            allColumnFeatures[feature] = featureColumns
            allColumns = [column for column in allColumns if column not in featureColumns]
        
        # get single-column features
        for feature in allColumns:
            allColumnFeatures[feature] = [feature]

        # calculate base score
        predictionResult = dataGenerator.predictOn(model = classifier.model, multiLabel = dataGeneratorProvider.multiLabel, batchSize = dataGeneratorProvider.batchSize)
        baseScore = self.metric(predictionResult.trueY, predictionResult.predictionYRounded)

        permutationScores = {}
        # permutate all features n times, and evaluate metrics
        for feature in allColumnFeatures.keys():
            columnsToPermutate = allColumnFeatures[feature]
        
            featureScores = []
            for i in range(0, self.permutations):
                # predict the data generators data and get true labels
                permutationGenerator = dataGenerator.permutateX(columnNames = columnsToPermutate)
                predictionResult = permutationGenerator.predictOn(model = classifier.model, multiLabel = dataGeneratorProvider.multiLabel)

                score = self.metric(predictionResult.trueY, predictionResult.predictionYRounded)
                featureScores.append(score)
                
                if verbose > 0:
                    print("feature: {feature} - permutation: {count}/{total} - score: {score:.3f} - difference: {difference:.3f}"
                        .format(count = i + 1, total = self.permutations, feature = feature, score = score, difference = baseScore - score))
            
            permutationScores[feature] = baseScore - np.mean(featureScores)
        
        # normalize to sum up to 1.0
        if normalize:
            metricSum = sum(permutationScores.values())
            for feature in permutationScores.keys():
                permutationScores[feature] = permutationScores[feature] / metricSum
        
        # save permutation scores
        if not self.storagePath is None:
            with open(self.storagePath.joinpath(classifier.name).joinpath("validation_feature_importance.csv"), "w") as file:  
                writer = csv.writer(file)
                for key, value in permutationScores.items():
                    writer.writerow([key, "{:.3f}".format(value)])
        
        if verbose > 0:
                print("permutation task finished ...")

        return permutationScores

class ExifPermutationTask(PermutationEvaluationTask):
    """ A permutation task for exif data features. The task will automatically discover categorical / binary exif features. """
    def __init__(self, permutations: int = 10, storagePath: str = None, metric: Callable[[Any, Any], float] = permutationScoreF1):
        super().__init__(permutations = permutations, storagePath = storagePath, combinedFeatures = [], metric = metric)

    def run(self, classifier: Classifier, dataGeneratorProvider: ExifImageProvider, normalize: bool = False, verbose: int = 1) -> dict:
        # create typed tag lists for the exif tags, since categorical and binary features are spread over mulitple columns
        _, categorical, binary = ExifTagTransformer.typedTagLists(exifTags = dataGeneratorProvider.exifTags, transformer = dataGeneratorProvider.exifTagTransformer)
        self.combinedFeatures = categorical + binary
        super().run(classifier = classifier, dataGeneratorProvider = dataGeneratorProvider, normalize = normalize, verbose = verbose)
