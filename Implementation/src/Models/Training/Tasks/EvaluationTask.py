from keras.models import Model, load_model
from Models.Training.Generators.BatchGeneratorProvider import BatchGeneratorProvider
import pandas as pd
from sklearn.metrics import classification_report

class EvaluationTask(object):
    """ An evaluation task evaluates the data of a data generator for a specific keras model.  """

    model: Model
    dataProvider: BatchGeneratorProvider

    def __init__(self, modelPath: str, provider: BatchGeneratorProvider):
        self.dataProvider = provider
        self.model = load_model(modelPath)

    def run(self) -> pd.DataFrame:
        # create data generator for evaluation
        dataGenerator, _, _ = self.dataProvider.createGenerators()

        # predict labels
        predictionResult = dataGenerator.predictOn(model = self.model, multiLabel = self.dataProvider.multiLabel)

        # create classification report
        classificationReport = classification_report(predictionResult.trueY, predictionResult.predictionYRounded, digits = 3, output_dict = True, target_names = self.dataProvider.classes) 
        reportDataFrame = pd.DataFrame(data = classificationReport).transpose().round(3)

        return reportDataFrame