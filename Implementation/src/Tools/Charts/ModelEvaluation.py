from cProfile import label
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from Tools.Charts.Charts import createBarChart, createSingleBarChart, createSingleBarChartHorizontal, createTrainingAccuracyLossChart
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class ClassificationReport(object):
    classNames: List
    metricExifOnly: List
    metricImageOnlyMobileNetV2: List
    metricImageOnlyEfficientNetB0: List
    metricImageOnlyEfficientNetB4: List
    metricImageOnlyResNet50V2: List
    metricMixedMobileNetV2: List
    metricMixedEfficientNetB0: List
    metricMixedEfficientNetB4: List
    metricMixedResNet50V2: List

    def __init__(self, 
                classNames: List,
                metricExifOnly: List,
                metricImageOnlyMobileNetV2: List,
                metricImageOnlyEfficientNetB0: List,
                metricImageOnlyEfficientNetB4: List,
                metricImageOnlyResNet50V2: List,
                metricMixedMobileNetV2: List,
                metricMixedEfficientNetB0: List,
                metricMixedEfficientNetB4: List,
                metricMixedResNet50V2: List) -> None:
        self.classNames = classNames
        self.metricExifOnly = metricExifOnly
        self.metricImageOnlyMobileNetV2 = metricImageOnlyMobileNetV2
        self.metricImageOnlyEfficientNetB0 = metricImageOnlyEfficientNetB0
        self.metricImageOnlyEfficientNetB4 = metricImageOnlyEfficientNetB4
        self.metricImageOnlyResNet50V2 = metricImageOnlyResNet50V2
        self.metricMixedMobileNetV2 = metricMixedMobileNetV2
        self.metricMixedEfficientNetB0 = metricMixedEfficientNetB0
        self.metricMixedEfficientNetB4 = metricMixedEfficientNetB4
        self.metricMixedResNet50V2 = metricMixedResNet50V2
    
    @property
    def bestTotalImageOnlyScore(self):
        return max(sum(self.metricImageOnlyEfficientNetB0),
                   sum(self.metricImageOnlyEfficientNetB4),
                   sum(self.metricImageOnlyMobileNetV2),
                   sum(self.metricImageOnlyResNet50V2))
    
    @property
    def bestTotalMixedModelScore(self):
        return max(sum(self.metricMixedEfficientNetB0),
                   sum(self.metricMixedEfficientNetB4),
                   sum(self.metricMixedMobileNetV2),
                   sum(self.metricMixedResNet50V2))

        
class EvaluationFiles(Enum):
    CLASS_NAMES = "class_names.txt"
    TRAINING_CLASSIFICATION_REPORT = "training_classification_report.csv"
    TRAINING_HISTORY = "training_history.csv"
    TRAINING_TIME = "training_time.txt"
    TEST_CLASSIFICATION_REPORT = "test_classification_report.csv"
    VALIDATION_CLASSIFICATION_REPORT = "validation_classification_report.csv"
    VALIDATION_MISSCLASSIFIED_IDS = "validation_misclassified_ids.csv"
    VALIDATION_FEATURE_IMPORTANCE = "validation_feature_importance.csv"
    MODEL_PARAMS = "model_parameters.txt"

    def read(self, path: Path) -> Any:
        if self == EvaluationFiles.CLASS_NAMES:
            return pd.read_csv(path).columns.to_numpy().tolist()
        elif any([self == EvaluationFiles.TRAINING_CLASSIFICATION_REPORT, 
                self == EvaluationFiles.TEST_CLASSIFICATION_REPORT,
                self == EvaluationFiles.VALIDATION_CLASSIFICATION_REPORT]):
            return pd.read_csv(path, index_col = 0).transpose()
        elif self == EvaluationFiles.VALIDATION_FEATURE_IMPORTANCE:
            return pd.read_csv(path, header = None, index_col = 0).transpose()
        elif self == EvaluationFiles.TRAINING_TIME:
            return float(pd.read_csv(path).columns.to_numpy()[0])
        elif self == EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS:
            return [elem[0] for elem in pd.read_csv(path, header = None).to_numpy().tolist()]
        elif self == EvaluationFiles.TRAINING_HISTORY:
            return pd.read_csv(path, index_col = 0).transpose()
        elif self == EvaluationFiles.MODEL_PARAMS:
            with open(path, "r") as modelParamsFile:
                lines = modelParamsFile.read().split("\n")
                for line in lines:
                    if "Total params:" in line:
                        return int(line.replace("Total params:", "").replace(",", "").strip())
            return None
        else:
            return None

class EvaluationTarget(Enum):
    # exif
    SUPER_EXIF_ONLY = "SUPER_EXIFONLY"
    SUB_EXIF_ONLY = "SUB_EXIFONLY"

    # images
    SUPER_IMAGEONLY_50_EFFICIENTNET_B0 = "SUPER_IMAGEONLY_FIFTY_EFFICIENTNETB0"
    SUPER_IMAGEONLY_50_EFFICIENTNET_B4 = "SUPER_IMAGEONLY_FIFTY_EFFICIENTNETB4"
    SUPER_IMAGEONLY_50_MOBILENET_V2 = "SUPER_IMAGEONLY_FIFTY_MOBILENETV2"
    SUPER_IMAGEONLY_50_RESNET_50V2 = "SUPER_IMAGEONLY_FIFTY_RESNET50V2"

    SUB_IMAGEONLY_50_EFFICIENTNET_B0 = "SUB_IMAGEONLY_FIFTY_EFFICIENTNETB0"
    SUB_IMAGEONLY_50_EFFICIENTNET_B4 = "SUB_IMAGEONLY_FIFTY_EFFICIENTNETB4"
    SUB_IMAGEONLY_50_MOBILENET_V2 = "SUB_IMAGEONLY_FIFTY_MOBILENETV2"
    SUB_IMAGEONLY_50_RESNET_50V2 = "SUB_IMAGEONLY_FIFTY_RESNET50V2"

    SUPER_IMAGEONLY_150_EFFICIENTNET_B0 = "SUPER_IMAGEONLY_HFIFTY_EFFICIENTNETB0"
    SUPER_IMAGEONLY_150_EFFICIENTNET_B4 = "SUPER_IMAGEONLY_HFIFTY_EFFICIENTNETB4"
    SUPER_IMAGEONLY_150_MOBILENET_V2 = "SUPER_IMAGEONLY_HFIFTY_MOBILENETV2"
    SUPER_IMAGEONLY_150_RESNET_50V2 = "SUPER_IMAGEONLY_HFIFTY_RESNET50V2"

    SUB_IMAGEONLY_150_EFFICIENTNET_B0 = "SUB_IMAGEONLY_HFIFTY_EFFICIENTNETB0"
    SUB_IMAGEONLY_150_EFFICIENTNET_B4 = "SUB_IMAGEONLY_HFIFTY_EFFICIENTNETB4"
    SUB_IMAGEONLY_150_MOBILENET_V2 = "SUB_IMAGEONLY_HFIFTY_MOBILENETV2"
    SUB_IMAGEONLY_150_RESNET_50V2 = "SUB_IMAGEONLY_HFIFTY_RESNET50V2"

    # mixed models
    SUPER_MIXED_50_EFFICIENTNET_B0 = "SUPER_MIXED_FIFTY_EFFICIENTNETB0"
    SUPER_MIXED_50_EFFICIENTNET_B4 = "SUPER_MIXED_FIFTY_EFFICIENTNETB4"
    SUPER_MIXED_50_MOBILENET_V2 = "SUPER_MIXED_FIFTY_MOBILENETV2"
    SUPER_MIXED_50_RESNET_50V2 = "SUPER_MIXED_FIFTY_RESNET50V2"

    SUB_MIXED_50_EFFICIENTNET_B0 = "SUB_MIXED_FIFTY_EFFICIENTNETB0"
    SUB_MIXED_50_EFFICIENTNET_B4 = "SUB_MIXED_FIFTY_EFFICIENTNETB4"
    SUB_MIXED_50_MOBILENET_V2 = "SUB_MIXED_FIFTY_MOBILENETV2"
    SUB_MIXED_50_RESNET_50V2 = "SUB_MIXED_FIFTY_RESNET50V2"

    SUPER_MIXED_150_EFFICIENTNET_B0 = "SUPER_MIXED_HFIFTY_EFFICIENTNETB0"
    SUPER_MIXED_150_EFFICIENTNET_B4 = "SUPER_MIXED_HFIFTY_EFFICIENTNETB4"
    SUPER_MIXED_150_MOBILENET_V2 = "SUPER_MIXED_HFIFTY_MOBILENETV2"
    SUPER_MIXED_150_RESNET_50V2 = "SUPER_MIXED_HFIFTY_RESNET50V2"

    SUB_MIXED_150_EFFICIENTNET_B0 = "SUB_MIXED_HFIFTY_EFFICIENTNETB0"
    SUB_MIXED_150_EFFICIENTNET_B4 = "SUB_MIXED_HFIFTY_EFFICIENTNETB4"
    SUB_MIXED_150_MOBILENET_V2 = "SUB_MIXED_HFIFTY_MOBILENETV2"
    SUB_MIXED_150_RESNET_50V2 = "SUB_MIXED_HFIFTY_RESNET50V2"

# path parts
PATH_SUB = "sub"
PATH_SUPER = "super"
PATH_EXIF_ONLY = "exifonly"
PATH_IMAGE_ONLY = "imageonly"
PATH_MIXED = "mixed"
PATH_50 = "50"
PATH_150 = "150"

class ModelEvaluation(object):
    """ The model evaluation is used to evaluate created models against each other. It provides access to the 
    evaluation files created during training and enables the generation of charts. """

    basePath: Path
    evaluationTargetFiles: Dict

    def __init__(self, directoryPath: str):
        path = Path(directoryPath).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError("There is no directory at the given base path location.")
        self.basePath = path

        self.evaluationTargetFiles = dict()
        for target in EvaluationTarget:
            self.evaluationTargetFiles[target] = self.createEvaluationFilePaths(target = target)

    def gainedImageIds(self, highResolution: bool = True):
        """ Returns image IDs of images correctly classified by mixed models but incorrectly by image only classifiers. """
        if highResolution:
            idsImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
        else:
            idsImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]

        imageIdsImageOnly = set(idsImageOnlyMobileNetV2 + idsImageOnlyEfficientNetB0 + idsImageOnlyEfficientNetB4 + idsImageOnlyResNet50V2)
        imageIdsMixed = set(idsMixedMobileNetV2 + idsMixedEfficientNetB0 + idsMixedEfficientNetB4 + idsMixedResNet50V2)
        return imageIdsImageOnly - imageIdsMixed

    def createEvaluationFilePaths(self, target: EvaluationTarget) -> Dict:
        """ Parses the evaluation files for a specific evaluation target. """
        path = None
        if target in [EvaluationTarget.SUB_EXIF_ONLY, 
                      EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0,
                      EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4,
                      EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2,
                      EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2,
                      EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0,
                      EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4,
                      EvaluationTarget.SUB_MIXED_150_MOBILENET_V2,
                      EvaluationTarget.SUB_MIXED_150_RESNET_50V2]:
            path = self.basePath.joinpath(PATH_150).joinpath(PATH_SUB)
        elif target in [EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0,
                        EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4,
                        EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2,
                        EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2,
                        EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0,
                        EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4,
                        EvaluationTarget.SUB_MIXED_50_MOBILENET_V2,
                        EvaluationTarget.SUB_MIXED_50_RESNET_50V2]:
            path = self.basePath.joinpath(PATH_50).joinpath(PATH_SUB)
        elif target in [EvaluationTarget.SUPER_EXIF_ONLY,
                        EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0,
                        EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4,
                        EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2,
                        EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2,
                        EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0,
                        EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4,
                        EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2,
                        EvaluationTarget.SUPER_MIXED_150_RESNET_50V2]:
            path = self.basePath.joinpath(PATH_150).joinpath(PATH_SUPER)
        elif target in [EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0,
                        EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4,
                        EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2,
                        EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2,
                        EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0,
                        EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4,
                        EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2,
                        EvaluationTarget.SUPER_MIXED_50_RESNET_50V2]: 
            path = self.basePath.joinpath(PATH_50).joinpath(PATH_SUPER)
        else:
            raise ValueError("couldn't select base path for evaluation target")

        
        if not path.exists() or not path.is_dir():
            raise ValueError("The evaluation directory does not contain all needed sub directories")

        neededMatches = 2
        targetNameParts = target.value.lower().split("_")
        if "super" in targetNameParts and not "exifonly" in targetNameParts:
            neededMatches = 3
        elif "sub" in targetNameParts and "exifonly" in targetNameParts:
            neededMatches = 1

        directoryList = [directory for directory in path.glob('**/*') if directory.is_dir()]
        targetDirectory = None
        for directory in directoryList:
            if targetDirectory == None:
                matches = 0
            else:
                break
            for part in targetNameParts:
                if part in directory.name.lower():
                    matches += 1
                    if matches == neededMatches:
                        targetDirectory = directory
                        break
        
        if targetDirectory == None:
            raise ValueError("couldn't find target directory for evaluation target")
        
        # create dictionary with evaluation file paths
        evaluationFilePaths = dict()
        for evaluationFile in EvaluationFiles:
            evaluationFilePath = targetDirectory.joinpath(evaluationFile.value)
            if evaluationFilePath.exists() and evaluationFilePath.is_file():
                evaluationFilePaths[evaluationFile] = evaluationFile.read(path = evaluationFilePath)
            elif evaluationFile != EvaluationFiles.VALIDATION_FEATURE_IMPORTANCE:
                raise ValueError("Missing evaluation file: " + evaluationFile.value + " for target " + target.name)
        return evaluationFilePaths
    
    def getClassificationReport(self, 
                                super: bool = False, 
                                highResolution: bool = False, 
                                metric: str = "f1-score",
                                customClasses: List = None,
                                reportType: EvaluationFiles = EvaluationFiles.VALIDATION_CLASSIFICATION_REPORT,
                                returnPlain: bool = False):
        if not customClasses == None:
            classNames = customClasses

        if super:
            if customClasses == None:
                classNames = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.CLASS_NAMES]
            valReportExifOnly = self.evaluationTargetFiles[EvaluationTarget.SUPER_EXIF_ONLY][reportType]
            if highResolution:
                valReportImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2][reportType]
                valReportImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0][reportType]
                valReportImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4][reportType]
                valReportImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2][reportType]
                valReportMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2][reportType]
                valReportMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0][reportType]
                valReportMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4][reportType]
                valReportMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_RESNET_50V2][reportType]
            else:
                valReportImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][reportType]
                valReportImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0][reportType]
                valReportImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4][reportType]
                valReportImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2][reportType]
                valReportMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2][reportType]
                valReportMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0][reportType]
                valReportMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4][reportType]
                valReportMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_RESNET_50V2][reportType]
        else:
            if customClasses == None:
                classNames = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.CLASS_NAMES]
            valReportExifOnly = self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][reportType]
            if highResolution:
                valReportImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][reportType]
                valReportImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][reportType]
                valReportImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][reportType]
                valReportImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][reportType]
                valReportMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][reportType]
                valReportMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][reportType]
                valReportMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][reportType]
                valReportMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_RESNET_50V2][reportType]
            else:
                valReportImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][reportType]
                valReportImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0][reportType]
                valReportImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4][reportType]
                valReportImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2][reportType]
                valReportMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_MOBILENET_V2][reportType]
                valReportMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0][reportType]
                valReportMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4][reportType]
                valReportMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_RESNET_50V2][reportType]
        
        return ClassificationReport(
            classNames,
            valReportExifOnly[classNames].loc[metric].to_numpy() if not returnPlain else valReportExifOnly,
            valReportImageOnlyMobileNetV2[classNames].loc[metric].to_numpy() if not returnPlain else valReportImageOnlyMobileNetV2,
            valReportImageOnlyEfficientNetB0[classNames].loc[metric].to_numpy() if not returnPlain else valReportImageOnlyEfficientNetB0,
            valReportImageOnlyEfficientNetB4[classNames].loc[metric].to_numpy() if not returnPlain else valReportImageOnlyEfficientNetB4,
            valReportImageOnlyResNet50V2[classNames].loc[metric].to_numpy() if not returnPlain else valReportImageOnlyResNet50V2,
            valReportMixedMobileNetV2[classNames].loc[metric].to_numpy() if not returnPlain else valReportMixedMobileNetV2,
            valReportMixedEfficientNetB0[classNames].loc[metric].to_numpy() if not returnPlain else valReportMixedEfficientNetB0,
            valReportMixedEfficientNetB4[classNames].loc[metric].to_numpy() if not returnPlain else valReportMixedEfficientNetB4,
            valReportMixedResNet50V2[classNames].loc[metric].to_numpy() if not returnPlain else valReportMixedResNet50V2)
    
    def createImageResolutionComparisonBarChart(self, 
                                                super: bool = False, 
                                                metric: str = "f1-score", 
                                                customClasses: List = None, 
                                                mixed: bool = False,
                                                figSize: Tuple = (14.5, 8), 
                                                barWidth: float = 0.88,
                                                labelOffset: float = 0.014,
                                                savePath = None):
        
        highRes = self.getClassificationReport(super = super, 
                                               metric = metric, 
                                               customClasses = customClasses, 
                                               highResolution = True)

        lowRes = self.getClassificationReport(super = super, 
                                              metric = metric, 
                                              customClasses = customClasses, 
                                              highResolution = False)

        if not mixed:
            data = [
                [highRes.metricImageOnlyMobileNetV2, highRes.metricImageOnlyEfficientNetB0, highRes.metricImageOnlyEfficientNetB4, highRes.metricImageOnlyResNet50V2],
                [lowRes.metricImageOnlyMobileNetV2, lowRes.metricImageOnlyEfficientNetB0, lowRes.metricImageOnlyEfficientNetB4, lowRes.metricImageOnlyResNet50V2]
            ]
        else:
            data = [
                [highRes.metricMixedMobileNetV2, highRes.metricMixedEfficientNetB0, highRes.metricMixedEfficientNetB4, highRes.metricMixedResNet50V2],
                [lowRes.metricMixedMobileNetV2, lowRes.metricMixedEfficientNetB0, lowRes.metricMixedEfficientNetB4, lowRes.metricMixedResNet50V2]
            ]

        createBarChart(
            data, 
            [   ["MobileNetV2 (150x150)", "EfficientNetB0 (150x150)", "EfficientNetB4 (150x150)", "ResNet50V2 (150x150)"], 
                ["MobileNetV2 (50x50)", "EfficientNetB0 (50x50)", "EfficientNetB4 (50x50)", "ResNet50V2 (50x50)"]
            ], 
            categoryLabels = highRes.classNames, 
            showValues = True, 
            valueFormat = "{:.2f}",
            title = "Image-Only " + metric.capitalize() + (" (Super)" if super else ""),
            yLabel = metric.capitalize(),
            figSize = figSize,
            savePath = savePath,
            barWidth = barWidth,
            labelOffset = labelOffset)
    
    def createMixedImageResolutionComparisonBarChart(self, 
                                                    super: bool = False, 
                                                    highResolution: bool = False, 
                                                    metric: str = "f1-score", 
                                                    customClasses = None, 
                                                    grid: bool = True,
                                                    figSize: Tuple = (14.5, 8), 
                                                    barWidth: float = 0.88,
                                                    labelOffset: float = 0.014,
                                                    savePath = None):

        cf = self.getClassificationReport(super = super, 
                                          metric = metric, 
                                          customClasses = customClasses, 
                                          highResolution = highResolution)
        data = [
            [cf.metricMixedMobileNetV2, cf.metricMixedEfficientNetB0, cf.metricMixedEfficientNetB4, cf.metricMixedResNet50V2],
            [cf.metricImageOnlyMobileNetV2, cf.metricImageOnlyEfficientNetB0, cf.metricImageOnlyEfficientNetB4, cf.metricImageOnlyResNet50V2]
        ]

        createBarChart(
            data, 
            [   ["MobileNetV2 Mixed", "EfficientNetB0 Mixed", "EfficientNetB4 Mixed", "ResNet50V2 Mixed"], 
                ["MobileNetV2 Image-Only", "EfficientNetB0 Image-Only", "EfficientNetB4 Image-Only", "ResNet50V2 Image-Only"]
            ], 
            categoryLabels = cf.classNames, 
            showValues = True, 
            valueFormat = "{:.2f}",
            title = "Mixed Model vs. Image Only " + metric.capitalize() + (" (Super)" if super else ""),
            yLabel = metric.capitalize(),
            figSize = figSize,
            barWidth = barWidth,
            savePath = savePath,
            grid = grid,
            labelOffset = labelOffset)
    
    def createMixedImageOnlyDelta(self, 
                                  super: bool = False, 
                                  metric: str = "f1-score", 
                                  figSize: Tuple = (7, 5), 
                                  barWidth: float = 0.80,
                                  legendPosition: str = "upper right",
                                  labelOffset: float = 0.001,
                                  savePath = None):

        cfHighRes = self.getClassificationReport(super = super, 
                                                 metric = metric, 
                                                 customClasses = ["micro avg"], 
                                                 highResolution = True)
        
        cfLowRes = self.getClassificationReport(super = super, 
                                                metric = metric, 
                                                customClasses = ["micro avg"], 
                                                highResolution = False)
        
        mobileNetV2Delta = np.concatenate([cfLowRes.metricMixedMobileNetV2 - cfLowRes.metricImageOnlyMobileNetV2] + [cfHighRes.metricMixedMobileNetV2 - cfHighRes.metricImageOnlyMobileNetV2])
        efficientNetB0Delta = np.concatenate([cfLowRes.metricMixedEfficientNetB0 - cfLowRes.metricImageOnlyEfficientNetB0] + [cfHighRes.metricMixedEfficientNetB0 - cfHighRes.metricImageOnlyEfficientNetB0])
        efficientNetB4Delta = np.concatenate([cfLowRes.metricMixedEfficientNetB4 - cfLowRes.metricImageOnlyEfficientNetB4] + [cfHighRes.metricMixedEfficientNetB4 - cfHighRes.metricImageOnlyEfficientNetB4])
        resNet50V2Delta = np.concatenate([cfLowRes.metricMixedResNet50V2 - cfLowRes.metricImageOnlyResNet50V2] + [cfHighRes.metricMixedResNet50V2 - cfHighRes.metricImageOnlyResNet50V2])

        createBarChart(
            [[mobileNetV2Delta, efficientNetB0Delta, efficientNetB4Delta, resNet50V2Delta]], 
            [["MobileNetV2", "EfficientNetB0", "EfficientNetB4", "ResNet50V2"]], 
            categoryLabels = ["50x50px", "150x150px"], 
            showValues = True, 
            valueFormat = "{:.3f}",
            labelPrefix = "+",
            title = "Mixed Model vs. Image Only " + metric.capitalize() + " Delta" + (" (Super)" if super else ""),
            yLabel = metric.capitalize() + " Delta",
            figSize = figSize,
            barWidth = barWidth,
            savePath = savePath,
            legendPosition = legendPosition,
            labelOffset = labelOffset)
    
    def createMixedImageExifComparison(self, 
                                  super: bool = False, 
                                  metric: str = "f1-score", 
                                  figSize = (5.5, 5), 
                                  barWidth: float = 0.6,
                                  labelOffset: float = 0.025,
                                  savePath = None):
        
        cfHighRes = self.getClassificationReport(super = super, 
                                                 metric = metric, 
                                                 customClasses = ["micro avg"], 
                                                 highResolution = True)
        
        cfLowRes = self.getClassificationReport(super = super, 
                                                metric = metric, 
                                                customClasses = ["micro avg"], 
                                                highResolution = False)
        
        scoresHigh = np.concatenate([cfHighRes.metricExifOnly] + 
                                    [[cfHighRes.bestTotalImageOnlyScore]] + 
                                    [[cfHighRes.bestTotalMixedModelScore]])
        scoresLow = np.concatenate([cfLowRes.metricExifOnly] + 
                                   [[cfLowRes.bestTotalImageOnlyScore]] + 
                                   [[cfLowRes.bestTotalMixedModelScore]])

        createBarChart(
            [[scoresLow, scoresHigh]], 
            [["50x50px", "150x150px"]], 
            categoryLabels = ["EXIF-Only", "Image-Only", "Mixed"], 
            showValues = True, 
            valueFormat = "{:.3f}",
            title = "EXIF vs. Best Image Only vs. Best Mixed " + metric.capitalize() + (" (Super)" if super else ""),
            yLabel = metric.capitalize(),
            figSize = figSize,
            savePath = savePath,
            barWidth = barWidth,
            grid = True,
            labelOffset = labelOffset)
    
    def createTrainingTimeMixedvsImageOnlyComparison(self, 
                                                    super: bool = False,
                                                    figSize: Tuple = (7, 5),
                                                    barWidth: float = 0.9,
                                                    savePath = None):
        if super:
            trainingTimeIO50 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])

            trainingTimeMixed50 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])
        
            trainingTimeIO150 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])

            trainingTimeMixed150 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])
        else:
            trainingTimeIO50 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                         self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])

            trainingTimeMixed50 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                            self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])
        
            trainingTimeIO150 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                          self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])

            trainingTimeMixed150 = np.array([self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.TRAINING_TIME],
                                             self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_RESNET_50V2][EvaluationFiles.TRAINING_TIME]])


        if super:
            print("super")
        sum50IO = np.sum(trainingTimeIO50)
        sum50Mixed = np.sum(trainingTimeMixed50)
        print("sum:50 IO " + str(sum50IO))
        print("sum:50 Mixed " + str(sum50Mixed))
        print("total_dif:50 " + str(sum50Mixed - sum50IO))
        print("frac:50 " + str(1 - (sum50Mixed / sum50IO)))
        
        sum150IO = np.sum(trainingTimeIO150)
        sum150Mixed = np.sum(trainingTimeMixed150)
        print("sum:150 IO " + str(sum150IO))
        print("sum:150 Mixed " + str(sum150Mixed))
        print("total_dif:150 " + str(sum150Mixed - sum150IO))
        print("frac:150 " + str(1 - (sum150Mixed / sum150IO)))
       
        print("...")

        createBarChart(
            [[trainingTimeIO50, trainingTimeMixed50, trainingTimeIO150, trainingTimeMixed150]], 
            [["Image-Only 50x50px", "Mixed 50x50px", "Image-Only 150x150px", "Mixed 150x150px"]], 
            categoryLabels = ["MobileNetV2", "EfficientNetB0", "EfficientNetB4", "ResNet50V2"], 
            showValues = False, 
            title = "Super" if super else "",
            yLabel = "Training Time in Seconds",
            figSize = figSize,
            savePath = savePath,
            barWidth = barWidth,
            grid = True)
        
    def createModelParameterCountComparision(self, figSize: Tuple = (12.5, 5), barWidth: float = 0.7, savePath = None):
        params = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.MODEL_PARAMS], 
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.MODEL_PARAMS], 
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_RESNET_50V2][EvaluationFiles.MODEL_PARAMS]])
        scale_y = 1e7
        labelFormatter = ticker.FuncFormatter(lambda x, pos: '{0:g}M'.format(x/scale_y))
        createSingleBarChart(params,
                             seriesLabels = [],
                             categoryLabels = ["EXIF MLP", 
                                               "MobileNetV2\nImage-Only", 
                                               "EfficientNetB0\nImage-Only", 
                                               "EfficientNetB4\nImage-Only", 
                                               "ResNet50V2\nImage-Only",
                                               "MobileNetV2\nMixed", 
                                               "EfficientNetB0\nMixed", 
                                               "EfficientNetB4\nMixed", 
                                               "ResNet50V2\nMixed"], 
                             figSize = figSize,
                             barWidth = barWidth,
                             labelOffset = -500000,
                             valueFormat = "{:.0f}",
                             labelPostFix = "k",
                             valueFormatFract = 1000,
                             labelPrefix = "~",
                             showValues = True,
                             yLimit = 2.5e7,
                             savePath = savePath,
                             yLabel = "Model Parameter Count",
                             yLabelFormatter = labelFormatter,
                             grid = True)
        
    def createEXIFFeatureImportanceChart(self, super: bool = False, figSize: Tuple = (8, 5), barHeight: float = 0.7, savePath = None):
        
        if super:
            featureImportanceDf = self.evaluationTargetFiles[EvaluationTarget.SUPER_EXIF_ONLY][EvaluationFiles.VALIDATION_FEATURE_IMPORTANCE]
        else:
            featureImportanceDf = self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][EvaluationFiles.VALIDATION_FEATURE_IMPORTANCE]

        featureImportanceDf = featureImportanceDf.reindex(sorted(featureImportanceDf.columns), axis=1)
        features = featureImportanceDf.columns.to_numpy()
        featureImportances = featureImportanceDf.iloc[0].to_numpy()

        createSingleBarChartHorizontal(featureImportances,
                                       seriesLabels = [],
                                       categoryLabels = features, 
                                       figSize = figSize,
                                       barHeight = barHeight,
                                       showValues = True,
                                       sc = True,
                                       valueFormat = "{:.3f}",
                                       title = "Average F1-Score Drop on 50 Feature Permutations" + (" (Super)" if super else ""),
                                       labelPrefix = "-",
                                       xLimit = 0.15,
                                       savePath = savePath,
                                       grid = False)
    
    def createTrainingComparisonPlot(self, 
                                     super: bool = False, 
                                     highResolution: bool = False, 
                                     loss: bool = False,
                                     savePath = None):

        rp = self.getClassificationReport(super = super,  
                                          customClasses = [], 
                                          highResolution = highResolution,
                                          reportType = EvaluationFiles.TRAINING_HISTORY,
                                          returnPlain = True)
        
        # identify accuracy metric key
        if loss:
            metricIndexKey = "loss"
        else:
            for value in rp.metricExifOnly.index.to_numpy():
                if "accuracy" in value and not "val" in value:
                    metricIndexKey = value
                    break
        
        pixels = " (150x150px)" if highResolution else " (50x50px)"
        s = " (Super)" if super else ""
        metric = "Accuracy" if not metricIndexKey == "loss" else "Loss"
        
        createTrainingAccuracyLossChart(dataFrames = [rp.metricExifOnly],
                                        dataIndexKey = metricIndexKey, 
                                        valDataIndexKey = "val_" + metricIndexKey,
                                        labels = ["EXIF MLP"],
                                        title = "Training " + metric + " at Epoch - EXIF-Only" + s,
                                        figSize = (5.5, 3.0),
                                        targetEpochPatience = 50,
                                        savePath = savePath)

        createTrainingAccuracyLossChart(dataFrames = [rp.metricImageOnlyMobileNetV2, 
                                                      rp.metricImageOnlyEfficientNetB0,
                                                      rp.metricImageOnlyEfficientNetB4,
                                                      rp.metricImageOnlyResNet50V2],
                                        dataIndexKey = metricIndexKey, 
                                        valDataIndexKey = "val_" + metricIndexKey,
                                        labels = ["MobileNetV2",
                                                  "EfficientNetB0",
                                                  "EfficientNetB4",
                                                  "ResNet50V2"],
                                        title = "Training " + metric + " at Epoch - Image-Only" + s + pixels,
                                        targetEpochPatience = 5,
                                        startFineTuningEpoch = 15,
                                        savePath = savePath)
        
        createTrainingAccuracyLossChart(dataFrames = [rp.metricMixedMobileNetV2, 
                                                      rp.metricMixedEfficientNetB0,
                                                      rp.metricMixedEfficientNetB4,
                                                      rp.metricMixedResNet50V2],
                                        dataIndexKey = metricIndexKey, 
                                        valDataIndexKey = "val_" + metricIndexKey,
                                        labels = ["MobileNetV2",
                                                  "EfficientNetB0",
                                                  "EfficientNetB4",
                                                  "ResNet50V2"],
                                        title = "Training " + metric + " at Epoch - Mixed" + s + pixels,
                                        targetEpochPatience = 5,
                                        startFineTuningEpoch = 15,
                                        savePath = savePath)
        

if __name__ == '__main__':
    modelEvaluation = ModelEvaluation(directoryPath = "/Users/ralflederer/Desktop/IndoorOutdoorMi")

    """ modelEvaluation.createImageResolutionComparisonBarChart(super = False, 
                                                            metric = "f1-score", 
                                                            customClasses = ["micro avg"], 
                                                            mixed = True) """
    
    # plot for mixed model vs image model performance - super
    """ modelEvaluation.createMixedImageResolutionComparisonBarChart(super = True, 
                                                                 metric = "f1-score", 
                                                                 highResolution = False,
                                                                 barWidth = 0.7,
                                                                 figSize=(6, 5),
                                                                 labelOffset = 0.025,
                                                                 savePath = None)
    
    # plot for mixed model vs image model performance
    modelEvaluation.createMixedImageResolutionComparisonBarChart(super = False, 
                                                                 metric = "f1-score", 
                                                                 highResolution = False,
                                                                 barWidth = 0.8,
                                                                 figSize=(14, 7),
                                                                 labelOffset = 0.016,
                                                                 savePath = None)

    # plot for delta (mixed image only)
    modelEvaluation.createMixedImageOnlyDelta(super = True, savePath = None) 
    modelEvaluation.createMixedImageOnlyDelta(super = False, savePath = None)

    # plot for exif vs. best image only vs  best mixed model
    modelEvaluation.createMixedImageExifComparison(super = True, savePath = None)
    modelEvaluation.createMixedImageExifComparison(super = False, savePath = None)

    # plot for training time mixed model vs image only
    modelEvaluation.createTrainingTimeMixedvsImageOnlyComparison(super = True, savePath = None)
    modelEvaluation.createTrainingTimeMixedvsImageOnlyComparison(super = False, savePath = None)

    # plot for model parameter count comparison
    modelEvaluation.createModelParameterCountComparision(savePath = None)

    # plot for exif tag feature importance (exif-only)
    modelEvaluation.createEXIFFeatureImportanceChart(super = True, savePath = None)
    modelEvaluation.createEXIFFeatureImportanceChart(super = False, savePath = None)

    # plots for training accuracy (exif only, image only, mixed) 
    modelEvaluation.createTrainingComparisonPlot(super = True, highResolution = False)
    modelEvaluation.createTrainingComparisonPlot(super = True, highResolution = True)
    modelEvaluation.createTrainingComparisonPlot(super = False, highResolution = False)
    modelEvaluation.createTrainingComparisonPlot(super = False, highResolution = True)

     # plots for training loss (exif only, image only, mixed) 
    modelEvaluation.createTrainingComparisonPlot(super = True, highResolution = False, loss = True)
    modelEvaluation.createTrainingComparisonPlot(super = True, highResolution = True, loss = True)
    modelEvaluation.createTrainingComparisonPlot(super = False, highResolution = False, loss = True)
    modelEvaluation.createTrainingComparisonPlot(super = False, highResolution = True, loss = True) """


    modelEvaluation.createModelParameterCountComparision(savePath = None)

    plt.show()
    #savePath = "/Users/ralflederer/Desktop/test.png"
    