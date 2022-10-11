from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from Tools.Charts.Charts import createBarChart
import numpy as np
import matplotlib.pyplot as plt

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
                                reportType: EvaluationFiles = EvaluationFiles.VALIDATION_CLASSIFICATION_REPORT):
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
            valReportExifOnly[classNames].loc[metric].to_numpy(),
            valReportImageOnlyMobileNetV2[classNames].loc[metric].to_numpy(),
            valReportImageOnlyEfficientNetB0[classNames].loc[metric].to_numpy(),
            valReportImageOnlyEfficientNetB4[classNames].loc[metric].to_numpy(),
            valReportImageOnlyResNet50V2[classNames].loc[metric].to_numpy(),
            valReportMixedMobileNetV2[classNames].loc[metric].to_numpy(),
            valReportMixedEfficientNetB0[classNames].loc[metric].to_numpy(),
            valReportMixedEfficientNetB4[classNames].loc[metric].to_numpy(),
            valReportMixedResNet50V2[classNames].loc[metric].to_numpy())
    
    def createImageResolutionComparisonBarChart(self, 
                                                super: bool = False, 
                                                metric: str = "f1-score", 
                                                customClasses: List = None, 
                                                mixed: bool = False,
                                                figSize: Tuple = (14.5, 8), 
                                                barWidth: float = 0.88,
                                                labelOffset: float = 0.014):
        
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
            barWidth = barWidth,
            labelOffset = labelOffset)
    
    def createMixedImageResolutionComparisonBarChart(self, 
                                                    super: bool = False, 
                                                    highResolution: bool = False, 
                                                    metric: str = "f1-score", 
                                                    customClasses = None, 
                                                    grid: bool = False,
                                                    figSize: Tuple = (14.5, 8), 
                                                    barWidth: float = 0.88,
                                                    labelOffset: float = 0.014):

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
            grid = grid,
            labelOffset = labelOffset)
    
    def createMixedImageOnlyDelta(self, 
                                  super: bool = False, 
                                  metric: str = "f1-score", 
                                  figSize: Tuple = (14.5, 8), 
                                  barWidth: float = 0.88,
                                  legendPosition: str = "upper right",
                                  labelOffset: float = 0.014):

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
            legendPosition = legendPosition,
            labelOffset = labelOffset)
    
    def createMixedImageExifComparison(self, 
                                  super: bool = False, 
                                  metric: str = "f1-score", 
                                  figSize: Tuple = (14.5, 8), 
                                  barWidth: float = 0.88,
                                  labelOffset: float = 0.014):
        
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
            barWidth = barWidth,
            grid = True,
            labelOffset = labelOffset)


if __name__ == '__main__':
    modelEvaluation = ModelEvaluation(directoryPath = "/Users/ralflederer/Desktop/IndoorOutdoorSc")

    """ modelEvaluation.createImageResolutionComparisonBarChart(super = False, 
                                                            metric = "f1-score", 
                                                            customClasses = ["micro avg"], 
                                                            mixed = True) """
    
     # plot for mixed model vs image model performance - super
    modelEvaluation.createMixedImageResolutionComparisonBarChart(super = True, 
                                                                 metric = "f1-score", 
                                                                 highResolution = False,
                                                                 barWidth = 0.7,
                                                                 figSize=(6, 5),
                                                                 grid = True,
                                                                 labelOffset = 0.025)
    
    # plot for mixed model vs image model performance
    modelEvaluation.createMixedImageResolutionComparisonBarChart(super = False, 
                                                                 metric = "f1-score", 
                                                                 highResolution = False,
                                                                 barWidth = 0.8,
                                                                 figSize=(14, 7),
                                                                 grid = True,
                                                                 labelOffset = 0.016)

    # plot for delta (mixed image only) - super
    modelEvaluation.createMixedImageOnlyDelta(super = True, 
                                              metric = "f1-score", 
                                              barWidth = 0.8,
                                              figSize=(7, 5),
                                              labelOffset = 0.001) 

    # plot for delta (mixed, image only)
    modelEvaluation.createMixedImageOnlyDelta(super = False, 
                                              metric = "f1-score", 
                                              barWidth = 0.8,
                                              figSize=(7, 5),
                                              labelOffset = 0.001)

    # plot for exif vs. best image only vs  best mixed model - super
    modelEvaluation.createMixedImageExifComparison(super = True, 
                                                  metric = "f1-score", 
                                                  barWidth = 0.6,
                                                  labelOffset = 0.025,
                                                  figSize = (5.5, 5))

    # plot for exif vs. best image only vs  best mixed model
    modelEvaluation.createMixedImageExifComparison(super = False, 
                                                  metric = "f1-score", 
                                                  barWidth = 0.6,
                                                  labelOffset = 0.025,
                                                  figSize = (5.5, 5))
    plt.show()
          
    #savePath = "/Users/ralflederer/Desktop/test.png"
