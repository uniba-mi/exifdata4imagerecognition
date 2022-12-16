from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from Tools.Charts.Charts import createBarChart, createSingleBarChart, createSingleBarChartHorizontal, createTrainingAccuracyLossChart, createImageOverviewChart
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from DomainModel.ExifData import ExifTag  
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

class FileList(object):
    fileExifOnly: List
    fileImageOnlyMobileNetV2: List
    fileImageOnlyEfficientNetB0: List
    fileImageOnlyEfficientNetB4: List
    fileImageOnlyResNet50V2: List
    fileMixedMobileNetV2: List
    fileMixedEfficientNetB0: List
    fileMixedEfficientNetB4: List
    fileMixedResNet50V2: List

    def __init__(self, 
                fileExifOnly: List,
                fileImageOnlyMobileNetV2: List,
                fileImageOnlyEfficientNetB0: List,
                fileImageOnlyEfficientNetB4: List,
                fileImageOnlyResNet50V2: List,
                fileMixedMobileNetV2: List,
                fileMixedEfficientNetB0: List,
                fileMixedEfficientNetB4: List,
                fileMixedResNet50V2: List) -> None:
        self.fileExifOnly = fileExifOnly
        self.fileImageOnlyMobileNetV2 = fileImageOnlyMobileNetV2
        self.fileImageOnlyEfficientNetB0 = fileImageOnlyEfficientNetB0
        self.fileImageOnlyEfficientNetB4 = fileImageOnlyEfficientNetB4
        self.fileImageOnlyResNet50V2 = fileImageOnlyResNet50V2
        self.fileMixedMobileNetV2 = fileMixedMobileNetV2
        self.fileMixedEfficientNetB0 = fileMixedEfficientNetB0
        self.fileMixedEfficientNetB4 = fileMixedEfficientNetB4
        self.fileMixedResNet50V2 = fileMixedResNet50V2

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
    TRAINING_HISTORY = "training_history.csv"
    TRAINING_TIME = "training_time.txt"
    VALIDATION_MISSCLASSIFIED_IDS = "validation_misclassified_ids.csv"
    VALIDATION_FEATURE_IMPORTANCE = "validation_feature_importance.csv"
    MODEL_PARAMS = "model_parameters.txt"
    VALIDATION_LABELS = "validation_labels.csv"
    TEST_LABELS = "test_labels.csv"
    TRAINING_LABELS = "training_labels.csv"

    def read(self, path: Path) -> Any:
        if self == EvaluationFiles.CLASS_NAMES:
            return pd.read_csv(path).columns.to_numpy().tolist()
        elif self == EvaluationFiles.VALIDATION_FEATURE_IMPORTANCE:
            return pd.read_csv(path, header = None, index_col = 0).transpose()
        elif self == EvaluationFiles.TRAINING_TIME:
            return float(pd.read_csv(path).columns.to_numpy()[0])
        elif self == EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS:
            return [elem[0] for elem in pd.read_csv(path, header = None).to_numpy().tolist()]
        elif self == EvaluationFiles.TRAINING_HISTORY:
            return pd.read_csv(path, index_col = 0).transpose()
        elif any([self == EvaluationFiles.VALIDATION_LABELS,
                  self == EvaluationFiles.TEST_LABELS,
                  self == EvaluationFiles.TRAINING_LABELS]):
            return pd.read_csv(path)
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
    datasetPath: Path
    evaluationTargetFiles: Dict

    def __init__(self, directoryPath: str, datasetDirectoryPath: str = None):
        path = Path(directoryPath).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError("There is no directory at the given base path location.")
        self.basePath = path

        if datasetDirectoryPath != None:
            self.datasetPath = Path(datasetDirectoryPath)
            if not self.datasetPath.exists() or not self.datasetPath.is_dir():
                raise ValueError("There is no dataset directory at the given dataset directory location.")

        self.evaluationTargetFiles = dict()
        for target in EvaluationTarget:
            self.evaluationTargetFiles[target] = self.createEvaluationFilePaths(target = target)
    
    @property
    def name(self) -> str:
        return self.basePath.stem

    def imageIds(self, super: bool = True, highResolution: bool = True, count: bool = False):
        """ Returns image IDs of images correctly classified by mixed models but incorrectly by image only classifiers. """
        if highResolution:
            idsImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2 if super else EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0 if super else EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4 if super else EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2 if super else EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2 if super else EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0 if super else EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4 if super else EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_RESNET_50V2 if super else EvaluationTarget.SUB_MIXED_150_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
        else:
            idsImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2 if super else EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0 if super else EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4 if super else EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2 if super else EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2 if super else EvaluationTarget.SUB_MIXED_50_MOBILENET_V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0 if super else EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4 if super else EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]
            idsMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_RESNET_50V2 if super else EvaluationTarget.SUB_MIXED_50_RESNET_50V2][EvaluationFiles.VALIDATION_MISSCLASSIFIED_IDS]

        if count:
            return [len(idsMixedMobileNetV2) - len(idsImageOnlyMobileNetV2), len(idsMixedEfficientNetB0) - len(idsImageOnlyEfficientNetB0), len(idsMixedEfficientNetB4) - len(idsImageOnlyEfficientNetB4), len(idsMixedResNet50V2) - len(idsImageOnlyResNet50V2)]
        else:
            imageIdsImageOnly = set(idsImageOnlyMobileNetV2).intersection(set(idsImageOnlyEfficientNetB0)).intersection(set(idsImageOnlyEfficientNetB4)).intersection(set(idsImageOnlyResNet50V2))
            imageIdsMixed = set(idsMixedMobileNetV2).intersection(set(idsMixedEfficientNetB0)).intersection(set(idsMixedEfficientNetB4)).intersection(set(idsMixedResNet50V2))
            # return 
            # - the image ids correctly classified by the mixed model but wrongly classified by image-only
            # - the union of image-only and mixed image ids (wrongly classified by both)
            return imageIdsImageOnly - imageIdsMixed, imageIdsMixed.union(imageIdsImageOnly)

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
    
    def exifTagDistribution(self):
        tagList = [file for file in self.basePath.glob('*tag_distribution.csv') if file.is_file()]
        if len(tagList) == 0:
            raise ValueError("tag distribution file not found in base directory")
        
        tagDistribution = pd.read_csv(tagList[0], header = None, index_col = 0).transpose()

        exifTags = []
        for tag in ExifTag:
            exifTags.append(tag.name)
        
        return tagDistribution[tagDistribution.columns.intersection(exifTags)]
    
    def getHighResLowResClassificationReport(self, super: bool = False, metric: str = "f1-score"):
        cfHighRes = self.getClassificationReport(super = super, 
                                                 metric = metric, 
                                                 customClasses = ["macro avg"],
                                                 highResolution = True)
        
        cfLowRes = self.getClassificationReport(super = super, 
                                                metric = metric, 
                                                customClasses = ["macro avg"],
                                                highResolution = False)
        return cfHighRes, cfLowRes

    def individualClassSupport(self):
        valClassificationReport = self.getClassificationReport(metric = "support", reportType = EvaluationFiles.VALIDATION_CLASSIFICATION_REPORT)
        testClassificationReport = self.getClassificationReport(metric = "support", reportType = EvaluationFiles.TEST_CLASSIFICATION_REPORT)
        trainingClassificationReport = self.getClassificationReport(metric = "support", reportType = EvaluationFiles.TRAINING_CLASSIFICATION_REPORT)

        support = np.add(valClassificationReport.metricExifOnly, testClassificationReport.metricExifOnly)
        support = np.add(support, trainingClassificationReport.metricExifOnly)
        support = list(zip(support, valClassificationReport.classNames))
        return support
    
    def trainingSetImageClassExamples(self, index: int = 0):
        if self.datasetPath == None:
            raise ValueError("no dataset path given for image examples")
        
        subDirs = [directory for directory in self.datasetPath.glob("*") if directory.is_dir()]
        images = []
        for subDir in subDirs:
            dataDirs = [directory for directory in subDir.glob("*") if directory.is_dir()]
            for dataDir in dataDirs:
                imagePath = dataDir.joinpath("images")
                imagePaths = [image for image in imagePath.glob("*") if image.is_file()]
                images.append((dataDir.stem, imagePaths[index]))  
        return images
    
    def trainingTimes(self, super: bool = False, exif: bool = False):
        if not super:
            if exif:
                return np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][EvaluationFiles.TRAINING_TIME]])
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
            if exif:
                return np.array([self.evaluationTargetFiles[EvaluationTarget.SUPER_EXIF_ONLY][EvaluationFiles.TRAINING_TIME]])
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

        return trainingTimeIO50, trainingTimeMixed50, trainingTimeIO150, trainingTimeMixed150
    
    def getFiles(self, fileType: EvaluationFiles, super: bool = True, highResolution: bool = True) -> FileList:
        """ Returns a specific type of evluation file for all target networks. """
        if super:
            fileExifOnly = self.evaluationTargetFiles[EvaluationTarget.SUPER_EXIF_ONLY][fileType]
            if highResolution:
                fileImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2][fileType]
                fileImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0][fileType]
                fileImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4][fileType]
                fileImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2][fileType]
                fileMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2][fileType]
                fileMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0][fileType]
                fileMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4][fileType]
                fileMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_150_RESNET_50V2][fileType]
            else:
                fileImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][fileType]
                fileImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0][fileType]
                fileImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4][fileType]
                fileImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2][fileType]
                fileMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2][fileType]
                fileMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0][fileType]
                fileMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4][fileType]
                fileMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUPER_MIXED_50_RESNET_50V2][fileType]
        else:
            fileExifOnly = self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][fileType]
            if highResolution:
                fileImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][fileType]
                fileImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][fileType]
                fileImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][fileType]
                fileImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][fileType]
                fileMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][fileType]
                fileMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][fileType]
                fileMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][fileType]
                fileMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_RESNET_50V2][fileType]
            else:
                fileImageOnlyMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][fileType]
                fileImageOnlyEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0][fileType]
                fileImageOnlyEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4][fileType]
                fileImageOnlyResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2][fileType]
                fileMixedMobileNetV2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_MOBILENET_V2][fileType]
                fileMixedEfficientNetB0 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0][fileType]
                fileMixedEfficientNetB4 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4][fileType]
                fileMixedResNet50V2 = self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_50_RESNET_50V2][fileType]
        
        return FileList(fileExifOnly,
                        fileImageOnlyMobileNetV2,
                        fileImageOnlyEfficientNetB0,
                        fileImageOnlyEfficientNetB4,
                        fileImageOnlyResNet50V2,
                        fileMixedMobileNetV2,
                        fileMixedEfficientNetB0,
                        fileMixedEfficientNetB4,
                        fileMixedResNet50V2)

    def createClassificationReport(self, 
                                   evaluationTarget: EvaluationTarget, 
                                   labelFile: EvaluationFiles = EvaluationFiles.VALIDATION_LABELS, 
                                   super: bool = False, 
                                   threshold: float = 0.5):
        """ Creates a classification report for the given target and label file. """
        labels = self.evaluationTargetFiles[evaluationTarget][labelFile]
        classNames = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2 if super else EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.CLASS_NAMES]

        trueY = []
        predictedY = []

        for index, row in labels.iterrows():
            classProbabilities = [float(x) for x in row["predicted_prob"].replace("[", "").replace("]", "").replace("\n", "").split(" ") if x != ""]
            classLabels = [int(x) for x in row["true"].replace("[", "").replace("]", "").replace("\n", "").replace(".", "").split(" ") if x != ""]

            if len(classProbabilities) != len(classLabels):
                raise ValueError("number of true labels don't match number of predicted labels")

            if super:
                predictedY.append([1 if i == np.argmax(classProbabilities) else 0 for i,c in enumerate(classProbabilities)])
            else:
                predictedY.append([1 if i >= threshold else 0 for i in classProbabilities])

            trueY.append(classLabels)

        cr = classification_report(trueY, predictedY, target_names = classNames, digits = 3, output_dict = True, zero_division = 0)
        df = pd.DataFrame(data = cr).round(3)
        return df

    def getClassificationReport(self, 
                                super: bool = False, 
                                highResolution: bool = False, 
                                metric: str = "f1-score",
                                customClasses: List = None,
                                reportType: int = -1):

        if reportType == 0:
            labelFile = EvaluationFiles.TRAINING_LABELS
        elif reportType == 1:
            labelFile = EvaluationFiles.TEST_LABELS
        else:
            labelFile = EvaluationFiles.VALIDATION_LABELS
        
        if not customClasses == None:
            classNames = customClasses

        if super:
            if customClasses == None:
                classNames = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.CLASS_NAMES]

            valReportExifOnly = self.createClassificationReport(EvaluationTarget.SUPER_EXIF_ONLY, super = True)
            if highResolution:
                valReportImageOnlyMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2, labelFile = labelFile, super = True)
                valReportImageOnlyEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B0, labelFile = labelFile, super = True)
                valReportImageOnlyEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_150_EFFICIENTNET_B4, labelFile = labelFile, super = True)
                valReportImageOnlyResNet50V2 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_150_RESNET_50V2, labelFile = labelFile, super = True)
                valReportMixedMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_150_MOBILENET_V2, labelFile = labelFile, super = True)
                valReportMixedEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B0, labelFile = labelFile, super = True)
                valReportMixedEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_150_EFFICIENTNET_B4, labelFile = labelFile, super = True)
                valReportMixedResNet50V2 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_150_RESNET_50V2, labelFile = labelFile, super = True)
            else:
                valReportImageOnlyMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2, labelFile = labelFile, super = True)
                valReportImageOnlyEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B0, labelFile = labelFile, super = True)
                valReportImageOnlyEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_50_EFFICIENTNET_B4, labelFile = labelFile, super = True)
                valReportImageOnlyResNet50V2 = self.createClassificationReport(EvaluationTarget.SUPER_IMAGEONLY_50_RESNET_50V2, labelFile = labelFile, super = True)
                valReportMixedMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_50_MOBILENET_V2, labelFile = labelFile, super = True)
                valReportMixedEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B0, labelFile = labelFile, super = True)
                valReportMixedEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_50_EFFICIENTNET_B4, labelFile = labelFile, super = True)
                valReportMixedResNet50V2 = self.createClassificationReport(EvaluationTarget.SUPER_MIXED_50_RESNET_50V2, labelFile = labelFile, super = True)
        else:
            if customClasses == None:
                classNames = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.CLASS_NAMES]

            valReportExifOnly = self.createClassificationReport(EvaluationTarget.SUB_EXIF_ONLY)
            if highResolution:
                valReportImageOnlyMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2, labelFile = labelFile)
                valReportImageOnlyEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0, labelFile = labelFile)
                valReportImageOnlyEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4, labelFile = labelFile)
                valReportImageOnlyResNet50V2 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2, labelFile = labelFile)
                valReportMixedMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_150_MOBILENET_V2, labelFile = labelFile)
                valReportMixedEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0, labelFile = labelFile)
                valReportMixedEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4, labelFile = labelFile)
                valReportMixedResNet50V2 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_150_RESNET_50V2, labelFile = labelFile)
            else:
                valReportImageOnlyMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2, labelFile = labelFile)
                valReportImageOnlyEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0, labelFile = labelFile)
                valReportImageOnlyEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B4, labelFile = labelFile)
                valReportImageOnlyResNet50V2 = self.createClassificationReport(EvaluationTarget.SUB_IMAGEONLY_50_RESNET_50V2, labelFile = labelFile)
                valReportMixedMobileNetV2 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_50_MOBILENET_V2, labelFile = labelFile)
                valReportMixedEfficientNetB0 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0, labelFile = labelFile)
                valReportMixedEfficientNetB4 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B4, labelFile = labelFile)
                valReportMixedResNet50V2 = self.createClassificationReport(EvaluationTarget.SUB_MIXED_50_RESNET_50V2, labelFile = labelFile)
            
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

        cfHighRes, cfLowRes = self.getHighResLowResClassificationReport(super = super, metric = metric)
        
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
    

    def createMixedImageOnlyAbsolutesGroup(self, 
                                           evaluations: List["ModelEvaluation"],
                                           categoryLabels: List[str],
                                           metric: str = "f1-score", 
                                           figSize: Tuple = (7, 5), 
                                           barWidth: float = 0.80,
                                           legendPosition: str = "lower left",
                                           labelOffset: float = 0.001,
                                           yLimit: float = None,
                                           savePath = None):
    
        exifOnly = np.empty(shape = (0))
        mobileNetV2ImageOnly = np.empty(shape = (0))
        efficientNetB0ImageOnly = np.empty(shape = (0))
        efficientNetB4ImageOnly = np.empty(shape = (0))
        resNet50V2ImageOnly = np.empty(shape = (0))

        mobileNetV2Mixed = np.empty(shape = (0))
        efficientNetB0Mixed = np.empty(shape = (0))
        efficientNetB4IMixed = np.empty(shape = (0))
        resNet50V2Mixed = np.empty(shape = (0))

        for modelEvaluation in evaluations:
            cfHighRes, cfLowRes = modelEvaluation.getHighResLowResClassificationReport(super = True, metric = metric)

            exifOnly = np.concatenate((exifOnly, np.concatenate([cfLowRes.metricExifOnly] + [cfLowRes.metricExifOnly])))
            mobileNetV2ImageOnly = np.concatenate((mobileNetV2ImageOnly, np.concatenate([cfLowRes.metricImageOnlyMobileNetV2] + [cfHighRes.metricImageOnlyMobileNetV2])))
            mobileNetV2Mixed = np.concatenate((mobileNetV2Mixed, np.concatenate([cfLowRes.metricMixedMobileNetV2] + [cfHighRes.metricMixedMobileNetV2])))
            efficientNetB0ImageOnly = np.concatenate((efficientNetB0ImageOnly, np.concatenate([cfLowRes.metricImageOnlyEfficientNetB0] + [cfHighRes.metricImageOnlyEfficientNetB0])))
            efficientNetB0Mixed = np.concatenate((efficientNetB0Mixed, np.concatenate([cfLowRes.metricMixedEfficientNetB0] + [cfHighRes.metricMixedEfficientNetB0])))
            efficientNetB4ImageOnly = np.concatenate((efficientNetB4ImageOnly, np.concatenate([cfLowRes.metricImageOnlyEfficientNetB4] + [cfHighRes.metricImageOnlyEfficientNetB4])))
            efficientNetB4IMixed = np.concatenate((efficientNetB4IMixed, np.concatenate([cfLowRes.metricMixedEfficientNetB4] + [cfHighRes.metricMixedEfficientNetB4])))
            resNet50V2ImageOnly = np.concatenate((resNet50V2ImageOnly, np.concatenate([cfLowRes.metricImageOnlyResNet50V2] + [cfHighRes.metricImageOnlyResNet50V2])))
            resNet50V2Mixed = np.concatenate((resNet50V2Mixed, np.concatenate([cfLowRes.metricMixedResNet50V2] + [cfHighRes.metricMixedResNet50V2])))

            cfHighRes, cfLowRes = modelEvaluation.getHighResLowResClassificationReport(super = False, metric = metric)

            exifOnly = np.concatenate((exifOnly, np.concatenate([cfLowRes.metricExifOnly] + [cfLowRes.metricExifOnly])))
            mobileNetV2ImageOnly = np.concatenate((mobileNetV2ImageOnly, np.concatenate([cfLowRes.metricImageOnlyMobileNetV2] + [cfHighRes.metricImageOnlyMobileNetV2])))
            mobileNetV2Mixed = np.concatenate((mobileNetV2Mixed, np.concatenate([cfLowRes.metricMixedMobileNetV2] + [cfHighRes.metricMixedMobileNetV2])))
            efficientNetB0ImageOnly = np.concatenate((efficientNetB0ImageOnly, np.concatenate([cfLowRes.metricImageOnlyEfficientNetB0] + [cfHighRes.metricImageOnlyEfficientNetB0])))
            efficientNetB0Mixed = np.concatenate((efficientNetB0Mixed, np.concatenate([cfLowRes.metricMixedEfficientNetB0] + [cfHighRes.metricMixedEfficientNetB0])))
            efficientNetB4ImageOnly = np.concatenate((efficientNetB4ImageOnly, np.concatenate([cfLowRes.metricImageOnlyEfficientNetB4] + [cfHighRes.metricImageOnlyEfficientNetB4])))
            efficientNetB4IMixed = np.concatenate((efficientNetB4IMixed, np.concatenate([cfLowRes.metricMixedEfficientNetB4] + [cfHighRes.metricMixedEfficientNetB4])))
            resNet50V2ImageOnly = np.concatenate((resNet50V2ImageOnly, np.concatenate([cfLowRes.metricImageOnlyResNet50V2] + [cfHighRes.metricImageOnlyResNet50V2])))
            resNet50V2Mixed = np.concatenate((resNet50V2Mixed, np.concatenate([cfLowRes.metricMixedResNet50V2] + [cfHighRes.metricMixedResNet50V2])))

        createBarChart(
            [[mobileNetV2Mixed, efficientNetB0Mixed, efficientNetB4IMixed, resNet50V2Mixed],
             [mobileNetV2ImageOnly, efficientNetB0ImageOnly, efficientNetB4ImageOnly, resNet50V2ImageOnly]], 
            [["MobileNetV2 Fusion", "EfficientNetB0 Fusion", "EfficientNetB4 Fusion", "ResNet50V2 Fusion"],
             ["MobileNetV2 Image-Only", "EfficientNetB0 Image-Only", "EfficientNetB4 Image-Only", "ResNet50V2 Image-Only"]], 
            categoryLabels = categoryLabels, 
            showValues = False, 
            title = "", #"Mixed Model vs. Image Only " + metric.capitalize(),
            yLabel = "$" + "F1_{M}$",
            figSize = figSize,
            barWidth = barWidth,
            grid = True,
            yLabelFormatter = ticker.FuncFormatter(lambda x, pos: '{0:.2f}'.format(x)),
            valueFormat = "{:.3f}",
            yLimit = yLimit,
            savePath = savePath,
            legendPosition = legendPosition,
            additionalLines = exifOnly,
            labelOffset = labelOffset)
    
    def createMixedImageOnlyDeltaGroup(self, 
                                       evaluations: List["ModelEvaluation"],
                                       categoryLabels: List[str],
                                       metric: str = "f1-score", 
                                       figSize: Tuple = (7, 5), 
                                       barWidth: float = 0.80,
                                       legendPosition: str = "upper right",
                                       labelOffset: float = 0.001,
                                       yLimit: float = None,
                                       savePath = None):

        mobileNetV2Delta = np.empty(shape = (0))
        efficientNetB0Delta = np.empty(shape = (0))
        efficientNetB4Delta = np.empty(shape = (0))
        resNet50V2Delta = np.empty(shape = (0))

        for modelEvaluation in evaluations:
            cfHighRes, cfLowRes = modelEvaluation.getHighResLowResClassificationReport(super = True, metric = metric)

            mobileNetV2Delta = np.concatenate((mobileNetV2Delta, np.concatenate([cfLowRes.metricMixedMobileNetV2 - cfLowRes.metricImageOnlyMobileNetV2] + [cfHighRes.metricMixedMobileNetV2 - cfHighRes.metricImageOnlyMobileNetV2])))
            efficientNetB0Delta = np.concatenate((efficientNetB0Delta, np.concatenate([cfLowRes.metricMixedEfficientNetB0 - cfLowRes.metricImageOnlyEfficientNetB0] + [cfHighRes.metricMixedEfficientNetB0 - cfHighRes.metricImageOnlyEfficientNetB0])))
            efficientNetB4Delta = np.concatenate((efficientNetB4Delta, np.concatenate([cfLowRes.metricMixedEfficientNetB4 - cfLowRes.metricImageOnlyEfficientNetB4] + [cfHighRes.metricMixedEfficientNetB4 - cfHighRes.metricImageOnlyEfficientNetB4])))
            resNet50V2Delta = np.concatenate((resNet50V2Delta, np.concatenate([cfLowRes.metricMixedResNet50V2 - cfLowRes.metricImageOnlyResNet50V2] + [cfHighRes.metricMixedResNet50V2 - cfHighRes.metricImageOnlyResNet50V2])))

            cfHighRes, cfLowRes = modelEvaluation.getHighResLowResClassificationReport(super = False, metric = metric)

            mobileNetV2Delta = np.concatenate((mobileNetV2Delta, np.concatenate([cfLowRes.metricMixedMobileNetV2 - cfLowRes.metricImageOnlyMobileNetV2] + [cfHighRes.metricMixedMobileNetV2 - cfHighRes.metricImageOnlyMobileNetV2])))
            efficientNetB0Delta = np.concatenate((efficientNetB0Delta, np.concatenate([cfLowRes.metricMixedEfficientNetB0 - cfLowRes.metricImageOnlyEfficientNetB0] + [cfHighRes.metricMixedEfficientNetB0 - cfHighRes.metricImageOnlyEfficientNetB0])))
            efficientNetB4Delta = np.concatenate((efficientNetB4Delta, np.concatenate([cfLowRes.metricMixedEfficientNetB4 - cfLowRes.metricImageOnlyEfficientNetB4] + [cfHighRes.metricMixedEfficientNetB4 - cfHighRes.metricImageOnlyEfficientNetB4])))
            resNet50V2Delta = np.concatenate((resNet50V2Delta, np.concatenate([cfLowRes.metricMixedResNet50V2 - cfLowRes.metricImageOnlyResNet50V2] + [cfHighRes.metricMixedResNet50V2 - cfHighRes.metricImageOnlyResNet50V2])))
        
        summedMobileNetV2DeltaLowRes = sum([x for ind, x in enumerate(mobileNetV2Delta) if (ind + 2) % 2 == 0]) / (len(evaluations) * 2)
        summedMobileNetV2DeltaHighRes = sum([x for ind, x in enumerate(mobileNetV2Delta) if (ind + 2) % 2 == 1]) / (len(evaluations) * 2)
        summedEfficientNetB0DeltaLowRes = sum([x for ind, x in enumerate(efficientNetB0Delta) if (ind + 2) % 2 == 0]) / (len(evaluations) * 2)
        summedEfficientNetB0DeltaHighRes = sum([x for ind, x in enumerate(efficientNetB0Delta) if (ind + 2) % 2 == 1]) / (len(evaluations) * 2)
        summedEfficientNetB4DeltaLowRes = sum([x for ind, x in enumerate(efficientNetB4Delta) if (ind + 2) % 2 == 0]) / (len(evaluations) * 2)
        summedEfficientNetB4DeltaHighRes = sum([x for ind, x in enumerate(efficientNetB4Delta) if (ind + 2) % 2 == 1]) / (len(evaluations) * 2)
        summedResNet50V2DeltaLowRes = sum([x for ind, x in enumerate(resNet50V2Delta) if (ind + 2) % 2 == 0]) / (len(evaluations) * 2)
        summedResNet50V2DeltaHighRes = sum([x for ind, x in enumerate(resNet50V2Delta) if (ind + 2) % 2 == 1]) / (len(evaluations) * 2)

        totalDeltaLow = [summedMobileNetV2DeltaLowRes, summedEfficientNetB0DeltaLowRes, summedEfficientNetB4DeltaLowRes, summedResNet50V2DeltaLowRes]
        totalDeltaHigh = [summedMobileNetV2DeltaHighRes, summedEfficientNetB0DeltaHighRes, summedEfficientNetB4DeltaHighRes, summedResNet50V2DeltaHighRes]

        createBarChart(
            [[mobileNetV2Delta, efficientNetB0Delta, efficientNetB4Delta, resNet50V2Delta]], 
            [["MobileNetV2", "EfficientNetB0", "EfficientNetB4", "ResNet50V2"]], 
            categoryLabels = categoryLabels, 
            showValues = False, 
            title = "", #"Mixed Model vs. Image Only " + metric.capitalize() + " Delta",
            yLabel = "Δ $" + "F1_{M}$",
            figSize = figSize,
            barWidth = barWidth,
            grid = True,
            yLimit = yLimit,
            savePath = savePath,
            legendPosition = legendPosition,
            labelOffset = labelOffset)
        
        createBarChart(
            [[totalDeltaLow, totalDeltaHigh]], 
            [["50x50px", "150x150px"]], 
            categoryLabels = ["MobileNetV2", "EfficientNetB0", "EfficientNetB4", "ResNet50V2"], 
            showValues = True, 
            valueFormat = "{:.3f}",
            labelPrefix = "+",
            title = "Mixed Model Total Average " + metric.capitalize() + " Gain",
            yLabel = metric.capitalize() + " Gain",
            figSize = (8.5, 5),
            barWidth = barWidth,
            grid = True,
            savePath = None,
            legendPosition = legendPosition,
            labelOffset = -0.001)
        
        createBarChart(
            [[[sum(totalDeltaLow) / len(totalDeltaLow)], [sum(totalDeltaHigh) / len(totalDeltaHigh)]]], 
            [["50x50px", "150x150px"]], 
            categoryLabels = [""], 
            showValues = True, 
            valueFormat = "{:.3f}",
            labelPrefix = "+",
            title = "Mixed Model Aggregated Total Average " + metric.capitalize() + " Gain",
            yLabel = metric.capitalize() + " Gain",
            figSize = (6, 5),
            barWidth = 0.5,
            grid = True,
            savePath = None,
            legendPosition = legendPosition,
            labelOffset = -0.001)
    
    def createMixedImageExifComparison(self, 
                                  super: bool = False, 
                                  metric: str = "f1-score", 
                                  figSize = (5.5, 5), 
                                  barWidth: float = 0.6,
                                  labelOffset: float = 0.025,
                                  savePath = None):
        
        cfHighRes = self.getClassificationReport(super = super, 
                                                 metric = metric, 
                                                 customClasses = ["macro avg"], 
                                                 highResolution = True)
        
        cfLowRes = self.getClassificationReport(super = super, 
                                                metric = metric, 
                                                customClasses = ["macro avg"], 
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

    def createTrainingTimeMixedvsImageOnlyComparisonGrouped(self,
                                                            evaluations: List["ModelEvaluation"],
                                                            super: bool = False,
                                                            combineSuperSub: bool = False,
                                                            separateCnns: bool = True,
                                                            figSize: Tuple = (6, 2.8), #5.0
                                                            barWidth: float = 0.8,
                                                            savePath = None):
        totalsIO50 = np.zeros(shape = (4))
        totalsMixed50 = np.zeros(shape = (4))
        totalsIO150 = np.zeros(shape = (4))
        totalsMixed150 = np.zeros(shape = (4))
        totalExif = np.zeros(shape = (1))

        fracs50MobileNetV2 = list()
        fracs50EfficientNetB0 = list()
        fracs50EfficientNetB4 = list()
        fracs50ResNet50V2 = list()

        fracs150MobileNetV2 = list()
        fracs150EfficientNetB0 = list()
        fracs150EfficientNetB4 = list()
        fracs150ResNet50V2 = list()

        targets = [super]

        if combineSuperSub:
            targets = [True, False] # targets for super and sub together
        
        loops = 0
        for modelEvaluation in evaluations:
            for target in targets:
                loops += 1
                trainingTimeIO50, trainingTimeMixed50, trainingTimeIO150, trainingTimeMixed150 = modelEvaluation.trainingTimes(super = target)
                exifTime = modelEvaluation.trainingTimes(super = target, exif = True)

                totalExif = np.add(totalExif, exifTime)
                totalsIO50 = np.add(totalsIO50, trainingTimeIO50)
                totalsMixed50 = np.add(totalsMixed50, trainingTimeMixed50)

                sum50IO = np.sum(trainingTimeIO50)
                sum50Mixed = np.sum(trainingTimeMixed50)
                frac50 = 1.0 - np.divide(trainingTimeMixed50, trainingTimeIO50)

                fracs50MobileNetV2.append(frac50[0])
                fracs50EfficientNetB0.append(frac50[1])
                fracs50EfficientNetB4.append(frac50[2])
                fracs50ResNet50V2.append(frac50[3])

                #print("sum:50 IO " + str(sum50IO))
                #print("sum:50 Mixed " + str(sum50Mixed))
                #print("total_dif:50 " + str(totalDif50))
                #print("frac:50 " + str(frac50))

                totalsIO150 = np.add(totalsIO150, trainingTimeIO150)
                totalsMixed150 = np.add(totalsMixed150, trainingTimeMixed150)
        
                sum150IO = np.sum(trainingTimeIO150)
                sum150Mixed = np.sum(trainingTimeMixed150)
                frac150 = 1.0 - np.divide(trainingTimeMixed150, trainingTimeIO150)

                fracs150MobileNetV2.append(frac150[0])
                fracs150EfficientNetB0.append(frac150[1])
                fracs150EfficientNetB4.append(frac150[2])
                fracs150ResNet50V2.append(frac150[3])

                #print("sum:150 IO " + str(sum150IO))
                #print("sum:150 Mixed " + str(sum150Mixed))
                #print("total_dif:150 " + str(totalDif150))
                #print("frac:150 " + str(frac150))

        
        avg50 = [np.sum(fracs50MobileNetV2) / len(fracs50MobileNetV2), np.sum(fracs50EfficientNetB0) / len(fracs50EfficientNetB0), np.sum(fracs50EfficientNetB4) / len(fracs50EfficientNetB4), np.sum(fracs50ResNet50V2) / len(fracs50ResNet50V2)]
        avg150 = [np.sum(fracs150MobileNetV2) / len(fracs150MobileNetV2), np.sum(fracs150EfficientNetB0) / len(fracs150EfficientNetB0), np.sum(fracs150EfficientNetB4) / len(fracs150EfficientNetB4), np.sum(fracs150ResNet50V2) / len(fracs150ResNet50V2)]
        allAvgs = avg50 + avg150

        averageIOTime50 = (sum(totalsIO50) / len(totalsIO50)) / loops
        averageIOTime150 = (sum(totalsIO150) / len(totalsIO150)) / loops
        averageMixed50Time = (sum(totalsMixed50) / len(totalsMixed50)) / loops
        averageMixed150Time = (sum(totalsMixed150) / len(totalsMixed150)) / loops
        averageExifTime = sum(totalExif) / loops

        print(averageIOTime50 / 60)
        print(averageIOTime150 / 60)
        print(averageMixed50Time / 60)
        print(averageMixed150Time / 60)
        print(averageExifTime / 60)

        if separateCnns:
            fracs50 = (1.0 - np.divide(totalsMixed50, totalsIO50)) * -100
            fracs150 = (1.0 - np.divide(totalsMixed150, totalsIO150)) * -100

            fracs50 = [x * -100 for x in avg50]
            fracs150 = [x * -100 for x in avg150]
        else:
            fracs50 = [(1.0 - (sum(totalsMixed50) / sum(totalsIO50))) * -100]
            fracs150 = [(1.0 - (sum(totalsMixed150) / sum(totalsIO150))) * -100]

            fracs50 = [(sum(avg50) / len(avg50)) * -100]
            fracs150 = [(sum(avg150) / len(avg150)) * -100]

        createBarChart(
            [[fracs50, fracs150]], 
            [["50x50px", "150x150px"]], 
            categoryLabels = None if not separateCnns else ["MobileNetV2", "EfficientNetB0", "EfficientNetB4", "ResNet50V2"], 
            showValues = True,
            showNegativeValues = True,
            valueFormat = "{:.1f}",
            labelOffset = 0.9, #-0.4,
            labelPostfix = "%",
            title = "" if combineSuperSub else "Super" if super else "",
            yLabel = "Δ Average Training Time in %", # \n(Mixed vs. Image-Only)",
            figSize = figSize,
            savePath = savePath,
            legendPosition = "lower right",
            barWidth = barWidth,
            grid = True)
    
    def createTrainingTimeMixedvsImageOnlyComparison(self, 
                                                    super: bool = False,
                                                    figSize: Tuple = (7, 6),
                                                    barWidth: float = 0.9,
                                                    savePath = None):
    
        trainingTimeIO50, trainingTimeMixed50, trainingTimeIO150, trainingTimeMixed150 = self.trainingTimes(super = super)
        
        print("50x50:")
        print(np.round(np.subtract(np.ones(shape = (4)), np.divide(trainingTimeMixed50, trainingTimeIO50)), 3) * -100)
        print("150x150:")
        print(np.round(np.subtract(np.ones(shape = (4)), np.divide(trainingTimeMixed150, trainingTimeIO150)), 3) * -100)

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
        
    def createModelParameterCountComparison(self, figSize: Tuple = (12.5, 5), barWidth: float = 0.7, savePath = None):
        params = np.array([self.evaluationTargetFiles[EvaluationTarget.SUB_EXIF_ONLY][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2][EvaluationFiles.MODEL_PARAMS], 
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0][EvaluationFiles.MODEL_PARAMS], 
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B4][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_RESNET_50V2][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_MOBILENET_V2][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B4][EvaluationFiles.MODEL_PARAMS],
                           self.evaluationTargetFiles[EvaluationTarget.SUB_MIXED_150_RESNET_50V2][EvaluationFiles.MODEL_PARAMS]])
        scale_y = 1e6
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

        featureImportanceDf = featureImportanceDf.reindex(sorted(featureImportanceDf.columns), axis = 1)
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

        rp = self.getFiles(fileType = EvaluationFiles.TRAINING_HISTORY, super = super, highResolution = highResolution)
        
        # identify accuracy metric key
        if loss:
            metricIndexKey = "loss"
        else:
            for value in rp.fileExifOnly.index.to_numpy():
                if "accuracy" in value and not "val" in value:
                    metricIndexKey = value
                    break
        
        pixels = " (150x150px)" if highResolution else " (50x50px)"
        s = " (Super)" if super else ""
        metric = "Accuracy" if not metricIndexKey == "loss" else "Loss"
        
        createTrainingAccuracyLossChart(dataFrames = [rp.fileExifOnly],
                                        dataIndexKey = metricIndexKey, 
                                        valDataIndexKey = "val_" + metricIndexKey,
                                        labels = ["EXIF MLP"],
                                        title = "Training " + metric + " at Epoch - EXIF-Only" + s,
                                        figSize = (5.5, 3.0),
                                        targetEpochPatience = 50,
                                        savePath = savePath)

        createTrainingAccuracyLossChart(dataFrames = [rp.fileImageOnlyMobileNetV2, 
                                                      rp.fileImageOnlyEfficientNetB0,
                                                      rp.fileImageOnlyEfficientNetB4,
                                                      rp.fileImageOnlyResNet50V2],
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
        
        createTrainingAccuracyLossChart(dataFrames = [rp.fileMixedMobileNetV2, 
                                                      rp.fileMixedEfficientNetB0,
                                                      rp.fileMixedEfficientNetB4,
                                                      rp.fileMixedResNet50V2],
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
        
    def createExifTagDistributionChart(self, customSet: pd.DataFrame = None, figSize: Tuple = (8, 5), barHeight: float = 0.7, savePath = None):
        distribution = self.exifTagDistribution() if customSet is None else customSet
        distribution = distribution.sort_values(by = 1, ascending = False, axis = 1)
        tags = distribution.columns.to_numpy()
        tagsCount = distribution.iloc[0].to_numpy()

        # convert to percentage
        tagsCount = (tagsCount / np.max(tagsCount)) * 100

        createSingleBarChartHorizontal(tagsCount,
                                       seriesLabels = [],
                                       categoryLabels = tags, 
                                       figSize = figSize,
                                       barHeight = barHeight,
                                       showValues = True,
                                       sc = True,
                                       valueFormat = "{:.0f}",
                                       title = ((self.basePath.stem + " - Dataset EXIF-Tag Distribution") if customSet is None else "EXIF-Tag Distribution"),
                                       xLimit = 105,
                                       labelPostFix = "%",
                                       labelOffset = 4,
                                       savePath = savePath,
                                       grid = False)
    
    def createClassesImagesOverviewChart(self, index: int = 0, figSize: Tuple = (9, 6), imagesPerRow: int = 6, savePath = None):
        createImageOverviewChart(self.trainingSetImageClassExamples(index = index), 
                                figSize = figSize, 
                                imagesPerRow = imagesPerRow,
                                savePath = savePath)
    
    def createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(self, 
                                                                        super: bool = True,
                                                                        highResolution: bool = True, 
                                                                        figSize: Tuple = (7, 4),
                                                                        imageIndex: List = None,
                                                                        imageIds: List = None,
                                                                        savePath = None):
        if self.datasetPath == None:
            raise ValueError("no dataset path given for image examples")
        
        gainedIds, _ = self.imageIds(highResolution = highResolution, super = super)

        if super:
            labels = self.evaluationTargetFiles[EvaluationTarget.SUPER_IMAGEONLY_150_MOBILENET_V2 if highResolution else EvaluationTarget.SUPER_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.VALIDATION_LABELS]
        else:
            labels = self.evaluationTargetFiles[EvaluationTarget.SUB_IMAGEONLY_150_MOBILENET_V2 if highResolution else EvaluationTarget.SUB_IMAGEONLY_50_MOBILENET_V2][EvaluationFiles.VALIDATION_LABELS]

        images = []
        imagePaths = [image for image in self.datasetPath.glob("**/*.jpg") if image.is_file()]
        gainedIds = list(gainedIds)

        if imageIndex == None:
            imageIndex = range(0, min(len(gainedIds), 20))

        for index in imageIndex:
                id = gainedIds[index]
                print(id)
                imagePath = [image for image in imagePaths if str(id) in image.stem][0]
                loc = labels[labels["id"] == id]
                trueLabels = loc["true_name"].item().replace("[", "").replace("]", "").replace("'", "")
                predictedLabels = loc["predicted_name"].item().replace("[", "").replace("]", "").replace("'", "")
                images.append(("true: " + trueLabels + "\npredicted: " + predictedLabels, imagePath))
            
            
        createImageOverviewChart(images, 
                                figSize = figSize, 
                                imagesPerRow = 5,
                                savePath = savePath)

    def createConfusionMatrix(self, evaluationTarget: EvaluationTarget, threshold: int = 0.5, argMaxOnly: bool = True, addNotPredictedClass: bool = False, savePath = None):
        labels = self.evaluationTargetFiles[evaluationTarget][EvaluationFiles.VALIDATION_LABELS]
        superClassNames = self.getFiles(fileType = EvaluationFiles.CLASS_NAMES, super = True).fileExifOnly
        classNames = self.getFiles(fileType = EvaluationFiles.CLASS_NAMES, super = False).fileExifOnly
        subOnlyClassNames = [x for x in classNames if x not in superClassNames]

        trueY = []
        predictedY = []

        # index for class 'other'
        otherIndex = len(classNames) + 1
        
        for index, row in labels.iterrows():
            imageId = row["id"]
            classProbabilities = [float(x) for x in row["predicted_prob"].replace("[", "").replace("]", "").replace("\n", "").split(" ") if x != ""]
            trueClassNames = set([x.strip() for x in row["true_name"].replace("[", "").replace("]", "").replace("\n", "").replace("'", "").split(",") if x != ""])

            # get predicted class names depending on probability and threshold
            predictedClassIndex = [1 if i >= threshold else 0 for i in classProbabilities]
            predictedClassNames = set([classNames[i] for i in range(len(predictedClassIndex)) if predictedClassIndex[i] == 1])

            # we only consider insances with assigned sub-label
            if len(trueClassNames) == 2:
                # determine true label index of sub-concepet label
                subOnlyLabel = [x for x in trueClassNames if x not in superClassNames][0]
                trueLabelIndex = classNames.index(subOnlyLabel)
                
                if not argMaxOnly:
                    # code without argMax
                    trueY.append(trueLabelIndex)
                    if subOnlyLabel in predictedClassNames:
                        # correct prediction
                        predictedY.append(trueLabelIndex)
                    else:
                        # false predcition
                        falseLabels = list(predictedClassNames.difference(trueClassNames))
                        if len(falseLabels) > 0:
                            # decision how to treat different cases
                            # we add the sub-label class as predicted value if there is exaclty one sub-class present in false labels, else we use class other
                            isSubConcepts = [(lambda x: x not in superClassNames)(x) for x in falseLabels]
                            if isSubConcepts.count(True) == 1:
                                subClassIndex = next(i for i,c in enumerate(isSubConcepts) if c == True)
                                subClass = falseLabels[subClassIndex]
                                predictedY.append(classNames.index(subClass))
                            else:
                                predictedY.append(otherIndex)
                        else:
                            # no sub-label was predicted at all
                            if addNotPredictedClass:
                                predictedY.append(otherIndex + 1)
                            else:
                                predictedY.append(otherIndex)
                else:
                    # code with argMax
                    predictedLabelIndex = np.argmax(classProbabilities)
                    predictedLabel = classNames[predictedLabelIndex]
                    if predictedLabel in trueClassNames and not predictedLabel in superClassNames:
                        # correct prediction
                        trueY.append(trueLabelIndex)
                        predictedY.append(predictedLabelIndex)
                    elif not predictedLabel in superClassNames:
                        # false prediction, another sub-concept has maximum probabilty
                        trueY.append(trueLabelIndex)
                        predictedY.append(predictedLabelIndex)
                    else:
                        # a super-concpet has maximum probability
                        trueY.append(trueLabelIndex)
                        predictedY.append(otherIndex)

        displayLabels = subOnlyClassNames + ["other"]
        if not argMaxOnly and addNotPredictedClass:
            displayLabels.append("not predicted")

        plot = ConfusionMatrixDisplay.from_predictions(trueY, 
                                                       predictedY, 
                                                       display_labels = displayLabels, 
                                                       cmap = "Blues", 
                                                       ax = plt.subplots(figsize = (11, 10))[1])
        
        plt.rcParams["axes.titlepad"] = 25
        #plt.title("Confusion Matrix", fontsize = 25)
        plt.xlabel("Predicted Label", fontsize = 20, labelpad = 20)
        plt.ylabel("True Label", fontsize = 20, labelpad = 20)
    
        plt.xticks(rotation = 90, fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.tight_layout()

        if not savePath is None:
            plot.figure_.savefig(savePath, dpi = 300)

if __name__ == '__main__':

    evaluations = []

    evaluations.append(ModelEvaluation(directoryPath = "/Users/ralflederer/Desktop/models/ObjectLandscape", 
                                       datasetDirectoryPath = "/Users/ralflederer/Desktop/res/landscape_object_multilabel"))

    evaluations.append(ModelEvaluation(directoryPath = "/Users/ralflederer/Desktop/models/MovingStatic", 
                                       datasetDirectoryPath = "/Users/ralflederer/Desktop/res/moving_static"))
    
    evaluations.append(ModelEvaluation(directoryPath = "/Users/ralflederer/Desktop/models/IndoorOutdoor", 
                                       datasetDirectoryPath = "/Users/ralflederer/Desktop/res/indoor_outdoor_multilabel"))

    savePath = "/Users/ralflederer/Desktop/test.png"

    """ modelEvaluation.createImageResolutionComparisonBarChart(super = False, 
                                                            metric = "f1-score", 
                                                            customClasses = ["macro avg"], 
                                                            mixed = True) """

    # individual charts
    index = 2

    #print(evaluations[index].imageIds(super = False, highResolution = True, count = True))
    """  
    # plot for mixed model vs image model performance - super
    evaluations[index].createMixedImageResolutionComparisonBarChart(super = True, 
                                                                    metric = "f1-score", 
                                                                    highResolution = False,
                                                                    barWidth = 0.7,
                                                                    figSize=(6, 5),
                                                                    labelOffset = 0.025,
                                                                    savePath = None)
    
    # plot for mixed model vs image model performance
    evaluations[index].createMixedImageResolutionComparisonBarChart(super = False, 
                                                                    metric = "f1-score", 
                                                                    highResolution = False,
                                                                    barWidth = 0.8,
                                                                    figSize=(18, 7), #28,9
                                                                    labelOffset = 0.016,
                                                                    savePath = None)

    # plot for delta (mixed image only)
    evaluations[index].createMixedImageOnlyDelta(super = True, savePath = None) 
    evaluations[index].createMixedImageOnlyDelta(super = False, savePath = None)

    # plot for exif vs. best image only vs  best mixed model
    evaluations[index].createMixedImageExifComparison(super = True, savePath = None)
    evaluations[index].createMixedImageExifComparison(super = False, savePath = None)

    # plot for training time mixed model vs image only
    evaluations[index].createTrainingTimeMixedvsImageOnlyComparison(super = True, savePath = None)
    evaluations[index].createTrainingTimeMixedvsImageOnlyComparison(super = False, savePath = None)

    # plot for model parameter count comparison
    evaluations[index].createModelParameterCountComparison(savePath = None)

    # plot for exif tag feature importance (exif-only)
    evaluations[index].createEXIFFeatureImportanceChart(super = True, savePath = None)
    evaluations[index].createEXIFFeatureImportanceChart(super = False, savePath = None)

    # plots for training accuracy (exif only, image only, mixed) 
    evaluations[index].createTrainingComparisonPlot(super = True, highResolution = False)
    evaluations[index].createTrainingComparisonPlot(super = True, highResolution = True)
    evaluations[index].createTrainingComparisonPlot(super = False, highResolution = False)
    evaluations[index].createTrainingComparisonPlot(super = False, highResolution = True)

     # plots for training loss (exif only, image only, mixed) 
    evaluations[index].createTrainingComparisonPlot(super = True, highResolution = False, loss = True)
    evaluations[index].createTrainingComparisonPlot(super = True, highResolution = True, loss = True)
    evaluations[index].createTrainingComparisonPlot(super = False, highResolution = False, loss = True)
    evaluations[index].createTrainingComparisonPlot(super = False, highResolution = True, loss = True) 
    
    # plot for exif tag distribution in the data set 
    evaluations[index].createExifTagDistributionChart(savePath = None) """
    
    # dataset-individual plots

    # object-landscape
    #evaluations[0].createClassesImagesOverviewChart(index = 4, imagesPerRow = 5, figSize = (7, 3.5), savePath = None)
    #evaluations[0].createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(imageIndex = [0, 1, 2, 3, 4, 6, 8, 10, 13, 15], highResolution = False, figSize = (8.5, 4.5), savePath = None)

    # moving-static
    #evaluations[1].createClassesImagesOverviewChart(index = 63, imagesPerRow = 5, figSize = (7, 3.5), savePath = savePath)
    #evaluations[1].createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(imageIndex = [1, 2, 5, 7, 9, 14, 17, 22, 24, 29], highResolution = True, savePath = None)
    #evaluations[1].createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(imageIndex = [2, 3, 51, 11, 13, 14, 28, 33, 34, 38], highResolution = False, savePath = None) 

    # indoor-outdoor
    #evaluations[2].createClassesImagesOverviewChart(savePath = None)
    #evaluations[2].createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(imageIndex = [10, 1, 2, 5, 11, 30, 38, 36, 25, 33, 34], highResolution = True, savePath = None)
    #evaluations[2].createWrongByImageOnlyCorrectByMixedClassifiedImageExampleChart(imageIndex = [5, 6, 9, 12, 14, 34, 46, 53, 59, 21], highResolution = False, savePath = None)
    
    # combined evaluations 

    # EXIF-Tag distribution
    combinedDistribution = None
    for evaluation in evaluations:
        exifTagDistribution = evaluation.exifTagDistribution()
        if combinedDistribution is None:
            combinedDistribution = exifTagDistribution
        else:
            combinedDistribution = combinedDistribution.add(exifTagDistribution, fill_value = 0)
    evaluations[0].createExifTagDistributionChart(customSet = combinedDistribution, savePath = savePath)

    # Trining Time Comparison
    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, super = True)
    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, super = False)
    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, combineSuperSub = True, savePath = None)
    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, combineSuperSub = True, separateCnns = False)

    #evaluations[index].createMixedImageOnlyDelta(super = True, savePath = None) 
    #evaluations[index].createMixedImageOnlyDelta(super = False, savePath = None)

    # Image-Only vs. Mixed Delta (Super, Sub)
    """ evaluations[index].createMixedImageOnlyDeltaGroup(evaluations = evaluations, 
                                                      categoryLabels = ["Landscape-Object\nSuper 50x50px", "Landscape-Object\nSuper 150x150px", "Landscape-Object\nSub 50x50px", "Landscape-Object\nSub 150x150px", 
                                                                        "Moving-Static\nSuper 50x50px", "Moving-Static\nSuper 150x150px", "Moving-Static\nSub 50x50px", "Moving-Static\nSub 150x150px", 
                                                                        "Indoor-Outdoor\nSuper 50x50px", "Indoor-Outdoor\nSuper 150x150px", "Indoor-Outdoor\nSub 50x50px", "Indoor-Outdoor\nSub 150x150px"], 
                                                      figSize = (20, 9), 
                                                      barWidth = 0.7,
                                                      #yLimit = 0.05,
                                                      savePath = None) """


    """ evaluations[index].createMixedImageOnlyAbsolutesGroup(evaluations = evaluations, 
                                                          categoryLabels = ["Landscape-Object\nSuper 50x50px", "Landscape-Object\nSuper 150x150px", "Landscape-Object\nSub 50x50px", "Landscape-Object\nSub 150x150px", 
                                                                        "Moving-Static\nSuper 50x50px", "Moving-Static\nSuper 150x150px", "Moving-Static\nSub 50x50px", "Moving-Static\nSub 150x150px", 
                                                                        "Indoor-Outdoor\nSuper 50x50px", "Indoor-Outdoor\nSuper 150x150px", "Indoor-Outdoor\nSub 50x50px", "Indoor-Outdoor\nSub 150x150px"], 
                                                          figSize = (20, 9), 
                                                          barWidth = 0.6,
                                                          yLimit = 1.001,
                                                          savePath = savePath) """                                  

    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, combineSuperSub = True, separateCnns = False)
    #evaluations[2].createTrainingTimeMixedvsImageOnlyComparison(super = True, savePath = None)
    #evaluations[index].createModelParameterCountComparison(savePath = None)

    #evaluations[0].createConfusionMatrix(EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0, argMaxOnly = True)
    #evaluations[0].createTrainingTimeMixedvsImageOnlyComparisonGrouped(evaluations = evaluations, combineSuperSub = True, separateCnns = False, savePath = savePath)

    """ cfBasePath = "/Users/ralflederer/Desktop/cf/"
    argMaxOnly = True
    addNotPredictedClass = False
    for evaluation in evaluations:
        evaluation.createConfusionMatrix(EvaluationTarget.SUB_EXIF_ONLY, argMaxOnly = argMaxOnly, addNotPredictedClass = addNotPredictedClass, savePath = cfBasePath + evaluation.name + "_exif.png")
        evaluation.createConfusionMatrix(EvaluationTarget.SUB_IMAGEONLY_50_EFFICIENTNET_B0, argMaxOnly = argMaxOnly, addNotPredictedClass = addNotPredictedClass, savePath = cfBasePath + evaluation.name + "_io_low.png")
        evaluation.createConfusionMatrix(EvaluationTarget.SUB_MIXED_50_EFFICIENTNET_B0, argMaxOnly = argMaxOnly, addNotPredictedClass = addNotPredictedClass, savePath = cfBasePath + evaluation.name + "_mixed_low.png")
        evaluation.createConfusionMatrix(EvaluationTarget.SUB_IMAGEONLY_150_EFFICIENTNET_B0, argMaxOnly = argMaxOnly, addNotPredictedClass = addNotPredictedClass, savePath = cfBasePath + evaluation.name + "_io_high.png")
        evaluation.createConfusionMatrix(EvaluationTarget.SUB_MIXED_150_EFFICIENTNET_B0, argMaxOnly = argMaxOnly, addNotPredictedClass = addNotPredictedClass, savePath = cfBasePath + evaluation.name + "_mixed_high.png") """

    plt.show()