import sys
import os
from Tools.GUI.ArgsHelper import argForParam, paramSet
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.EvaluationTask import EvaluationTask

PARAM_MODEL_FILE_PATH = "-model"
PARAM_DATASET_FILE_PATH = "-datapath"
PARAM_TRAINING_DATA_FORMAT = "-tformat"
PARAM_HELP = "-help"
PARAM_CACHE_STORAGE_PATH = "-cachepath"
PARAM_HELP = "-help"
PARAM_EXIF_ONLY = "-eo"
PARAM_IMAGE_ONLY = "-io"
PARAM_IMAGE_SIZE = "-size"
PARAM_USE_SUPER_CONCEPTS = "-super"

def printUsage():
    print("Evaluation Usage: ")
    print("{model}: path to the keras model (.keras) file containing the model to use for evaluation".format(model = PARAM_MODEL_FILE_PATH))
    print("{datapath}: path to the zip file containing the data to evaluate".format(datapath = PARAM_DATASET_FILE_PATH))

    print("\toptional:")
    print("\t \t{cachepath}: path to the caching directory".format(cachepath = PARAM_CACHE_STORAGE_PATH))
    print("\t\t{tformat}: data format of the data to evaluate, [flickr, mirflickr], default = flickr".format(tformat = PARAM_TRAINING_DATA_FORMAT))
    print("\t \t{super}: if set, uses super concepts for evaluation".format(super = PARAM_USE_SUPER_CONCEPTS))
    print("\t \t{io}: model is image only".format(io = PARAM_IMAGE_ONLY))
    print("\t \t{eo}: model is exif only".format(eo = PARAM_EXIF_ONLY))
    print("\t \t{size}: image size to use (width/height), comma-separated".format(size = PARAM_IMAGE_SIZE))


def test():
    # create data provider
    dataProvider = ExifImageProvider(dataSetsPaths = { "/Users/ralflederer/Desktop/res/mi_indoor_outdoor_multilabel.zip" : ExifImageTrainingDataFormats.toTrainingDataFormat(name = "flickr") },
                                             cachePath = "/Users/ralflederer/Desktop/res",
                                             trainSize = 1.0,
                                             testSize = 0.0,
                                             validationSize = 0.0,
                                             batchSize = 32,
                                             imageSize = (150, 150),
                                             useSuperConcepts = True,
                                             exifOnly = False,
                                             imageOnly = False)
            
    #  create evaluation task
    evaluationTask = EvaluationTask(modelPath = "/Users/ralflederer/Desktop/model.keras", provider = dataProvider)
    evaluationReport = evaluationTask.run()
    print(evaluationReport.to_string())

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0 or (len(args) == 1 and paramSet(args, PARAM_HELP)):
        printUsage()
    else:
        try:
            modelFilePath = argForParam(args, PARAM_MODEL_FILE_PATH)
            datasetFilePath = argForParam(args, PARAM_DATASET_FILE_PATH)
            trainingDataFormat = argForParam(args, PARAM_TRAINING_DATA_FORMAT, "flickr")
            cacheDirectoryPath = argForParam(args, PARAM_CACHE_STORAGE_PATH)
            imageOnly = paramSet(args, PARAM_IMAGE_ONLY)
            exifOnly = paramSet(args, PARAM_EXIF_ONLY)
            useSuperConcepts = paramSet(args, PARAM_USE_SUPER_CONCEPTS)

            # input data checks
            if not os.path.exists(modelFilePath) or os.path.isdir(modelFilePath):
                raise ValueError("error: the given model file does not exist.")

            targetFormat = ExifImageTrainingDataFormats.toTrainingDataFormat(name = trainingDataFormat)
            if targetFormat is None:
                raise ValueError("error: unknown training data format: " + targetFormat)

            if not exifOnly:
                if paramSet(args, PARAM_IMAGE_SIZE):
                    imageSize = argForParam(args, PARAM_IMAGE_SIZE)
                    imageSize = imageSize.split(",") if not imageSize == None else []
                    if len(imageSize) != 2:
                        raise ValueError("error: invalid value for image size")
                    else:
                        imageSize = (int(imageSize[0]), int(imageSize[1]))
            
            # create data provider
            dataProvider = ExifImageProvider(dataSetsPaths = { datasetFilePath : targetFormat },
                                             cachePath = cacheDirectoryPath,
                                             trainSize = 1.0,
                                             testSize = 0.0,
                                             validationSize = 0.0,
                                             batchSize = 32,
                                             imageSize = imageSize if not exifOnly else None,
                                             useSuperConcepts = useSuperConcepts,
                                             exifOnly = exifOnly,
                                             imageOnly = imageOnly)
            
            #  create evaluation task
            evaluationTask = EvaluationTask(modelPath = modelFilePath, provider = dataProvider)
            evaluationReport = evaluationTask.run()
            print(evaluationReport.to_string())
        
        except Exception as exception:
            print(" ") 
            print(exception)
            print(" ")