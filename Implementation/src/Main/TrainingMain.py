import sys
import tensorflow
from DomainModel.TrainingDataFormat import ExifImageTrainingDataFormats
from Models.Classifiers.MLP import MLPClassifier
from Models.Classifiers.CNN import CNNClassifier, validModelNames, modelNameToModelFunction, standardModelNames
from Models.Classifiers.MixedModel import MixedModel
from Models.Training.Generators.BatchGeneratorProvider import ExifImageProvider
from Models.Training.Tasks.TrainingTask import TrainingTask
from Models.Training.Tasks.PermutationTask import ExifPermutationTask
from Tools.GUI.ArgsHelper import argForParam, paramSet

# learning parameters
PARAM_BATCH_SIZE = "-bs"
PARAM_EPOCHS = "-epochs"
PARAM_EARLY_STOPPING_PATIENCE_EPOCHS = "-esepochs"
PARAM_USE_SUPER_CONCEPTS = "-super"
PARAM_FILTER_CONCEPTS = "-filter"
PARAM_DATASET_FILE_PATH = "-datapath"
PARAM_CACHE_STORAGE_PATH = "-cachepath"
PARAM_MODEL_STORAGE_PATH = "-outpath"
PARAM_MODEL_NAME = "-name"
PARAM_SEED = "-seed"
PARAM_HELP = "-help"
PARAM_TRAIN_SIZE = "-trainsize"
PARAM_TEST_SIZE = "-testsize"
PARAM_VAL_SIZE = "-valsize"
PARAM_OPTIMIZATION_CRITERIA = "-optimize"

# params only relevant for cnn training
PARAM_IMAGE_ONLY = "-io"
PARAM_IMAGE_SIZE = "-size"
PARAM_FINE_TUNE_EPOCHS = "-tuneepochs"
PARAM_FINE_TUNE_LAYERS = "-tunelayers"
PARAM_BASE_MODEL_TYPE = "-basemodel"

# params only relevant for exif training
PARAM_EXIF_ONLY = "-eo"
PARAM_EXIF_TAGS = "-tags"
PARAM_EXIF_PERMUTATIONS = "-permutations"

# data format parameters
PARAM_TRAINING_DATA_FORMAT = "-tformat"

def printUsage():
    print("Training Usage: ")
    print(" ")
    print("{name}: output name of the final model".format(name = PARAM_MODEL_NAME))
    print("{datapath}: paths to the zip files containing the training data, comma-separated".format(datapath = PARAM_DATASET_FILE_PATH))
    print("{outpath}: path to the directory to store the created model / evaluation files".format(outpath = PARAM_MODEL_STORAGE_PATH))
    print("{batchSize}: batchSize to use for training".format(batchSize = PARAM_BATCH_SIZE))
    print("{epochs}: number of epochs to use for training".format(epochs = PARAM_EPOCHS))
    print("{esepochs}: number of early stopping patience epochs to use for training".format(esepochs = PARAM_EARLY_STOPPING_PATIENCE_EPOCHS))
    print("\toptional:")
    print("\t \t{cachepath}: path to the caching directory".format(cachepath = PARAM_CACHE_STORAGE_PATH))
    print("\t \t{super}: if set, uses super concepts for training".format(super = PARAM_USE_SUPER_CONCEPTS))
    print("\t \t{filter}: list of concepts to filter from the input data source, comma-separated".format(filter = PARAM_FILTER_CONCEPTS))
    print("\t \t{seed}: seed to use for training".format(seed = PARAM_SEED))
    print("\t \t{io}: training will be performed with image data only".format(io = PARAM_IMAGE_ONLY))
    print("\t \t{eo}: training will be performed with exif data only".format(eo = PARAM_EXIF_ONLY))
    print("\t \t{trainsize}: amount of training data to use for training, default = 0.7".format(trainsize = PARAM_TRAIN_SIZE))
    print("\t \t{testsize}: amount of training data to use for testing, default = 0.2".format(testsize = PARAM_TEST_SIZE))
    print("\t \t{valsize}: amount of training data to use for validation, default = 0.1".format(valsize = PARAM_VAL_SIZE))
    print("\t \t{tformat}: data format of the training data, comma-separated for multiple zip-files [flickr, mirflickr], default = flickr".format(tformat = PARAM_TRAINING_DATA_FORMAT))
    print("\t \t{optimize}: optimization criteria [accuracy, loss], default = accuracy".format(optimize = PARAM_OPTIMIZATION_CRITERIA))

    print(" ")
    print("\tparameters for cnn training (optional when exif only is set):")
    print("\t \t{basemodel}: name of the base-model architecture to use, see: ".format(basemodel = PARAM_BASE_MODEL_TYPE))
    print("\t \t{size}: image size to use (width/height), comma-separated".format(size = PARAM_IMAGE_SIZE))
    print("\t \t{tuneepochs}: number of fine-tune epochs to use for training".format(tuneepochs = PARAM_FINE_TUNE_EPOCHS))
    print("\t \t{tunelayers}: number of fine-tune layers of the model".format(tunelayers = PARAM_FINE_TUNE_LAYERS))

    print(" ")
    print("\tparameters for exif training (optional):")
    print("\t \t{tags}: list of exif tags to use for training, comma-separated".format(tags = PARAM_EXIF_TAGS))
    print("\t \t{permutations}: number of permutations used to assess the feature importance of exif tags, default = 50 (only when exif only)".format(permutations = PARAM_EXIF_PERMUTATIONS))

# main entry point
if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0 or (len(args) == 1 and paramSet(args, PARAM_HELP)):
        printUsage()
    else:
        try:
            # mandatory training args
            if all([paramSet(args, PARAM_MODEL_NAME), 
                        paramSet(args, PARAM_DATASET_FILE_PATH), 
                        paramSet(args, PARAM_MODEL_STORAGE_PATH), 
                        paramSet(args, PARAM_BATCH_SIZE),
                        paramSet(args, PARAM_EPOCHS),
                        paramSet(args, PARAM_EARLY_STOPPING_PATIENCE_EPOCHS)]):
                name = argForParam(args, PARAM_MODEL_NAME)
                datasetFilePath = argForParam(args, PARAM_DATASET_FILE_PATH)
                modelOutputPath = argForParam(args, PARAM_MODEL_STORAGE_PATH)
                batchSize = int(argForParam(args, PARAM_BATCH_SIZE))
                epochs = int(argForParam(args, PARAM_EPOCHS))
                earlyStoppingPatienceEpochs = int(argForParam(args, PARAM_EARLY_STOPPING_PATIENCE_EPOCHS))
            else:
                raise ValueError("error: not all mandatory parameters for training are set")

            # optional training args
            cacheDirectoryPath = argForParam(args, PARAM_CACHE_STORAGE_PATH)
            useSuperConcepts = paramSet(args, PARAM_USE_SUPER_CONCEPTS)
            filterConcepts = argForParam(args, PARAM_FILTER_CONCEPTS)
            filterConcepts = filterConcepts.split(",") if not filterConcepts == None else []
            trainSize = float(argForParam(args, PARAM_TRAIN_SIZE, 0.7))
            testSize = float(argForParam(args, PARAM_TEST_SIZE, 0.2))
            valSize = float(argForParam(args, PARAM_VAL_SIZE, 0.1))
            seed = int(argForParam(args, PARAM_SEED, 131))
            imageOnly = paramSet(args, PARAM_IMAGE_ONLY)
            exifOnly = paramSet(args, PARAM_EXIF_ONLY)
            trainingDataFormats = argForParam(args, PARAM_TRAINING_DATA_FORMAT, "flickr")
            optimizationCriteria = argForParam(args, PARAM_OPTIMIZATION_CRITERIA, "accuracy")
            permutations = int(argForParam(args, PARAM_EXIF_PERMUTATIONS, 50))
            
            # create dictionariy of { zipFilePath : dataformat }, either mulitple ones or single one
            zipFilePaths = {}
            if "," in datasetFilePath:
                filePaths = [path.strip() for path in datasetFilePath.split(",")]
                formats = [format.strip() for format in trainingDataFormats.split(",")] if "," in trainingDataFormats else [trainingDataFormats]
            
                for index in range(len(filePaths)):
                    currentFormat = ExifImageTrainingDataFormats.toTrainingDataFormat(name = formats[index] if len(formats) >= index else formats[-1])
                    if currentFormat is None:
                        raise ValueError("error: unknown training data format: " + currentFormat)
                    else:
                        zipFilePaths[filePaths[index]] = currentFormat
            else:
                targetFormat = ExifImageTrainingDataFormats.toTrainingDataFormat(name = trainingDataFormats)
                if targetFormat is None:
                    raise ValueError("error: unknown training data format: " + targetFormat)
                zipFilePaths[datasetFilePath] = targetFormat

            if exifOnly and imageOnly:
                raise ValueError("error: cannot set exif-only and image-only at the same time")

            # mandatory image args
            if not exifOnly:
                if all([paramSet(args, PARAM_IMAGE_SIZE), 
                        paramSet(args, PARAM_FINE_TUNE_EPOCHS), 
                        paramSet(args, PARAM_FINE_TUNE_LAYERS)]):
                    baseModelName = argForParam(args, PARAM_BASE_MODEL_TYPE, "all")
                    if not baseModelName in validModelNames() and not baseModelName == "all":
                        raise ValueError("error: the given base model name is invalid, possible values are " + str(validModelNames()))
                    imageSize = argForParam(args, PARAM_IMAGE_SIZE)
                    imageSize = imageSize.split(",") if not imageSize == None else []
                    if len(imageSize) != 2:
                        raise ValueError("error: invalid value for image size")
                    else:
                        imageSize = (int(imageSize[0]), int(imageSize[1]))
                    fineTuneEpochs = int(argForParam(args, PARAM_FINE_TUNE_EPOCHS))
                    fineTuneLayers = int(argForParam(args, PARAM_FINE_TUNE_LAYERS))
                else:
                    raise ValueError("error: not all mandatory parameters for cnn training are set")

            # optional exif args
            exifTags = argForParam(args, PARAM_EXIF_TAGS)
            exifTags = exifTags.split(",") if not exifTags == None else []
            
            # create data provider
            dataProvider = ExifImageProvider(dataSetsPaths = zipFilePaths,
                                             cachePath = cacheDirectoryPath,
                                             trainSize = trainSize,
                                             testSize = testSize,
                                             validationSize = valSize,
                                             batchSize = batchSize,
                                             imageSize = imageSize if not exifOnly else None,
                                             useSuperConcepts = useSuperConcepts,
                                             filterConcepts = filterConcepts,
                                             exifOnly = exifOnly,
                                             imageOnly = imageOnly,
                                             exifTags = exifTags,
                                             seed = seed)
            
            # create training task for given configuration
            tasks = []
            if exifOnly:
                tasks.append(
                    TrainingTask(classifier = MLPClassifier(name = name), 
                            storagePath = modelOutputPath, 
                            provider = dataProvider)
                )
            elif imageOnly:
                for modelName in standardModelNames() if baseModelName == "all" else [baseModelName]:
                    tasks.append(
                        TrainingTask(classifier = CNNClassifier(name = name + "_" + modelName,
                                                                 modelFunction = modelNameToModelFunction(modelName = modelName),
                                                                 fineTuneLayersCount = fineTuneLayers,
                                                                 fineTuneEpochs = fineTuneEpochs),
                                     storagePath = modelOutputPath, 
                                     provider = dataProvider)
                    )                    
            else:
                for modelName in standardModelNames() if baseModelName == "all" else [baseModelName]:
                    tasks.append(
                        TrainingTask(classifier = MixedModel(name = name + "_" + modelName, 
                                                      classifier1 = CNNClassifier(name = "CNN", modelFunction = modelNameToModelFunction(modelName = modelName)), 
                                                      classifier2 = MLPClassifier(name = "MLP"), 
                                                      fineTuneLayersCount = fineTuneLayers, 
                                                      fineTuneEpochs = fineTuneEpochs), 
                                    storagePath = modelOutputPath, 
                                    provider = dataProvider)
                    )

            # run training tasks
            tensorflow.get_logger().setLevel("ERROR")
            for task in tasks:
                task.run(epochs = epochs, earlyStoppingPatience = earlyStoppingPatienceEpochs, optimize = optimizationCriteria)

            if exifOnly:
                # create permutation task to assess the feature importance of individual exif features
                permutationTask = ExifPermutationTask(permutations = permutations, storagePath = modelOutputPath)
                permutationTask.run(classifier = task.classifier, dataGeneratorProvider = dataProvider)
                
        except Exception as exception:
            print(" ") 
            print(exception)
            print(" ")
