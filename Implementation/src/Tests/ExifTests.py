from calendar import c
import unittest
from DomainModel.ExifData import ExifData
from DomainModel.Image import Image
from DomainModel.TrainingConcept import TrainingConcept
from DomainModel.TrainingData import EXIFImageTrainingData
from Transformers.Exif.ExifTagFilter import ExifTagFilter
from Transformers.Exif.ExifTagTransformer import ExifTagTransformer, TransformError, TransformError, transformContrast, transformFlash, transformFocalLength, transformFracture, transformFractureOrDecimal, transformNumber, transformOrientation, transformSaturation, transformSharpness
from unittest.mock import Mock

exifData = ExifData(tags = {
            "ExposureTime" : "1/10",
            "FNumber" : "2.8",
            "FocalLength" : "4.6 mm"
            }, id="12345")

class ExifDataTests(unittest.TestCase):

    def testExifDataProvidesCorrectInformation(self):
        self.assertTrue(exifData.id == "12345")
        self.assertTrue(exifData.tagNames == {'ExposureTime', 'FNumber', 'FocalLength'})
        self.assertTrue(exifData.tags == {'ExposureTime': '1/10', 'FNumber': '2.8', 'FocalLength': '4.6 mm'})
        self.assertTrue(exifData.containsTag(tag = "ExposureTime"))
        self.assertFalse(exifData.containsTag(tag = "UndefinedTag"))
        self.assertTrue(exifData.containsTags(tags = ["ExposureTime", "FNumber"]))
    
    def testExifDataTypes(self):
        testData = ExifData(tags = {
            "ExposureTime" : 0.3,
            "FNumber" : 2.8,
            "FocalLength" : 4.6,
            "Flash" : 1,
            "Orientation" : 4,
            "Saturation" : 1,
            "Sharpness" : 2,
            "Contrast" : 3 }, id = "1")
        
        numerical, categorical, binary = ExifData.typedTagLists(testData.tagNames)
        self.assertEqual(["Flash"], sorted(binary))
        self.assertEqual(["Contrast", "Orientation", "Saturation", "Sharpness"], sorted(categorical))
        self.assertEqual(["ExposureTime", "FNumber", "FocalLength"], sorted(numerical))


class ExifTagFilterTests(unittest.TestCase):

    def testExifTagFilterFiltersTagsCorrectly(self):
        filter = ExifTagFilter(filterTags = ["ExposureTime", "FNumber"])
        filtered = filter.transform(exifData)
        self.assertEqual({'ExposureTime': '1/10', 'FNumber': '2.8'}, filtered.tags)
    
    def testExifTagFilterReturnsNoneIfMissingValueIsFilteredButNoDummyValueSet(self):
        filter = ExifTagFilter(filterTags = ["ExposureTime", "UndefinedTag"])
        self.assertRaises(TransformError, filter.transform, exifData)

    def testExifTagFilterAppliesDummyValueForUnknownTag(self):
        filter = ExifTagFilter(filterTags = ["ExposureTime", "UndefinedTag"], dummyValue = "Test")
        filtered = filter.transform(exifData)
        self.assertEqual({'ExposureTime': '1/10', 'UndefinedTag': 'Test'}, filtered.tags)


class ExifTransformerTests(unittest.TestCase):

    def testExifTransformerTransformsTagsCorrectly(self):
        transformed = ExifTagTransformer.standard().transform(exifData)
        self.assertEqual({'ExposureTime': 0.1, 'FNumber': 2.8, "FocalLength" : 4.6}, transformed.tags, transformed.tags)
        self.assertEqual(transformed.id, exifData.id)
    
    def testExifTransformerThrowsErrorIfTransformerFunctionNotRegisteredForTag(self):
        testData = ExifData(tags = {
            "ExposureTime" : "1/10",
            "FNumber" : "2.8",
            "Unknowing" : "Unknowing"
        }, id="12345")
        self.assertRaises(TransformError, ExifTagTransformer.standard().transform, testData)
    
    def testExifTransformerThrowsErrorIfMalformedTagIsContained(self):
        testData = ExifData(tags = {
            "ExposureTime" : "1/10",
            "FNumber" : "2.8",
            "FocalLength" : "badTag"
        }, id="12345")
        self.assertRaises(TransformError, ExifTagTransformer.standard().transform, testData)

    def testExifTransformerNotTransformingWhitelistedValues(self):
        testData = ExifData(tags = {
            "ExposureTime" : "1/10",
            "FNumber" : "2.8",
            "FocalLength" : None
        }, id="12345")
        transformer = ExifTagTransformer.standard()
        transformer.addWhitelistedValue(value = None)
        transformer.addWhitelistedValue(value = "1/10")
        transformed = transformer.transform(data = testData)
        self.assertEqual([None, "1/10"], transformer.whiteList, transformer.whiteList)
        self.assertEqual({'ExposureTime': "1/10", 'FNumber': 2.8, "FocalLength" : None}, transformed.tags, transformed.tags)


class ExifTransformerFunctionsTests(unittest.TestCase):
    
    def testTransformNumericAndFractures(self):
        self.assertEqual(3.5, transformNumber("3.5"))
        self.assertRaises(TransformError, transformNumber, "a123")
        self.assertRaises(TransformError, transformNumber, "13mm")
        self.assertRaises(TransformError, transformNumber, None)

        self.assertEqual(0.05, transformFracture("1/20"))
        self.assertRaises(TransformError, transformFracture, "aa/1")
        self.assertRaises(TransformError, transformFracture, "3")
        self.assertRaises(TransformError, transformFracture, None)

        self.assertEqual(0.05, transformFractureOrDecimal("0.05"))
        self.assertEqual(0.05, transformFractureOrDecimal("1/20"))
        self.assertRaises(TransformError, transformFractureOrDecimal, "33d")
        self.assertRaises(TransformError, transformFractureOrDecimal, "33d/3")
        self.assertRaises(TransformError, transformFractureOrDecimal, None)

    def testTransformFocalLength(self):
        self.assertEqual(3.5, transformFocalLength("3.5 mm"))
        self.assertEqual(3.5, transformFocalLength("3.5mm"))
        self.assertEqual(3.5, transformFocalLength("3.5"))
        self.assertRaises(TransformError, transformFocalLength, "33mmde")
        self.assertRaises(TransformError, transformFocalLength, "33 cde")
        self.assertRaises(TransformError, transformFocalLength, None)
    
    def testTransformFlash(self):
        self.assertEqual(1, transformFlash("1"))
        self.assertEqual(1, transformFlash("on"))
        self.assertEqual(1, transformFlash("fired"))
        self.assertEqual(0, transformFlash("0"))
        self.assertEqual(0, transformFlash("off"))
        self.assertEqual(0, transformFlash("did not fire"))
        self.assertEqual(0, transformFlash("no flash"))
        self.assertRaises(TransformError, transformFlash, None)
    
    def testTransformSaturation(self):
        self.assertEqual(0, transformSaturation("low saturation"))
        self.assertEqual(1, transformSaturation("normal saturation"))
        self.assertEqual(2, transformSaturation("high saturation"))
        self.assertEqual(0, transformSaturation("low"))
        self.assertEqual(1, transformSaturation("normal"))
        self.assertEqual(2, transformSaturation("high"))
        self.assertRaises(TransformError, transformSaturation, "some saturation")
        self.assertRaises(TransformError, transformSaturation, None)
    
    def testTransformContrast(self):
        self.assertEqual(0, transformContrast("low contrast"))
        self.assertEqual(1, transformContrast("normal contrast"))
        self.assertEqual(2, transformContrast("high contrast"))
        self.assertEqual(0, transformContrast("low"))
        self.assertEqual(1, transformContrast("normal"))
        self.assertEqual(2, transformContrast("high"))
        self.assertRaises(TransformError, transformContrast, "some contrast")
        self.assertRaises(TransformError, transformContrast, None)
    
    def testTransformSharpness(self):
        self.assertEqual(0, transformSharpness("soft contrast"))
        self.assertEqual(1, transformSharpness("normal contrast"))
        self.assertEqual(2, transformSharpness("hard contrast"))
        self.assertEqual(0, transformSharpness("soft"))
        self.assertEqual(1, transformSharpness("normal"))
        self.assertEqual(2, transformSharpness("hard"))
        self.assertRaises(TransformError, transformSharpness, "some sharpness")
        self.assertRaises(TransformError, transformSharpness, None)
    
    def testTransformOrientation(self):
        self.assertEqual(0, transformOrientation("unknown"))
        self.assertEqual(1, transformOrientation("horizontal (normal)"))
        self.assertEqual(2, transformOrientation("mirror horizontal"))
        self.assertEqual(3, transformOrientation("rotate 180"))
        self.assertEqual(4, transformOrientation("mirror vertical"))
        self.assertEqual(5, transformOrientation("mirror horizontal and rotate 270 cw"))
        self.assertEqual(6, transformOrientation("rotate 90 cW"))
        self.assertEqual(7, transformOrientation("mirror horizontal and rotate 90 cw"))
        self.assertEqual(8, transformOrientation("rotate 270 cw"))
        self.assertRaises(TransformError, transformOrientation, "something")
        self.assertRaises(TransformError, transformOrientation, None)

class ExifTrainingDataTests(unittest.TestCase):

    def testTrainingConceptDoesNotContainDuplicateExifImageTrainingData(self):
        testImage1 = Mock(spec = Image)
        testImage1.id = "12345"

        # two duplicate instances
        trainingInstance1 = EXIFImageTrainingData(exifData = exifData, image = testImage1)
        trainingInstance2 = EXIFImageTrainingData(exifData = exifData, image = testImage1)
        trainingInstance3 = EXIFImageTrainingData(exifData = ExifData(tags = {"Test" : "1234" }), image = testImage1)
        trainingInstance4 = EXIFImageTrainingData(exifData = ExifData(tags = {"Test" : "1234" }), image = testImage1)

        concept = TrainingConcept(names = ("test"))
        concept.addTrainingInstances({ trainingInstance1, trainingInstance2, trainingInstance3, trainingInstance4 })
        self.assertEqual(2, len(concept.instances))

if __name__ == '__main__':
    unittest.main()