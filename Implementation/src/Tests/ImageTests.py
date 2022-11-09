from DomainModel.Image import Image
import unittest

class ImagefDataTests(unittest.TestCase):

    # Note: To successfully all the image tests, indoor / outdoor data set must be extracted within the resource folder
    def testImageDataProvidesCorrectInformation(self):
        path = "resources/indoor_outdoor_multilabel/indoor/indoor,bathroom/images/9531426769_01562a4e91_q.jpg"
        imageTest = Image(path, id="9531426769")

        self.assertTrue(imageTest.path.exists())
        self.assertEqual(imageTest.id, "9531426769")
        self.assertEqual(imageTest.path.name, "9531426769_01562a4e91_q.jpg")


if __name__ == '__main__':
    unittest.main()