# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model, download_mmdet_yolox_tiny_model

MODEL_DEVICE = "cuda:0"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320
IMAGE_PATH = "tests/data/utg-UThong-m-mt.jpg"

MODEL_PATH = "tests\models\yolov5_s-p6-v62_syncbn_fast_4xb8-130e_asdr6-2-200-split82_1280_anchOptm.pth"
CONFIG_PATH = "tests\configs\yolov5\yolov5_s-p6-v62_syncbn_fast_4xb8-150e_asdr6-2-200-split82_1280_anchOptm.py"        

class TestMmdetDetectionModel(unittest.TestCase):
    # def test_load_model(self):
    #     from sahi.models.mmdetcustom import MmdetDetectionModelCustom

    #     mmdet_detection_model = MmdetDetectionModelCustom(
    #         model_path=MODEL_PATH,
    #         config_path=CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         device=MODEL_DEVICE,
    #         category_remapping=None,
    #         load_at_init=True,
    #     )

    #     self.assertNotEqual(mmdet_detection_model.model, None)

    # def test_perform_inference_with_mask_output(self):
    #     from sahi.models.mmdetcustom import MmdetDetectionModelCustom

    #     # init model
    #     mmdet_detection_model = MmdetDetectionModelCustom(
    #         model_path=MODEL_PATH,
    #         config_path=CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         device=MODEL_DEVICE,
    #         category_remapping=None,
    #         load_at_init=True,
    #         image_size=IMAGE_SIZE,
    #     )

    #     # prepare image
    #     image = read_image(IMAGE_PATH)

    #     # perform inference
    #     mmdet_detection_model.perform_inference(image)
    #     original_predictions = mmdet_detection_model.original_predictions

    #     # check actual prediction structures
    #     pred = original_predictions[0]
    #     self.assertTrue("bboxes" in pred)
    #     self.assertTrue("masks" in pred)
    #     self.assertTrue("scores" in pred)
    #     self.assertTrue("labels" in pred)

    #     # all annotations have the same length
    #     n_preds = len(pred["bboxes"])
    #     self.assertTrue(len(pred["bboxes"]) == n_preds)
    #     self.assertTrue(len(pred["masks"]) == n_preds)
    #     self.assertTrue(len(pred["labels"]) == n_preds)
    #     self.assertTrue(len(pred["scores"]) == n_preds)

    #     boxes = np.array(pred["bboxes"])
    #     scores = np.array(pred["scores"])

    #     # ensure all prediction scores are greater then 0.5
    #     idx = np.where(scores >= 0.5)[0]

    #     # compare
    #     self.assertTrue([446, 304, 490, 346] in boxes[idx].astype(int))

    # def test_perform_inference_without_mask_output(self):
    #     from sahi.models.mmdetcustom import MmdetDetectionModelCustom

    #     # init model
    #     mmdet_detection_model = MmdetDetectionModelCustom(
    #         model_path=MODEL_PATH,
    #         config_path=CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         device=MODEL_DEVICE,
    #         category_remapping=None,
    #         load_at_init=True,
    #     )

    #     # prepare image

    #     image = read_image(IMAGE_PATH)

    #     # perform inference
    #     mmdet_detection_model.perform_inference(image)
    #     original_predictions = mmdet_detection_model.original_predictions

    #     pred = original_predictions[0]
    #     n_preds = len(pred["bboxes"])
    #     self.assertTrue(len(pred["bboxes"]) == n_preds)
    #     self.assertTrue(len(pred["labels"]) == n_preds)
    #     self.assertTrue(len(pred["scores"]) == n_preds)
    #     boxes = np.array(pred["bboxes"].cpu())
    #     labels = np.array(pred["labels"].cpu())
    #     scores = np.array(pred["scores"].cpu())

    #     # find box of first component detection with conf greater than 0.5
    #     idx = np.where((scores >= 0.5) & (labels == 2))[0][0]

        # # compare
        # self.assertEqual(boxes[idx].astype(int).tolist(), [2424, 801, 2505, 844])

    # def test_convert_original_predictions_with_mask_output(self):
    #     from sahi.models.mmdet import MmdetDetectionModel

    #     # init model
    #     download_mmdet_cascade_mask_rcnn_model()

    #     mmdet_detection_model = MmdetDetectionModel(
    #         model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
    #         config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         device=MODEL_DEVICE,
    #         category_remapping=None,
    #         load_at_init=True,
    #         image_size=IMAGE_SIZE,
    #     )

    #     # prepare image
    #     image = read_image(IMAGE_PATH)

    #     # perform inference
    #     mmdet_detection_model.perform_inference(image)

    #     # convert predictions to ObjectPrediction list
    #     mmdet_detection_model.convert_original_predictions()
    #     object_predictions = mmdet_detection_model.object_prediction_list

    #     # compare
    #     self.assertEqual(len(object_predictions), 3)
    #     self.assertEqual(object_predictions[0].category.id, 2)
    #     self.assertEqual(object_predictions[0].category.name, "car")
    #     self.assertEqual(
    #         object_predictions[0].bbox.to_xywh(),
    #         [448, 308, 41, 36],
    #     )
    #     self.assertEqual(object_predictions[2].category.id, 2)
    #     self.assertEqual(object_predictions[2].category.name, "car")
    #     self.assertEqual(
    #         object_predictions[2].bbox.to_xywh(),
    #         [381, 280, 33, 30],
    #     )
    #     for object_prediction in object_predictions:
    #         self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    # def test_convert_original_predictions_without_mask_output(self):
    #     from sahi.models.mmdet import MmdetDetectionModel

    #     # init model
    #     download_mmdet_yolox_tiny_model()

    #     mmdet_detection_model = MmdetDetectionModel(
    #         model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
    #         config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         device=MODEL_DEVICE,
    #         category_remapping=None,
    #         load_at_init=True,
    #         image_size=IMAGE_SIZE,
    #     )

    #     # prepare image
    #     image = read_image(IMAGE_PATH)

    #     # perform inference
    #     mmdet_detection_model.perform_inference(image)

    #     # convert predictions to ObjectPrediction list
    #     mmdet_detection_model.convert_original_predictions()
    #     object_predictions = mmdet_detection_model.object_prediction_list

    #     # compare
    #     self.assertEqual(len(object_predictions), 3)
    #     self.assertEqual(object_predictions[0].category.id, 2)
    #     self.assertEqual(object_predictions[0].category.name, "car")
    #     np.testing.assert_almost_equal(object_predictions[0].bbox.to_xywh(), [448, 308, 41, 36], decimal=1)
    #     self.assertEqual(object_predictions[1].category.id, 2)
    #     self.assertEqual(object_predictions[1].category.name, "car")
    #     np.testing.assert_almost_equal(object_predictions[1].bbox.to_xywh(), [320, 327, 58, 36], decimal=1)
    #     self.assertEqual(object_predictions[2].category.id, 2)
    #     self.assertEqual(object_predictions[2].category.name, "car")
    #     np.testing.assert_almost_equal(object_predictions[2].bbox.to_xywh(), [381, 280, 33, 30], decimal=1)

    #     for object_prediction in object_predictions:
    #         self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    # # def test_perform_inference_without_mask_output_with_automodel(self):
    #     from sahi import AutoDetectionModel

    #     # init model
    #     download_mmdet_yolox_tiny_model()

    #     mmdet_detection_model = AutoDetectionModel.from_pretrained(
    #         model_type="mmdet",
    #         model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
    #         config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #         category_remapping=None,
    #         load_at_init=True,
    #         image_size=IMAGE_SIZE,
    #     )

    #     # prepare image
    #     image = read_image(IMAGE_PATH)

    #     # perform inference
    #     mmdet_detection_model.perform_inference(image)
    #     original_predictions = mmdet_detection_model.original_predictions

    #     pred = original_predictions[0]
    #     n_preds = len(pred["bboxes"])
    #     self.assertTrue(len(pred["bboxes"]) == n_preds)
    #     self.assertTrue(len(pred["labels"]) == n_preds)
    #     self.assertTrue(len(pred["scores"]) == n_preds)
    #     boxes = np.array(pred["bboxes"])
    #     labels = np.array(pred["labels"])
    #     scores = np.array(pred["scores"])

    #     # find box of first car detection with conf greater than 0.5
    #     idx = np.where((scores >= 0.5) & (labels == 2))[0][0]

    #     # compare
    #     self.assertEqual(boxes[idx].astype(int).tolist(), [320, 323, 380, 365])

    # #

    def test_example(self):
        # arrange an instance segmentation model for test
        # import required functions, classes
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction, predict
        from sahi.utils.cv import read_image
        from sahi.utils.file import download_from_url
        from sahi.utils.mmdet import download_mmdet_cascade_mask_rcnn_model, download_mmdet_config

        detection_model = AutoDetectionModel.from_pretrained(
        model_type='mmdetcustom',
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        confidence_threshold=0.5,
        image_size=1280,
        device="cuda", # 'cpu' or 'cuda:0'
        )

        result = get_prediction(IMAGE_PATH, detection_model)
        result.export_visuals(export_dir="demo_data/",text_size=1,rect_th=4)

        result = get_sliced_prediction(
        IMAGE_PATH,
        detection_model,
        slice_height = 320,
        slice_width = 320,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
        )
        result.export_visuals(export_dir="demo_sliced_data/",text_size=1,rect_th=4)

if __name__ == "__main__":
    unittest.main()
