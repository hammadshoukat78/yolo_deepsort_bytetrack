from ultralytics import YOLO
import cv2
import torch


class YoloDetector:
    def __init__(self, conf_thold, device, weights_path, expected_objs):
        # taking the image file path, confidence level and GPU or CPU selection
        self._conf_thres = conf_thold
        self._device = device
        self._model = YOLO(weights_path)
        self._classes = expected_objs
        self._names = self._model.names

    @property
    def names(self):
        return self._names

    def detect(self, image):
        results = self._model.predict(source=image, show=False, save=False, conf=self._conf_thres,
                                      device=self._device)
        for result in results:
            boxes = result.boxes.data
            list_of_dicts = []
            for row in boxes:
                bbox = row[:4].tolist()
                confidence = row[4].item()
                obj_id = int(row[5].item())

                # obj_name = 'car'  # Assuming all objects are cars; adjust as needed
                list_of_dicts.append({'bbox': bbox, 'class_id': obj_id, 'confidence': confidence,
                                      "class_name": self._names[obj_id]})

            return list_of_dicts

