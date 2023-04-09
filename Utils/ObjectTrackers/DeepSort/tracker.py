import os
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.append('C:\\Users\\hammad\\Desktop\\yolo\\Utils\\ObjectTracker')

from Helper.deep_sort.deep_sort import DeepSort
from Helper.deep_sort.utils.parser import get_config


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


class DeepSortTracker():
    def __init__(self, names):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("Utils\\ObjectTrackers\\DeepSort\\Helper\\deep_sort\\configs\\deep_sort.yaml")
        self._deepsort = DeepSort('osnet_x0_25',
                                  max_dist=cfg.DEEPSORT.MAX_DIST,
                                  max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                  max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                  nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                  use_cuda=True)
        self._names = names

    def track(self, pred, original_image):
        # Process detections
        detections_per_frame = []

        for item in pred:
            bbox = item['bbox']
            confidence = item['confidence']
            obj_class = item['class_id']
            detections_per_frame.append(bbox + [confidence, obj_class])

        det_in_tensor = [torch.tensor(detections_per_frame)]
        for i, det in enumerate(det_in_tensor):
            # print(det)
            if det is not None and len(det):
                det = det[:, :6]
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # print(xywhs, confs, clss)
                # pass detections to deepsort
                # print(type(im0))
                outputs = self._deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), original_image)
                # print("deepsort predictions : ", outputs)
                predictions = []
                for i in range(len(outputs)):
                    # print(outputs[i])
                    bbox = outputs[i][0:4]
                    bbox = bbox.tolist()
                    bbox = [int(x) for x in bbox]
                    tracking_id = outputs[i][4]
                    obj_id = outputs[i][-1]
                    class_names = self._names
                    obj_name = class_names[obj_id]
                    predictions.append({'bbox': bbox, 'tracking_id': tracking_id, 'object_name': obj_name})

                return predictions
