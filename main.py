import cv2
import sys

from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Utils.ObjectTrackers.DeepSort.tracker import DeepSortTracker
from Utils.ObjectTrackers.ByteTrack.tracker import ByteTracker


detector_obj = YoloDetector(conf_thold=0.5,
                            device='cpu',
                            weights_path="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                            expected_objs=[0, 1, 2])
# deepsort_obj = DeepSortTracker(detector_obj.names)
bytetracker_obj = ByteTracker()

cap = cv2.VideoCapture("trafic.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = detector_obj.detect(frame)
    # outputs = deepsort_obj.track(pred=res, original_image=frame)
    outputs = bytetracker_obj.track(detections=res)
    print(outputs)
    for item in outputs:
        bbox = item['bbox']
        tracking_id = item['tracking_id']
        object_name = item['object_name']

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 225), 2)

        # Put text on the box
        text = f"{tracking_id} {object_name}"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
