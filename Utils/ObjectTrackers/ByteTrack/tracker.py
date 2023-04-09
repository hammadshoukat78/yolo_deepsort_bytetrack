import numpy as np
import sys
sys.path.append("Utils/ObjectTrackers/ByteTrack")
from Helper.byte_tracker import BYTETracker


class ByteTracker():
    def __init__(self):
        self._track_thresh = 0.5
        self._track_buffer = 30
        self._match_thresh = 0.8
        self._frame_rate = 25
        self._tracker_obj = BYTETracker(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            mot20=False,
            frame_rate=25
        )

    def track(self, detections):
        # list of dict are coming here resolve into list of tuples
        results = [(item['bbox'], item['confidence'], item['class_name']) for item in detections]
        results = self._tracker_obj.update(detections=results)
        outputs = []
        for track in results:
            track_id = track.track_id
            bbox = track.tlwh
            bbox = bbox.tolist()
            bbox = [int(x) for x in bbox]
            det_class = track.det_class
            outputs.append({'bbox': bbox, 'tracking_id': track_id, 'object_name': det_class})

        return outputs
