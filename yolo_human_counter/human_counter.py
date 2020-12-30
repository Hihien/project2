from .yolov5 import Detector

import numpy as np

__all__ = ['YoloHumanCounter']


class YoloHumanCounter(Detector):
    def __init__(self,
                 weights,
                 img_size=(640, 640),
                 conf_thresh=0.4,
                 iou_thresh=0.5,
                 agnostic_nms=False,
                 device='cpu',
                 human_class_id=0):
        super(YoloHumanCounter, self).__init__(weights, img_size, conf_thresh, iou_thresh, agnostic_nms, device)
        self.human_class_id = human_class_id

    def count(self, imgs, return_detection=False):
        single_input = isinstance(imgs, np.ndarray)
        if single_input:
            imgs = [imgs]
        dets = self.detect(imgs)
        dets = [det[det[:, -1].eq(self.human_class_id)] for det in dets]
        counts = [det.size(0) for det in dets]
        if single_input:
            counts, dets = counts[0], dets[0]
        if return_detection:
            return counts, dets
        return counts

    def __call__(self, imgs, return_detection=False):
        return self.count(imgs, return_detection=return_detection)
