from collections.abc import Sequence

import torch

from .sort import Sort

__all__ = ['Tracker']


class Tracker:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age, min_hits, iou_threshold)

    def __call__(self, det):
        return self.update(det)

    def update(self, det):
        det = torch.nn.functional.pad(det, [0, 1])
        if len(det):
            tracks, filter_mask = self.tracker.update(det[:, :4])
            if len(filter_mask): det = det[[_ for _ in range(len(det)) if _ not in filter_mask]]
            if len(tracks):
                det[:, :4] = tracks[:, :4].clamp_min_(0)  # assign smoothed bounding boxes
                det[:, -1] = tracks[:, 4]  # assign track_ids
        return det
