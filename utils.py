import random

import cv2

__all__ = [
    'MeanEstimator',
    'plot_bboxes',
]

__colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(32)]


class MeanEstimator:
    def __init__(self):
        self.sum = self.mean = self.count = 0

    def update(self, val):
        self.count += 1
        if self.count == 1:
            self.sum = self.mean = val
        else:
            self.sum += val
            self.mean = self.sum / self.count
        return self.mean


def plot_bboxes(img, det, label=None, line_thickness=3):
    # Plots one bounding box on image img
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(det.cpu().numpy()):
        color = __colors[i % len(__colors)]
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(img, p1, p2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label is not None:
            font_thickness = max(line_thickness // 3, 1)
            font_scale = max(line_thickness / 10, 0.2)
            text_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=font_thickness)[0]
            p2 = p1[0] + text_size[0], p1[1] - text_size[1] - 3
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (p1[0], p1[1] - 2), 0, font_scale, [225, 255, 255],
                        thickness=font_thickness, lineType=cv2.LINE_AA)
