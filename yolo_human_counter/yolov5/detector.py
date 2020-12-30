import os

from .detection_helpers import *
from .models.experimental import attempt_load

__all__ = ['Detector']


class Detector:
    def __init__(self,
                 weights,
                 img_size=(640, 640),
                 conf_thresh=0.4,
                 iou_thresh=0.5,
                 agnostic_nms=False,
                 device='cpu'):
        self.weights = os.path.abspath(weights).replace(os.sep, '/')
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)
        self.agnostic_nms = agnostic_nms
        self.model = attempt_load(weights, map_location=self.device)
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.device.type == 'cuda':  # warm-up
            self.model(torch.rand(1, 3, self.img_size[1], self.img_size[0], device=self.device))

    @torch.no_grad()
    def detect(self, im0s):
        imgs = []
        ratio_pads = []
        for im0 in im0s:
            # Convert
            img, gain, pad = letterbox(im0, new_shape=self.img_size)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(np.expand_dims(img, 0))
            imgs.append(img)
            ratio_pads.append((gain, pad))
        imgs = torch.from_numpy(np.concatenate(imgs)).float().to(self.device).div(255)

        # Inference
        detections = self.model(imgs)[0].cpu()
        detections = non_max_suppression(detections, self.conf_thresh, self.iou_thresh, agnostic=self.agnostic_nms)
        # Process detections
        for i in range(len(detections)):  # detections per image
            im0 = im0s[i]
            det = detections[i]
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(det[:, :4], imgs.shape[2:], im0.shape)
                detections[i] = det
        return detections

    def __call__(self, im0s):
        return self.detect(im0s)

    def __repr__(self):
        rep = f'{self.__class__.__name__}('
        rep += f'weights="{self.weights}"'
        rep += f', img_size={self.img_size}'
        rep += f', conf_thresh={self.conf_thresh}'
        rep += f', iou_thresh={self.iou_thresh}'
        rep += f', agnostic_nms={self.agnostic_nms}'
        rep += f', device={self.device})'
        return rep
