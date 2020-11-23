import time

import cv2
import torch

from tqdm import tqdm

from sort import Tracker
from yolov5 import Detector
from utils import *


def main():
    print('Connecting to camera')
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Unable to connect to camera!'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Loading models')
    detector = Detector('weights/yolov5s.pt', img_size=(640, 640),
                        conf_thresh=0.4, iou_thresh=0.5, agnostic_nms=False,
                        device=device)
    tracker = Tracker(max_age=10, min_hits=3, iou_threshold=0.5)
    person_cls_id = detector.names.index('person')  # get id of 'person' class
    fps_estimator = MeanEstimator()

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Starting capture, camera_fps={cam_fps}')

    # Start of demo
    win_name = 'Camera Pi Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(win_name, width, height)
    pbar = tqdm(desc=f'[{win_name}]')

    while True:
        start_it = time.time()
        ret, img = cap.read()
        if not ret:
            print('Unable to read camera')
            break
        detections = detector.detect([img])[0]

        num_people = 0
        detections = detections[detections[:, -1].eq(person_cls_id)]  # filter person
        detections = tracker(detections)  # add track_id
        num_people = len(detections)
        # draw detections
        plot_bboxes(img, detections,
                    label='Person',
                    line_thickness=5)
        # draw counting
        overlay = img.copy()
        count_str = f'Number of people: {num_people}'
        text_size = cv2.getTextSize(count_str, 0, fontScale=0.5, thickness=1)[0]
        cv2.rectangle(overlay, (10, 10 + 10), (15 + text_size[0], 10 + 20 + text_size[1]), (255, 255, 255), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.putText(img, count_str, (12, 10 + 15 + text_size[1]), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # show
        cv2.imshow(win_name, img)
        elapsed_time = time.time() - start_it
        fps = fps_estimator.update(1 / elapsed_time)
        pbar.set_description(f'[{win_name}] num_detections={num_people} fps={fps:.02f} elapsed_time={elapsed_time:.03f}')
        pbar.update(1)

        # check key pressed
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # q or esc to quit
            break
        elif key == 32:  # space to pause
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
