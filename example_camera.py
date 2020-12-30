import time

import sys
import cv2
import torch

from tqdm import tqdm

from yolo_human_counter import YoloHumanCounter
from yolo_human_counter import plot_bboxes, plot_count


def main():
    print('Connecting to camera')
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Unable to connect to camera!'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Loading models')
    human_counter = YoloHumanCounter('weights/yolov5s.pt', img_size=(640, 640),
                                     conf_thresh=0.4, iou_thresh=0.5, agnostic_nms=False,
                                     device=device)

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Starting capture, camera_fps={int(cap.get(cv2.CAP_PROP_FPS))}')

    # Start of demo
    win_name = 'Camera Pi Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(win_name, width, height)
    pbar = tqdm(desc=f'[{win_name}]', file=sys.stdout)

    while True:
        start_it = time.time()
        ret, img = cap.read()
        if not ret:
            print('Unable to read camera')
            break
        num_people, det = human_counter(img, return_detection=True)

        # visualize
        img = plot_bboxes(img, det,
                          label='Person',
                          thickness=5)
        img = plot_count(img, num_people,
                         label='Person',
                         thickness=5)

        # show
        cv2.imshow(win_name, img)
        elapsed_time = time.time() - start_it
        pbar.set_description(f'[{win_name}] num_detections={num_people} elapsed_time={elapsed_time:.03f}')
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
