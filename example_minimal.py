import cv2

from yolo_human_counter import YoloHumanCounter


if __name__ == '__main__':
    # initialize counter object that manage pre-trained weights internally
    counter = YoloHumanCounter("weights/yolov5m.pt")

    # read opencv image
    img = cv2.imread('images/project.jpg')
    # detect and count in 1 line
    num_human = counter(img)

    print('Number of human detected:', num_human)
