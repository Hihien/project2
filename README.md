# Human detection and counting
## Dependencies
### Libraries:
- torch
- torchvision
- opencv-contrib-python

### Pretrained weights:
Download các file `.pt` về, tạo một folder tên `weights` và copy tất cả vào.

| Model (scale)         |                                 Download Link                           |
|-----------------------|:-----------------------------------------------------------------------:|
| yolov5s (small)       | https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt |
| yolov5m (medium)      | https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt |
| yolov5l (large)       | https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5l.pt |
| yolov5x (extra-large) | https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5x.pt |

## Installation
Đầu tiên clone repo về và mở terminal, cd vào folder `yolo_human_counter` (chứa file `setup.py`). Từ đó gõ lệnh:
```commandline
pip install .
```
Sau khi cài đặt thành công, thư viện có thể được import như một package bình thường.
```python
import yolo_human_counter
```

## Usage
Chạy example sử dụng camera:
```commandline
python example.py
```

### Programming
Dưới đây là *Minimal Working Example*:
```python
import cv2

from yolo_human_counter import YoloHumanCounter

# initialize counter object that manages pre-trained weights internally
counter = YoloHumanCounter('weights/yolov5m.pt')

# read opencv image
img = cv2.imread('images/project.jpg')
# detect and count in 1 line
num_human = counter(img)
```
