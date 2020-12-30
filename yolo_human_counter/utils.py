import cv2

__all__ = ['plot_bboxes',
           'plot_count']


def plot_bboxes(img, det, label=None, color=(200, 200, 0), thickness=3):
    img = img.copy()
    for i, obj in enumerate(det.cpu().numpy()):
        x1, y1, x2, y2, conf, cls = obj[:6]
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(img, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        if label is not None:
            font_thickness = max(thickness // 3, 1)
            font_scale = max(thickness / 10, 0.2)
            text_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=font_thickness)[0]
            p2 = p1[0] + text_size[0], p1[1] - text_size[1] - 3
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (p1[0], p1[1] - 2), 0, font_scale, [225, 255, 255],
                        thickness=font_thickness, lineType=cv2.LINE_AA)
    return img


def plot_count(img, count, label=None, color=(0, 0, 0), thickness=1):
    overlay = img.copy()
    count_str = f'count: {count}'
    if label is not None:
        count_str = str(label) + ' ' + count_str
    text_size = cv2.getTextSize(count_str, 0, fontScale=0.5, thickness=thickness)[0]
    cv2.rectangle(overlay, (10, 10 + 10), (15 + text_size[0], 10 + 20 + text_size[1]), (255, 255, 255), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, count_str, (12, 10 + 15 + text_size[1]), 0, 0.5, color, thickness=thickness, lineType=cv2.LINE_AA)
    return img
