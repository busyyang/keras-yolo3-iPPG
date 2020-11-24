import sys
import argparse
from yolo import YOLO
from PIL import Image


def iou(rec1, rec2):
    """
    rec1 and rec2 in (xmin,ymin,xmax,ymax)
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        print('IOU=0')
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def detect_img(yolo):
    import numpy as np
    import matplotlib.pyplot as plt
    with open('data/pulse_training_list.txt', 'r') as f:
        anns = f.readlines()
    anns = [ann.strip() for ann in anns]
    ious = []
    for ann in anns:
        line = ann.split(' ')
        img = line[0]
        line = line[1].split(',')
        bbox = [int(line[i]) for i in range(0, 4)]
        image = Image.open(img)
        r_image, rst_box = yolo.detect_image_IOU(image)
        if rst_box:
            _iou = iou(rst_box, bbox)
        else:
            print('没有检测到位置')
            _iou = 0
        ious.append(_iou)
        # r_image.show()
    yolo.close_session()
    print('mIOU={:.4f}\tmaxIOU={:.4f}\tminIOU={:.4f}'.format(np.mean(ious), np.max(ious), np.min(ious)))
    plt.plot(ious)
    plt.show()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if True:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
