import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import json
from typing import Tuple
from coco_utils import get_image_infos
from matplotlib import pyplot as plt
import os
import pdb
import shutil
import numpy as np
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import glob
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='/home/ameneh/UofA/git/objects_counting_dmap_old/test_coco_viz')
parser.add_argument('--test_dir', type=str, default='test')

def rotate_box(corners, angle=-90, cx=512, cy=1232, h=2464, w=1024):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated
def xywh2xyxy(x, y, w, h) -> Tuple[int]:
    '''convert bbox format from (x, y, w, h) to (x0, y0, x1, y1)'''
    return int(x), int(y), int(x + w), int(y + h)


def center_xy(x, y, w, h) -> Tuple[int]:
    return int(x + (w // 2)), int(y + (h // 2))


def save_img_gt(source, annotation_path, out):
    annotations = json.load(open(annotation_path, 'r'))
    os.makedirs(Path(out, 'images'), exist_ok=True)
    os.makedirs(Path(out, 'ground_truth_points'), exist_ok=True)

    #rotate
    old_image_path = os.path.join(source, 'images')
    Iwidth = annotations['images'][0]['width']
    Iheight = annotations['images'][0]['height']
    new_json_data = annotations
    points =[]
    for number, annt in enumerate(new_json_data['annotations']):
        x, y, w, h = annt['bbox']
        corner = np.hstack((x, y, x + w, y, x, y + h, x + w, y + h))
        # pdb.set_trace()
        new_c = rotate_box(corner, angle=90, cx=Iwidth // 2, cy=Iheight // 2, h=Iheight, w=Iwidth)[0]

        new_json_data['annotations'][number]['bbox'] = [new_c[0], new_c[1], new_c[-2] - new_c[0],
                                                        new_c[-1] - new_c[1]]


    print(len(new_json_data['images']))
    for number, images in enumerate(new_json_data['images']):
        new_json_data['images'][number]['width'] = Iheight
        new_json_data['images'][number]['height'] = Iwidth
        image = Image.open(os.path.join(old_image_path, images['file_name']))
        img = image.rotate(90, expand=True)

        frame_name = images['file_name']
        img_path = Path(out, 'images', Path(source).name + f'_{frame_name}')
        new_json_data['images'][number]['file_name'] = '{}'.format(Path(source).name + f'_{frame_name}')
        try:
            img.save(img_path)
        except:
            pdb.set_trace()


    img_infos = get_image_infos(new_json_data)
    # print(args.n)
    for i, img_info in enumerate(img_infos):

        points = []
        for ann_info in img_info.anns:
            # BBOX here!

            if ann_info.category_id == 1:
                x, y = center_xy(*ann_info.bbox)
                points.append((x, y))
        img_path = Path(out, 'images', img_info.file_name)
        np.save(str(img_path).replace('.PNG', '.npy').replace('images', 'ground_truth_points'), points)

    return new_json_data

def main():
    coco = None
    val_coco = None
    for source_ in data2_sources:
        print(source_)
        #source_ = Path(ROOT_DIR, data)
        new_json_data_t = save_img_gt(source_, os.path.join(source_, 'annotations', "instances_default.json"), TEST_SAVE_DIR)
        #TODO save new json

if __name__ == "__main__":
    args = parser.parse_args()
    ROOT_DIR = args.root

    TEST_SAVE_DIR = args.test_dir
    data2_sources = []
    f = glob.glob(os.path.join(ROOT_DIR, '*'))
    
    for i in f:
        if os.path.isdir(i):
            list_sub =  os.listdir(i)
            
            if 'images' in list_sub:
                data2_sources.append(i)
            else:
                for sub in list_sub:
                    if os.path.isdir(os.path.join(i,sub)):
                        data2_sources.append(os.path.join(i,sub))
    print(TEST_SAVE_DIR)
    print(data2_sources)
    print(ROOT_DIR)

    main()

